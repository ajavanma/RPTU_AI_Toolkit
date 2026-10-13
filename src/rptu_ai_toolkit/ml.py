"""Training and point-level inference on validated, portable prepared datasets.

PyTorch and safetensors are optional and imported only when a model workflow is
invoked. A bundle binds weights to the feature, voxel and label contract used
for training; loading it never executes a pickle payload.
"""

import importlib.metadata
from itertools import chain
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

from . import __version__
from .artifacts import new_output_directory, write_json, write_npz
from .datasets import _fields, _float, _nonfinite, _object, sha256_file
from .models import MLRuntimeError, build_model, resolve_runtime
from .prepared import load_prepared_manifest, load_prepared_scene
from .checkpoints import load_checkpoint, save_checkpoint
from .runs import TrainingRun, read_completion, read_run_spec


_ARCHITECTURES = {"torch": "feature-mlp-64-64-v1", "warpconvnet": "minkunet18-v1"}
_SPLITS = ("train", "val", "test", "predict")


def _dependencies():
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise MLRuntimeError("Model workflows require the ml extra. Install from source with: python -m pip install '.[ml]'") from error
    return torch, load_file, save_file


def _positive_integer(value, name):
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise MLRuntimeError(f"{name}: expected a positive integer no greater than 2147483647")
    return value


def _check_output(output):
    try:
        if os.path.lexists(Path(output).expanduser()):
            raise MLRuntimeError(f"{output}: output already exists; choose a new path")
    except (TypeError, OSError, RuntimeError) as error:
        raise MLRuntimeError(f"{output}: invalid output path: {error}") from error


def _contract(dataset):
    return {
        "coordinate_unit": dataset.coordinate_unit,
        "feature_names": list(dataset.feature_names),
        "voxel_size": dataset.voxel_size,
        "label_mapping": [
            {"dataset_id": entry.id, "train_id": index, "name": entry.name}
            for index, entry in enumerate(dataset.classes)
        ],
        "ignore_label": dataset.ignore_label,
        "training_ignore_label": -1,
        "quantization": "floor",
        "feature_reduction": "mean",
        "label_reduction": "majority_nonignored_then_smallest_train_id",
    }


class _SceneSelection:
    """Reiterable scene metadata; arrays are loaded one scene at a time."""

    def __init__(self, dataset, scenes):
        self.dataset = dataset
        self.scenes = tuple(scenes)

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        for scene in self.scenes:
            yield scene, load_prepared_scene(self.dataset, scene)


def _selected_scenes(dataset, split, max_voxels, *, required=True):
    if split not in _SPLITS:
        raise MLRuntimeError(f"split: expected one of {', '.join(_SPLITS)}")
    _positive_integer(max_voxels, "max_voxels")
    scenes = [scene for scene in dataset.scenes if scene.split == split]
    if required and not scenes:
        raise MLRuntimeError(f"prepared dataset has no {split!r} scenes")
    for scene in scenes:
        if scene.voxel_count > max_voxels:
            raise MLRuntimeError(
                f"scene {scene.id!r}: {scene.voxel_count} voxels exceeds max_voxels={max_voxels}; "
                "increase the voxel size or explicitly raise --max-voxels"
            )
    return _SceneSelection(dataset, scenes)


def _normalization(scenes):
    # Merge per-scene population moments, keeping validation/test data entirely
    # outside the fitted statistics. Reject overflow rather than storing NaNs.
    count, mean, variance = 0, None, None
    with np.errstate(over="ignore", invalid="ignore"):
        for _, sample in scenes:
            batch_count = len(sample.features)
            batch_mean = sample.features.mean(axis=0)
            batch_variance = sample.features.var(axis=0)
            if count == 0:
                mean, variance = batch_mean, batch_variance
            else:
                fraction = batch_count / (count + batch_count)
                delta = batch_mean - mean
                variance = (1 - fraction) * variance + fraction * batch_variance + fraction * (1 - fraction) * delta**2
                mean = mean + fraction * delta
            count += batch_count
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise MLRuntimeError("training features overflow normalization; rescale the input features")
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    return {"kind": "training-voxel-zscore-v1", "mean": mean.tolist(), "scale": scale.tolist()}


def _normalized_features(sample, normalization):
    with np.errstate(over="ignore", invalid="ignore"):
        features = ((sample.features - np.asarray(normalization["mean"])) / np.asarray(normalization["scale"])).astype(np.float32)
    if not np.isfinite(features).all():
        raise MLRuntimeError("normalized features must be finite float32 values; rescale input features")
    return np.ascontiguousarray(features)


def _inputs(sample, normalization, device, torch):
    return (
        torch.from_numpy(np.ascontiguousarray(sample.voxel_coords)).to(device=device, dtype=torch.int64),
        torch.from_numpy(_normalized_features(sample, normalization)).to(device=device),
    )


def _logits(model, sample, normalization, device, classes, torch):
    result = model(*_inputs(sample, normalization, device, torch))
    if not isinstance(result, torch.Tensor) or result.shape != (len(sample.features), classes):
        raise MLRuntimeError("model returned logits with an unexpected shape")
    if not result.is_floating_point() or not torch.isfinite(result).all().item():
        raise MLRuntimeError("model returned nonfinite logits")
    return result


def _runtime(device, backend, torch):
    return {
        "toolkit_version": __version__,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "warpconvnet_version": importlib.metadata.version("warpconvnet") if backend == "warpconvnet" else None,
        "device": device,
    }


def _point_predictions(sample, voxel_predictions, dataset):
    valid = sample.point_to_voxel >= 0
    train_labels = np.full(len(valid), -1, dtype=np.int64)
    train_labels[valid] = voxel_predictions[sample.point_to_voxel[valid]]
    inverse = np.asarray([entry.id for entry in dataset.classes], dtype=np.int64)
    labels = np.full(len(valid), dataset.ignore_label, dtype=np.int64)
    labels[valid] = inverse[train_labels[valid]]
    arrays = {
        "point_rows": np.arange(len(valid), dtype=np.int64),
        "point_indices": sample.point_indices,
        "labels": labels,
        "valid_mask": valid,
    }
    if sample.point_labels is not None:
        targets = np.full(len(valid), dataset.ignore_label, dtype=np.int64)
        target_valid = sample.point_labels >= 0
        targets[target_valid] = inverse[sample.point_labels[target_valid]]
        arrays["target_labels"] = targets
    return arrays, train_labels


def _metrics(confusion, dataset, total_points, valid_points, scored_points):
    true_positive = np.diag(confusion)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - true_positive
    ious = [float(tp / count) if count else None for tp, count in zip(true_positive, union)]
    present = [value for value in ious if value is not None]
    return {
        "total_points": total_points,
        "valid_points": valid_points,
        "scored_points": scored_points,
        "excluded_points": total_points - scored_points,
        "confusion_matrix": confusion.tolist(),
        "confusion_axes": {"rows": "target_train_id", "columns": "predicted_train_id"},
        "classes": [
            {"dataset_id": label.id, "train_id": index, "name": label.name, "iou": ious[index]}
            for index, label in enumerate(dataset.classes)
        ],
        "accuracy": float(true_positive.sum() / scored_points) if scored_points else None,
        "mean_iou": float(np.mean(present)) if present else None,
    }


def _infer(scenes, dataset, model, normalization, device, torch, *, on_prediction=None):
    classes = len(dataset.classes)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    total = valid = scored = 0
    model.eval()
    with torch.no_grad():
        for scene, sample in scenes:
            predicted = _logits(model, sample, normalization, device, classes, torch).argmax(dim=1).cpu().numpy()
            arrays, train_labels = _point_predictions(sample, predicted, dataset)
            total += len(train_labels)
            valid += int(arrays["valid_mask"].sum())
            if sample.point_labels is not None:
                included = arrays["valid_mask"] & (sample.point_labels >= 0)
                scored += int(included.sum())
                np.add.at(confusion, (sample.point_labels[included], train_labels[included]), 1)
            if on_prediction is not None:
                on_prediction(scene, arrays)
    return _metrics(confusion, dataset, total, valid, scored)


def _read_training_report(path, dataset, history, backend, device, has_validation):
    try:
        report = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                            parse_constant=_nonfinite, parse_float=_float)
        _fields(report, {"schema_version", "kind", "backend", "device", "architecture", "epochs", "train", "val"}, "training report")
        for key, expected in {
            "schema_version": 1, "kind": "rptu-training-report", "backend": backend,
            "device": device, "architecture": _ARCHITECTURES[backend], "epochs": history,
        }.items():
            if json.dumps(report[key], sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise MLRuntimeError(f"completed training report {key}: does not match its run/checkpoint")
        for split in ("train", "val"):
            metrics = report[split]
            if split == "val" and not has_validation:
                if metrics is not None:
                    raise MLRuntimeError("training report val: expected null without a validation split")
                continue
            _fields(metrics, {"total_points", "valid_points", "scored_points", "excluded_points", "confusion_matrix",
                              "confusion_axes", "classes", "accuracy", "mean_iou"}, "training report metrics")
            counts = [metrics[key] for key in ("total_points", "valid_points", "scored_points")]
            if any(type(count) is not int or not 0 <= count <= 2**63 - 1 for count in counts):
                raise MLRuntimeError("training report metrics: expected nonnegative integer point counts")
            total, valid, scored = counts
            if not scored <= valid <= total:
                raise MLRuntimeError("training report metrics: inconsistent point counts")
            confusion = metrics["confusion_matrix"]
            classes = len(dataset.classes)
            if (not isinstance(confusion, list) or len(confusion) != classes
                    or any(not isinstance(row, list) or len(row) != classes for row in confusion)
                    or any(type(value) is not int or not 0 <= value <= 2**63 - 1 for row in confusion for value in row)
                    or sum(sum(row) for row in confusion) != scored):
                raise MLRuntimeError("training report confusion_matrix: invalid dimensions or counts")
            expected = _metrics(np.asarray(confusion, dtype=np.int64), dataset, total, valid, scored)
            if json.dumps(metrics, sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise MLRuntimeError("training report metrics: inconsistent scores/class mapping")
        return report
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
        if isinstance(error, MLRuntimeError):
            raise
        raise MLRuntimeError(f"cannot load completed training report: {error}") from error


def train_model(prepared, output, *, device="auto", backend="auto", epochs=None,
                learning_rate=None, seed=None, max_voxels=None, resume=False) -> dict:
    """Train or resume the same immutable run, retaining completed epoch state."""
    if type(resume) is not bool:
        raise MLRuntimeError("resume: expected a boolean")
    saved = read_run_spec(output) if resume else None
    if not resume:
        _check_output(output)
    defaults = {"epochs": 20, "learning_rate": 0.001, "seed": 0, "max_voxels": 100000}
    supplied = {"epochs": epochs, "learning_rate": learning_rate, "seed": seed, "max_voxels": max_voxels}
    try:
        settings = {key: value if value is not None else (saved["training"][key] if saved is not None else defaults[key])
                    for key, value in supplied.items()}
        if saved is not None:
            if backend == "auto":
                backend = saved["backend"]
            if device == "auto":
                device = saved["runtime"]["device"]
    except (KeyError, TypeError) as error:
        raise MLRuntimeError("run specification: missing training settings/runtime") from error
    if not isinstance(backend, str) or backend not in ("auto", "torch", "warpconvnet"):
        raise MLRuntimeError("run backend: expected auto, torch or warpconvnet")
    if not isinstance(device, str) or device not in ("auto", "cpu", "cuda"):
        raise MLRuntimeError("run device: expected auto, cpu or cuda")
    epochs, learning_rate, seed, max_voxels = (settings[key] for key in defaults)
    _positive_integer(epochs, "epochs")
    _positive_integer(max_voxels, "max_voxels")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise MLRuntimeError("seed: expected an integer in 0..4294967295")
    _finite_list([learning_rate], 1, "learning_rate", positive=True)
    dataset = load_prepared_manifest(prepared)
    training = _selected_scenes(dataset, "train", max_voxels)
    validation = _selected_scenes(dataset, "val", max_voxels, required=False)
    if not any(np.any(sample.labels >= 0) for _, sample in training):
        raise MLRuntimeError("train split has no nonignored training labels")
    normalization = _normalization(training)
    # Validate source hashes and all used float32 conversions before any writes.
    for _, sample in chain(training, validation):
        _normalized_features(sample, normalization)
    torch, load_file, save_file = _dependencies()
    device, backend = resolve_runtime(device=device, backend=backend)
    spec = {
        "schema_version": 1, "kind": "rptu-training-specification", "backend": backend,
        "architecture": _ARCHITECTURES[backend], "input_channels": len(dataset.feature_names),
        "num_classes": len(dataset.classes), "contract": _contract(dataset), "normalization": normalization,
        "training": {
            "seed": seed, "epochs": epochs, "learning_rate": float(learning_rate),
            "max_voxels": max_voxels, "prepared_sha256": sha256_file(dataset.source),
            "train_scene_ids": [scene.id for scene in training.scenes],
        },
        "optimizer": "Adam", "runtime": _runtime(device, backend, torch),
        "optimizer_steps_per_epoch": sum(bool(np.any(sample.labels >= 0)) for _, sample in training),
        "torch_threads": torch.get_num_threads(),
    }
    # Compare before acquiring a run or appending attempt metadata. The locked
    # comparison below also protects against a run.json change during validation.
    if saved is not None and json.dumps(saved, sort_keys=True) != json.dumps(spec, sort_keys=True):
        def differences(before, after, prefix=""):
            if isinstance(before, dict) and isinstance(after, dict):
                result = []
                for key in sorted(before.keys() | after.keys()):
                    field = f"{prefix}.{key}" if prefix else key
                    if key not in before or key not in after:
                        result.append(field)
                    else:
                        result.extend(differences(before[key], after[key], field))
                return result
            return [] if json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True) else [prefix]
        changed = ", ".join(differences(saved, spec))
        raise MLRuntimeError(f"resume specification differs from the original run: {changed}")
    rng_devices = [torch.cuda.current_device()] if device == "cuda" else []
    try:
        with torch.random.fork_rng(devices=rng_devices):
            # Seed only RNGs covered by fork_rng; don't alter unrelated GPUs.
            torch.random.default_generator.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
            model = build_model(backend, len(dataset.feature_names), len(dataset.classes), device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
            with TrainingRun(output, resume=resume) as run:
                run.bind_spec(spec)
                history = load_checkpoint(run, model, optimizer, torch, load_file, device, epochs)
                completion = run.completed()
                if completion is not None:
                    if history is None or len(history) != epochs:
                        raise MLRuntimeError("completed run must reference its final epoch checkpoint")
                    # Inference validation checks the final model's numeric state.
                    _load_model(run.path, dataset, device)
                    if json.dumps(completion["summary"], sort_keys=True) != json.dumps({"epochs": epochs}, sort_keys=True):
                        raise MLRuntimeError("completed run summary does not match its epoch target")
                    return _read_training_report(run.path / "training.json", dataset, history, backend, device, bool(validation))
                with run.attempt():
                    if history is None:
                        history = []
                        save_checkpoint(run, model, optimizer, history, torch, save_file, device)
                    for epoch in range(len(history), epochs):
                        model.train()
                        total_loss, total_labels = 0.0, 0
                        for _, sample in training:
                            supervised = int((sample.labels >= 0).sum())
                            if not supervised:
                                continue
                            optimizer.zero_grad(set_to_none=True)
                            logits = _logits(model, sample, normalization, device, len(dataset.classes), torch)
                            labels = torch.from_numpy(np.ascontiguousarray(sample.labels)).to(device=device, dtype=torch.int64)
                            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-1)
                            if not torch.isfinite(loss).item():
                                raise MLRuntimeError("training produced a nonfinite loss")
                            loss.backward()
                            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()):
                                raise MLRuntimeError("training produced nonfinite gradients; lower the learning rate")
                            optimizer.step()
                            total_loss += float(loss.detach().cpu()) * supervised
                            total_labels += supervised
                        history.append({"epoch": epoch + 1, "loss": total_loss / total_labels})
                        save_checkpoint(run, model, optimizer, history, torch, save_file, device)
                    training_metrics = _infer(training, dataset, model, normalization, device, torch)
                    validation_metrics = _infer(validation, dataset, model, normalization, device, torch)
                    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
                    if any(value.is_floating_point() and not torch.isfinite(value).all().item() for value in state.values()):
                        raise MLRuntimeError("training produced nonfinite model weights")
                    bundle = {
                        "schema_version": 1, "kind": "rptu-segmentation-model", "backend": backend,
                        "architecture": _ARCHITECTURES[backend], "input_channels": len(dataset.feature_names),
                        "num_classes": len(dataset.classes), "contract": _contract(dataset),
                        "normalization": normalization, "training": spec["training"], "runtime": spec["runtime"],
                    }
                    report = {
                        "schema_version": 1, "kind": "rptu-training-report", "backend": backend,
                        "device": device, "architecture": _ARCHITECTURES[backend],
                        "epochs": history, "train": training_metrics,
                        "val": validation_metrics if validation else None,
                    }
                    staging = run.path / f".publish-{uuid.uuid4().hex}"
                    staging.mkdir()
                    try:
                        weights = staging / "weights.safetensors"
                        save_file(state, str(weights))
                        bundle["weights"] = {"path": "weights.safetensors", "sha256": sha256_file(weights)}
                        write_json(staging / "training.json", report)
                        write_json(staging / "bundle.json", bundle)
                        run.publish(staging, {"epochs": epochs})
                    finally:
                        shutil.rmtree(staging)
    except RuntimeError as error:
        if isinstance(error, MLRuntimeError):
            raise
        raise MLRuntimeError(f"training failed: {error}") from error
    return report


def _finite_list(value, count, location, *, positive=False):
    if not isinstance(value, list) or len(value) != count:
        raise MLRuntimeError(f"{location}: expected {count} numbers")
    for item in value:
        if type(item) not in (int, float):
            raise MLRuntimeError(f"{location}: expected finite numbers")
        try:
            number = float(item)
        except (OverflowError, ValueError) as error:
            raise MLRuntimeError(f"{location}: expected finite numbers") from error
        if not math.isfinite(number) or (positive and number <= 0):
            raise MLRuntimeError(f"{location}: expected finite {'positive ' if positive else ''}numbers")


def _checksum(value, location):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MLRuntimeError(f"{location}: expected a lowercase SHA-256 checksum")


def _read_bundle(path, dataset):
    try:
        directory = Path(path).expanduser().resolve()
        if os.path.lexists(directory / "run.json") and read_completion(directory) is None:
            raise MLRuntimeError("training run is incomplete; resume training before loading its model")
        source = directory / "bundle.json"
        if not source.resolve().is_relative_to(directory):
            raise MLRuntimeError("bundle.json must remain inside the model directory")
        document = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_object,
                              parse_constant=_nonfinite, parse_float=_float)
        root = _fields(document, {
            "schema_version", "kind", "backend", "architecture", "input_channels", "num_classes",
            "contract", "normalization", "weights", "training", "runtime",
        }, "model bundle")
        if type(root["schema_version"]) is not int or root["schema_version"] != 1:
            raise MLRuntimeError("model bundle schema_version: expected integer 1")
        if root["kind"] != "rptu-segmentation-model":
            raise MLRuntimeError("model bundle kind: expected 'rptu-segmentation-model'")
        if root["backend"] not in _ARCHITECTURES or root["architecture"] != _ARCHITECTURES[root["backend"]]:
            raise MLRuntimeError("model bundle architecture: unsupported architecture/backend combination")
        for key, expected in (("input_channels", len(dataset.feature_names)), ("num_classes", len(dataset.classes))):
            if type(root[key]) is not int or root[key] != expected:
                raise MLRuntimeError(f"model bundle {key}: does not match the prepared dataset")
        contract = _fields(root["contract"], set(_contract(dataset)), "model bundle contract")
        # Canonical JSON comparison also rejects boolean/integer substitutions.
        for key, expected in _contract(dataset).items():
            if json.dumps(contract[key], sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise MLRuntimeError(f"model bundle contract.{key}: does not match the prepared dataset")
        normalization = _fields(root["normalization"], {"kind", "mean", "scale"}, "model normalization")
        if normalization["kind"] != "training-voxel-zscore-v1":
            raise MLRuntimeError("model normalization.kind: unsupported normalization")
        _finite_list(normalization["mean"], root["input_channels"], "normalization.mean")
        _finite_list(normalization["scale"], root["input_channels"], "normalization.scale", positive=True)
        weights = _fields(root["weights"], {"path", "sha256"}, "model weights")
        if weights["path"] != "weights.safetensors":
            raise MLRuntimeError("model weights.path: expected 'weights.safetensors'")
        _checksum(weights["sha256"], "model weights.sha256")
        weights_path = directory / weights["path"]
        if not weights_path.resolve().is_relative_to(directory):
            raise MLRuntimeError("model weights must remain inside the model directory")
        if sha256_file(weights_path) != weights["sha256"]:
            raise MLRuntimeError("model weights: SHA-256 checksum mismatch")
        training = _fields(root["training"], {"seed", "epochs", "learning_rate", "max_voxels", "prepared_sha256", "train_scene_ids"}, "model training")
        for key in ("epochs", "max_voxels"):
            _positive_integer(training[key], f"model training.{key}")
        if type(training["seed"]) is not int or not 0 <= training["seed"] <= 2**32 - 1:
            raise MLRuntimeError("model training.seed: expected an integer in 0..4294967295")
        _finite_list([training["learning_rate"]], 1, "model training.learning_rate", positive=True)
        _checksum(training["prepared_sha256"], "model training.prepared_sha256")
        ids = training["train_scene_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) for value in ids) or len(ids) != len(set(ids)):
            raise MLRuntimeError("model training.train_scene_ids: expected unique scene IDs")
        runtime = _fields(root["runtime"], {"toolkit_version", "torch_version", "cuda_version", "warpconvnet_version", "device"}, "model runtime")
        for key in ("toolkit_version", "torch_version"):
            if not isinstance(runtime[key], str) or not runtime[key]:
                raise MLRuntimeError(f"model runtime.{key}: expected a version string")
        for key in ("cuda_version", "warpconvnet_version"):
            if runtime[key] is not None and (not isinstance(runtime[key], str) or not runtime[key]):
                raise MLRuntimeError(f"model runtime.{key}: expected a version string or null")
        if runtime["device"] not in ("cpu", "cuda"):
            raise MLRuntimeError("model runtime.device: expected cpu or cuda")
        return root, weights_path, sha256_file(source)
    except (OSError, RuntimeError, TypeError, ValueError, RecursionError) as error:
        if isinstance(error, MLRuntimeError):
            raise
        raise MLRuntimeError(f"{path}: cannot load model bundle: {error}") from error


def _load_model(path, dataset, device):
    bundle, weights_path, bundle_sha256 = _read_bundle(path, dataset)
    torch, load_file, _ = _dependencies()
    from safetensors import SafetensorError
    device, backend = resolve_runtime(device=device, backend=bundle["backend"])
    model = build_model(backend, bundle["input_channels"], bundle["num_classes"], device)
    try:
        state = load_file(str(weights_path), device="cpu")
        expected = model.state_dict()
        if set(state) != set(expected):
            raise MLRuntimeError("model weights: tensor names do not match the declared architecture")
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise MLRuntimeError(f"model weights {name!r}: shape or dtype does not match the declared architecture")
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise MLRuntimeError(f"model weights {name!r}: expected finite values")
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError, SafetensorError) as error:
        if isinstance(error, MLRuntimeError):
            raise
        raise MLRuntimeError(f"cannot load numeric model weights: {error}") from error
    return model, bundle, bundle_sha256, device, torch


def _run_inference(prepared, model_path, output, *, device, split, max_voxels, evaluate):
    _check_output(output)
    dataset = load_prepared_manifest(prepared)
    scenes = _selected_scenes(dataset, split, max_voxels)
    model, bundle, bundle_sha256, device, torch = _load_model(model_path, dataset, device)
    for _, sample in scenes:
        _normalized_features(sample, bundle["normalization"])
    try:
        report = {
            "schema_version": 1,
            "kind": "rptu-evaluation" if evaluate else "rptu-predictions",
            "prepared_sha256": sha256_file(dataset.source),
            "model_bundle_sha256": bundle_sha256,
            "model_weights_sha256": bundle["weights"]["sha256"],
            "split": split,
            "backend": bundle["backend"],
            "architecture": bundle["architecture"],
            "device": device,
            "contract": _contract(dataset),
            "scenes": [],
        }
        with new_output_directory(output) as directory:
            (directory / "scenes").mkdir()

            def write_prediction(scene, arrays):
                relative = f"scenes/{scene.id}.npz"
                write_npz(directory / relative, arrays)
                report["scenes"].append({
                    "id": scene.id, "path": relative, "sha256": sha256_file(directory / relative),
                    "input_points": scene.input_points, "valid_points": scene.valid_points,
                    "source_sha256": scene.sha256,
                })

            metrics = _infer(scenes, dataset, model, bundle["normalization"], device, torch,
                             on_prediction=write_prediction)
            if evaluate:
                if metrics["scored_points"] == 0:
                    raise MLRuntimeError(
                        f"split {split!r} has no valid nonignored point labels to evaluate; "
                        "use predict for unlabeled data"
                    )
                report["metrics"] = metrics
            write_json(directory / ("evaluation.json" if evaluate else "predictions.json"), report)
    except RuntimeError as error:
        raise MLRuntimeError(f"inference failed: {error}") from error
    return report


def evaluate_model(prepared, model, output, *, device="auto", split="test", max_voxels=100000) -> dict:
    """Evaluate original-row targets, excluding invalid, ignored and unlabeled rows."""
    return _run_inference(prepared, model, output, device=device, split=split, max_voxels=max_voxels, evaluate=True)


def predict_model(prepared, model, output, *, device="auto", split="predict", max_voxels=100000) -> dict:
    """Write original-row predictions in the dataset's original label ID space."""
    return _run_inference(prepared, model, output, device=device, split=split, max_voxels=max_voxels, evaluate=False)
