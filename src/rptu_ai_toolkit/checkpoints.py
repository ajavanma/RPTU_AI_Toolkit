"""Immutable, pickle-free epoch checkpoints for the built-in Adam trainer."""

import json
import os
from pathlib import Path
import re
import stat
import uuid

from .datasets import _fields, _float, _nonfinite, _object, sha256_file
from .models import MLRuntimeError
from .runs import atomic_json, fsync_directory


_CHECKPOINT = re.compile(r"checkpoints/epoch-[0-9]{8,10}-[0-9a-f]{32}/checkpoint.json")


def _regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise MLRuntimeError(f"{path}: expected a regular checkpoint file")


def _read_json(path):
    try:
        _regular(path)
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                          parse_constant=_nonfinite, parse_float=_float)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        raise MLRuntimeError(f"{path}: invalid checkpoint JSON: {error}") from error


def _equal(actual, expected, location):
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise MLRuntimeError(f"{location}: does not match this training run")


def _inside(directory, relative):
    path = directory / relative
    if any(part.is_symlink() for part in (path, *path.parents) if part != directory.parent) or not path.resolve().is_relative_to(directory.resolve()):
        raise MLRuntimeError("checkpoint files must remain inside the training run")
    return path


def _tensor(value):
    # Clone detaches mmap-backed inputs and avoids shared storage in safetensors.
    return value.detach().cpu().contiguous().clone()


def _optimizer_layout(optimizer):
    state = optimizer.state_dict()
    return state["param_groups"], [parameter for group in optimizer.param_groups for parameter in group["params"]]


def save_checkpoint(run, model, optimizer, history, torch, save_file, device):
    """Publish a complete immutable checkpoint, then atomically advance latest."""
    epoch = len(history)
    relative = f"checkpoints/epoch-{epoch:08d}-{uuid.uuid4().hex}"
    directory = _inside(run.path, relative)
    directory.parent.mkdir(exist_ok=True)
    directory.mkdir()
    tensors = {f"model.{name}": _tensor(value) for name, value in model.state_dict().items()}
    state = optimizer.state_dict()
    parameter_ids = sorted(state["state"])
    for parameter_id in parameter_ids:
        values = state["state"][parameter_id]
        if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise MLRuntimeError("checkpoint optimizer: unsupported Adam state")
        for name, value in values.items():
            tensors[f"adam.{parameter_id}.{name}"] = _tensor(value)
    tensors["rng.cpu"] = torch.random.get_rng_state().clone()
    if device == "cuda":
        tensors["rng.cuda"] = torch.cuda.get_rng_state().cpu().clone()
    if any(value.is_floating_point() and not torch.isfinite(value).all().item() for value in tensors.values()):
        raise MLRuntimeError("checkpoint tensors must contain finite values")
    weights = directory / "state.safetensors"
    save_file(tensors, str(weights))
    with weights.open("rb") as stream:
        os.fsync(stream.fileno())
    metadata = {
        "schema_version": 1, "kind": "rptu-training-checkpoint", "run_id": run.run_id,
        "run_sha256": sha256_file(run.path / "run.json"), "epoch": epoch,
        "history": history, "optimizer": {"kind": "Adam", "param_groups": json.loads(json.dumps(state["param_groups"])),
                                           "parameter_ids": parameter_ids},
        "state": {"path": "state.safetensors", "sha256": sha256_file(weights)},
    }
    atomic_json(directory / "checkpoint.json", metadata)
    fsync_directory(directory)
    fsync_directory(directory.parent)
    run.atomic_json("latest.json", {
        "schema_version": 1, "kind": "rptu-latest-checkpoint", "run_id": run.run_id,
        "path": f"{relative}/checkpoint.json", "sha256": sha256_file(directory / "checkpoint.json"),
    })
    return metadata


def load_checkpoint(run, model, optimizer, torch, load_file, device, epochs):
    """Validate every field/tensor before restoring model, Adam and torch RNG."""
    from safetensors import SafetensorError
    pointer_path = _inside(run.path, "latest.json")
    if not pointer_path.exists():
        directory = _inside(run.path, "checkpoints")
        if directory.exists() and any(not re.fullmatch(r"epoch-00000000-[0-9a-f]{32}", child.name)
                                      for child in directory.iterdir()):
            raise MLRuntimeError("latest checkpoint pointer is missing from an existing training run")
        return None
    try:
        pointer = _fields(_read_json(pointer_path), {"schema_version", "kind", "run_id", "path", "sha256"}, "latest checkpoint")
        _equal(pointer["schema_version"], 1, "latest schema_version")
        _equal(pointer["kind"], "rptu-latest-checkpoint", "latest kind")
        _equal(pointer["run_id"], run.run_id, "latest run_id")
        if not isinstance(pointer["path"], str) or not _CHECKPOINT.fullmatch(pointer["path"]):
            raise MLRuntimeError("latest checkpoint path: expected an immutable checkpoint path")
        metadata_path = _inside(run.path, pointer["path"])
        _regular(metadata_path)
        if sha256_file(metadata_path) != pointer["sha256"]:
            raise MLRuntimeError("checkpoint metadata: SHA-256 checksum mismatch")
        metadata = _fields(_read_json(metadata_path), {
            "schema_version", "kind", "run_id", "run_sha256", "epoch", "history", "optimizer", "state",
        }, "checkpoint")
        for key, expected in (("schema_version", 1), ("kind", "rptu-training-checkpoint"),
                              ("run_id", run.run_id), ("run_sha256", sha256_file(run.path / "run.json"))):
            _equal(metadata[key], expected, f"checkpoint {key}")
        epoch = metadata["epoch"]
        if type(epoch) is not int or not 0 <= epoch <= epochs:
            raise MLRuntimeError("checkpoint epoch: outside the requested training run")
        if not metadata_path.parent.name.startswith(f"epoch-{epoch:08d}-"):
            raise MLRuntimeError("checkpoint directory does not match its completed epoch")
        history = metadata["history"]
        if not isinstance(history, list) or len(history) != epoch:
            raise MLRuntimeError("checkpoint history: expected one entry per completed epoch")
        for index, item in enumerate(history):
            item = _fields(item, {"epoch", "loss"}, "checkpoint history entry")
            _equal(item["epoch"], index + 1, "checkpoint history epoch")
            if type(item["loss"]) not in (int, float) or not 0 <= item["loss"] < float("inf"):
                raise MLRuntimeError("checkpoint history loss: expected a finite nonnegative number")
        saved_optimizer = _fields(metadata["optimizer"], {"kind", "param_groups", "parameter_ids"}, "checkpoint optimizer")
        groups, parameters = _optimizer_layout(optimizer)
        _equal(saved_optimizer["kind"], "Adam", "checkpoint optimizer kind")
        _equal(saved_optimizer["param_groups"], groups, "checkpoint optimizer param_groups")
        expected_ids = [index for index, parameter in enumerate(parameters) if parameter.requires_grad] if epoch else []
        _equal(saved_optimizer["parameter_ids"], expected_ids, "checkpoint optimizer parameter_ids")
        state_info = _fields(metadata["state"], {"path", "sha256"}, "checkpoint state")
        _equal(state_info["path"], "state.safetensors", "checkpoint state path")
        state_path = _inside(run.path, str(metadata_path.parent.relative_to(run.path) / state_info["path"]))
        _regular(state_path)
        if sha256_file(state_path) != state_info["sha256"]:
            raise MLRuntimeError("checkpoint state: SHA-256 checksum mismatch")
        tensors = load_file(str(state_path), device="cpu")
        model_state = model.state_dict()
        expected = {f"model.{name}": value for name, value in model_state.items()}
        for parameter_id in expected_ids:
            parameter = parameters[parameter_id]
            for name in ("exp_avg", "exp_avg_sq"):
                expected[f"adam.{parameter_id}.{name}"] = parameter
            expected[f"adam.{parameter_id}.step"] = torch.tensor(0.0, dtype=torch.float32)
        expected["rng.cpu"] = torch.random.get_rng_state()
        if device == "cuda":
            expected["rng.cuda"] = torch.cuda.get_rng_state().cpu()
        if set(tensors) != set(expected):
            raise MLRuntimeError("checkpoint state: tensor names do not match the model/optimizer/RNG contract")
        for name, value in tensors.items():
            if value.dtype != expected[name].dtype or value.shape != expected[name].shape:
                raise MLRuntimeError(f"checkpoint tensor {name!r}: shape or dtype mismatch")
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise MLRuntimeError(f"checkpoint tensor {name!r}: expected finite values")
            if name.startswith("adam.") and name.endswith(".step"):
                step = value.item()
                if step != epoch * run.spec["optimizer_steps_per_epoch"]:
                    raise MLRuntimeError("checkpoint Adam step: does not match completed epochs")
            if name.startswith("adam.") and name.endswith(".exp_avg_sq") and (value < 0).any().item():
                raise MLRuntimeError("checkpoint Adam exp_avg_sq: expected nonnegative values")
        optimizer_state = {
            "param_groups": saved_optimizer["param_groups"],
            "state": {parameter_id: {name: tensors[f"adam.{parameter_id}.{name}"].clone()
                                     for name in ("step", "exp_avg", "exp_avg_sq")}
                      for parameter_id in expected_ids},
        }
        model.load_state_dict({name: tensors[f"model.{name}"] for name in model_state}, strict=True)
        optimizer.load_state_dict(optimizer_state)
        # Construction and loading may use random numbers; restore RNG last.
        torch.random.set_rng_state(tensors["rng.cpu"])
        if device == "cuda":
            torch.cuda.set_rng_state(tensors["rng.cuda"])
        return history
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, RecursionError, SafetensorError) as error:
        if isinstance(error, MLRuntimeError):
            raise
        raise MLRuntimeError(f"cannot restore checkpoint: {error}") from error
