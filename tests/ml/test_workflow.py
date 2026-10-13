"""Real CPU training and artifact round trips; no GPU host is required."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rptu_ai_toolkit.artifacts import write_json, write_npz
from rptu_ai_toolkit.datasets import DatasetError, sha256_file
from rptu_ai_toolkit.ml import evaluate_model, predict_model, train_model
from rptu_ai_toolkit.models import MLRuntimeError


class TestModelWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.prepared = cls.root / "prepared"
        (cls.prepared / "scenes").mkdir(parents=True)
        cls.manifest = cls.prepared / "prepared.json"
        cls.arrays = {
            "voxel_coords": np.column_stack((np.arange(6), np.zeros(6), np.zeros(6))).astype(np.int64),
            "features": np.array([[-1, 0], [-1, 1], [1, 2], [-1, 3], [1, 4], [0, 5]], dtype=np.float64),
            "labels": np.array([0, 0, 1, 0, 1, -1], dtype=np.int64),
            "point_labels": np.array([0, 1, 0, 1, 0, 1, -1, 2, -1], dtype=np.int64),
            "point_indices": np.array([90, 20, 70, 10, 60, 40, 30, 50, 80], dtype=np.int64),
            "point_to_voxel": np.array([0, 0, 1, 2, 3, 4, 5, -1, 5], dtype=np.int64),
        }
        cls.document = {
            "schema_version": 1, "kind": "rptu-prepared-dataset", "source_manifest_sha256": "a" * 64,
            "coordinate_unit": "m", "feature_names": ["signal", "height"],
            "label_mapping": [
                {"dataset_id": 42, "train_id": 0, "name": "object"},
                {"dataset_id": 0, "train_id": 1, "name": "floor"},
                {"dataset_id": 7, "train_id": 2, "name": "wall"},
            ],
            "ignore_label": 255, "training_ignore_label": -1, "voxel_size": 0.5,
            "quantization": "floor", "feature_reduction": "mean",
            "label_reduction": "majority_nonignored_then_smallest_train_id", "scenes": [],
        }
        for index, split in enumerate(("train", "val", "test", "predict")):
            arrays = copy.deepcopy(cls.arrays)
            arrays["voxel_coords"][:, 0] += index * 20
            if split == "val":
                # Held-out feature magnitudes must not leak into normalization.
                arrays["features"] *= 100
            if split == "predict":
                arrays.pop("labels")
                arrays.pop("point_labels")
            path = cls.prepared / "scenes" / f"{split}.npz"
            write_npz(path, arrays)
            cls.document["scenes"].append({
                "id": split, "split": split, "path": f"scenes/{split}.npz",
                "sha256": sha256_file(path), "source_sha256": str(index + 1) * 64,
                "input_points": 9, "valid_points": 8, "voxel_count": 6,
                "labeled": split != "predict",
            })
        write_json(cls.manifest, cls.document)
        cls.model = cls.root / "model"
        cls.training_report = train_model(cls.manifest, cls.model, device="cpu", backend="torch",
                                          epochs=60, learning_rate=0.02, seed=9)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)
        cls.temporary.cleanup()

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.case.cleanup)
        self.directory = Path(self.case.name)

    def copy_model(self):
        import shutil
        destination = self.directory / "model"
        shutil.copytree(self.model, destination)
        return destination

    def write_bundle(self, model, bundle):
        (model / "bundle.json").write_text(json.dumps(bundle))

    def test_real_training_reduces_loss_and_saves_only_numeric_weights(self):
        losses = [epoch["loss"] for epoch in self.training_report["epochs"]]
        self.assertLess(losses[-1], losses[0] * 0.1)
        self.assertEqual(self.training_report["device"], "cpu")
        self.assertEqual(self.training_report["backend"], "torch")
        self.assertEqual({path.name for path in self.model.iterdir()}, {"bundle.json", "weights.safetensors", "training.json"})
        bundle = json.loads((self.model / "bundle.json").read_text())
        self.assertEqual(bundle["weights"]["sha256"], sha256_file(self.model / "weights.safetensors"))
        np.testing.assert_allclose(bundle["normalization"]["mean"], self.arrays["features"].mean(axis=0))
        np.testing.assert_allclose(bundle["normalization"]["scale"], self.arrays["features"].std(axis=0))
        self.assertEqual(bundle["training"]["train_scene_ids"], ["train"])

    def test_reload_evaluation_uses_original_point_targets_and_ignores_invalid_rows(self):
        output = self.directory / "evaluation"
        report = evaluate_model(self.manifest, self.model, output, device="cpu")
        metrics = report["metrics"]
        self.assertEqual((metrics["total_points"], metrics["valid_points"], metrics["scored_points"]), (9, 8, 6))
        self.assertEqual(metrics["confusion_matrix"], [[3, 0, 0], [1, 2, 0], [0, 0, 0]])
        self.assertAlmostEqual(metrics["accuracy"], 5 / 6)
        self.assertAlmostEqual(metrics["classes"][0]["iou"], 3 / 4)
        self.assertAlmostEqual(metrics["classes"][1]["iou"], 2 / 3)
        self.assertIsNone(metrics["classes"][2]["iou"])
        self.assertAlmostEqual(metrics["mean_iou"], (3 / 4 + 2 / 3) / 2)
        with np.load(output / "scenes/test.npz", allow_pickle=False) as scene:
            self.assertEqual(scene["target_labels"].tolist(), [42, 0, 42, 0, 42, 0, 255, 7, 255])
        self.assertEqual(json.loads((output / "evaluation.json").read_text()), report)

    def test_prediction_preserves_row_order_source_ids_and_dataset_label_ids(self):
        output = self.directory / "predictions"
        report = predict_model(self.manifest, self.model, output, device="cpu")
        self.assertNotIn("metrics", report)
        self.assertEqual(report["scenes"][0]["sha256"], sha256_file(output / "scenes/predict.npz"))
        with np.load(output / "scenes/predict.npz", allow_pickle=False) as scene:
            self.assertEqual(set(scene.files), {"point_rows", "point_indices", "labels", "valid_mask"})
            np.testing.assert_array_equal(scene["point_rows"], np.arange(9))
            np.testing.assert_array_equal(scene["point_indices"], self.arrays["point_indices"])
            self.assertEqual(scene["labels"][:6].tolist(), [42, 42, 42, 0, 42, 0])
            self.assertEqual(scene["labels"][7], 255)
            self.assertFalse(scene["valid_mask"][7])
            self.assertTrue(set(scene["labels"].tolist()).issubset({42, 0, 7, 255}))
        with self.assertRaisesRegex(MLRuntimeError, "no.*scor|no.*label"):
            evaluate_model(self.manifest, self.model, self.directory / "unlabelled", device="cpu", split="predict")
        self.assertFalse((self.directory / "unlabelled").exists())

    def test_cpu_seed_reproduces_weights_and_preserves_callers_rng_state(self):
        import torch
        torch.manual_seed(812)
        before = torch.random.get_rng_state().clone()
        first, second = self.directory / "first", self.directory / "second"
        for destination in (first, second):
            train_model(self.manifest, destination, device="cpu", backend="torch", epochs=2, seed=18)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(sha256_file(first / "weights.safetensors"), sha256_file(second / "weights.safetensors"))

    def test_incompatible_contracts_and_strict_bundle_metadata_fail_before_output(self):
        model = self.copy_model()
        original = json.loads((model / "bundle.json").read_text())
        changes = [
            ("schema_version", True), ("schema_version", 2), ("architecture", "unknown"),
            ("input_channels", True), ("input_channels", 999), ("num_classes", 99), ("extra", 1),
        ]
        for name, value in changes:
            with self.subTest(name=name, value=value):
                bundle = copy.deepcopy(original)
                bundle[name] = value
                self.write_bundle(model, bundle)
                output = self.directory / "no-output"
                with self.assertRaises(MLRuntimeError):
                    predict_model(self.manifest, model, output, device="cpu")
                self.assertFalse(output.exists())
        for name, value in (("feature_names", ["height", "signal"]), ("coordinate_unit", "cm"),
                            ("voxel_size", 0.25), ("ignore_label", -1), ("label_mapping", []),
                            ("training_ignore_label", -1.0)):
            with self.subTest(contract=name):
                bundle = copy.deepcopy(original)
                bundle["contract"][name] = value
                self.write_bundle(model, bundle)
                with self.assertRaisesRegex(MLRuntimeError, f"contract.{name}"):
                    predict_model(self.manifest, model, self.directory / "no-output", device="cpu")

    def test_corrupt_weights_and_rehashed_wrong_tensor_shape_names_or_nonfinite_fail(self):
        import torch
        from safetensors.torch import load_file, save_file
        model = self.copy_model()
        original = json.loads((model / "bundle.json").read_text())
        weights = model / "weights.safetensors"
        state = load_file(str(weights))
        weights.write_bytes(b"not a tensor archive")
        with self.assertRaisesRegex(MLRuntimeError, "checksum mismatch"):
            predict_model(self.manifest, model, self.directory / "no-output", device="cpu")
        for mode, message in (("malformed", "numeric model weights"), ("shape", "shape or dtype"),
                              ("names", "tensor names"), ("nan", "finite")):
            with self.subTest(mode=mode):
                changed = {name: value.clone() for name, value in state.items()}
                name = next(iter(changed))
                if mode == "shape":
                    changed[name] = changed[name][:-1]
                elif mode == "names":
                    changed["not_a_parameter"] = changed.pop(name)
                elif mode == "nan":
                    changed[name].view(-1)[0] = float("nan")
                if mode != "malformed":
                    save_file(changed, str(weights))
                bundle = copy.deepcopy(original)
                bundle["weights"]["sha256"] = sha256_file(weights)
                self.write_bundle(model, bundle)
                with self.assertRaisesRegex(MLRuntimeError, message):
                    predict_model(self.manifest, model, self.directory / "no-output", device="cpu")
                self.assertFalse((self.directory / "no-output").exists())

    def test_normalization_and_fixed_weight_path_are_validated(self):
        model = self.copy_model()
        original = json.loads((model / "bundle.json").read_text())
        for field, value in (("mean", [0]), ("mean", [True, 0]), ("mean", [float("inf"), 0]),
                             ("scale", [0, 1]), ("scale", [-1, 1])):
            with self.subTest(field=field, value=value):
                bundle = copy.deepcopy(original)
                bundle["normalization"][field] = value
                self.write_bundle(model, bundle)
                with self.assertRaises(MLRuntimeError):
                    predict_model(self.manifest, model, self.directory / "no-output", device="cpu")
        original["weights"]["path"] = "../weights.safetensors"
        self.write_bundle(model, original)
        with self.assertRaisesRegex(MLRuntimeError, "weights.path"):
            predict_model(self.manifest, model, self.directory / "no-output", device="cpu")

    def test_no_overwrite_voxel_limit_and_failed_write_cleanup(self):
        marker = self.directory / "existing"
        marker.mkdir()
        (marker / "keep").write_text("unchanged")
        for action in (train_model, evaluate_model, predict_model):
            args = (self.manifest, marker) if action is train_model else (self.manifest, self.model, marker)
            with self.assertRaisesRegex(MLRuntimeError, "output already exists"):
                action(*args, device="cpu")
        self.assertEqual((marker / "keep").read_text(), "unchanged")
        for action in (train_model, evaluate_model, predict_model):
            args = (self.manifest, self.directory / "limited") if action is train_model else (self.manifest, self.model, self.directory / "limited")
            with self.assertRaisesRegex(MLRuntimeError, "exceeds max_voxels"):
                action(*args, device="cpu", max_voxels=5)
            self.assertFalse((self.directory / "limited").exists())
        with patch("rptu_ai_toolkit.ml.write_npz", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(DatasetError, "disk full"):
                predict_model(self.manifest, self.model, self.directory / "failed", device="cpu")
        self.assertFalse((self.directory / "failed").exists())

    def test_duplicate_json_fields_missing_splits_and_invalid_arguments_fail(self):
        model = self.copy_model()
        contents = (model / "bundle.json").read_text()
        (model / "bundle.json").write_text(contents.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
        with self.assertRaisesRegex(MLRuntimeError, "duplicate"):
            predict_model(self.manifest, model, self.directory / "no-output", device="cpu")
        for kwargs in ({"epochs": True}, {"epochs": 0}, {"learning_rate": float("nan")},
                       {"learning_rate": 0}, {"learning_rate": 10**400},
                       {"seed": -1}, {"max_voxels": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(MLRuntimeError):
                train_model(self.manifest, self.directory / "no-output", device="cpu", **kwargs)
        with self.assertRaisesRegex(MLRuntimeError, "split"):
            predict_model(self.manifest, self.model, self.directory / "no-output", device="cpu", split="unknown")

    def test_missing_training_split_and_all_ignored_training_targets_fail(self):
        import shutil
        prepared = self.directory / "prepared"
        shutil.copytree(self.prepared, prepared)
        manifest = prepared / "prepared.json"
        document = copy.deepcopy(self.document)
        document["scenes"] = [scene for scene in document["scenes"] if scene["split"] != "train"]
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(MLRuntimeError, "no 'train' scenes"):
            train_model(manifest, self.directory / "no-output", device="cpu", epochs=1)
        document = copy.deepcopy(self.document)
        arrays = copy.deepcopy(self.arrays)
        arrays["labels"][:] = -1
        arrays["point_labels"][:] = -1
        training_scene = prepared / "scenes/train.npz"
        training_scene.unlink()
        write_npz(training_scene, arrays)
        document["scenes"][0]["sha256"] = sha256_file(training_scene)
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(MLRuntimeError, "no nonignored training labels"):
            train_model(manifest, self.directory / "no-output", device="cpu", epochs=1)
        self.assertFalse((self.directory / "no-output").exists())

    def test_training_can_omit_validation_and_checks_scene_bytes_before_output(self):
        import shutil
        prepared = self.directory / "prepared"
        shutil.copytree(self.prepared, prepared)
        manifest = prepared / "prepared.json"
        document = copy.deepcopy(self.document)
        document["scenes"] = [scene for scene in document["scenes"] if scene["split"] == "train"]
        manifest.write_text(json.dumps(document))
        report = train_model(manifest, self.directory / "trained", device="cpu", backend="torch", epochs=1)
        self.assertIsNone(report["val"])
        (prepared / "scenes/train.npz").write_bytes(b"changed scene")
        with self.assertRaisesRegex(DatasetError, "checksum mismatch"):
            train_model(manifest, self.directory / "no-output", device="cpu", epochs=1)
        self.assertFalse((self.directory / "no-output").exists())

    def test_scene_loading_and_prediction_writes_stream_with_failure_cleanup(self):
        import shutil
        from rptu_ai_toolkit import ml
        from rptu_ai_toolkit.prepared import load_prepared_manifest

        prepared = self.directory / "prepared"
        shutil.copytree(self.prepared, prepared)
        document = copy.deepcopy(self.document)
        arrays = copy.deepcopy(self.arrays)
        arrays.pop("labels")
        arrays.pop("point_labels")
        arrays["voxel_coords"][:, 0] += 100
        second = prepared / "scenes/predict2.npz"
        write_npz(second, arrays)
        metadata = copy.deepcopy(document["scenes"][-1])
        metadata.update(id="predict2", path="scenes/predict2.npz",
                        sha256=sha256_file(second), source_sha256="5" * 64)
        document["scenes"].append(metadata)
        manifest = prepared / "prepared.json"
        manifest.write_text(json.dumps(document))
        dataset = load_prepared_manifest(manifest)

        with patch.object(ml, "load_prepared_scene", wraps=ml.load_prepared_scene) as load:
            scenes = ml._selected_scenes(dataset, "predict", 100000)
            load.assert_not_called()
            iterator = iter(scenes)
            self.assertEqual(next(iterator)[0].id, "predict")
            self.assertEqual(load.call_count, 1)
            self.assertEqual(next(iterator)[0].id, "predict2")
            self.assertEqual(load.call_count, 2)

        real_logits = ml._logits
        for fail_second in (False, True):
            output = self.directory / ("failed-stream" if fail_second else "stream")
            calls = []

            def checked_logits(*args, **kwargs):
                if calls:
                    self.assertTrue((output / "scenes/predict.npz").is_file())
                    if fail_second:
                        raise MLRuntimeError("second scene failed")
                calls.append(True)
                return real_logits(*args, **kwargs)

            with patch.object(ml, "_logits", side_effect=checked_logits):
                if fail_second:
                    with self.assertRaisesRegex(MLRuntimeError, "second scene failed"):
                        predict_model(manifest, self.model, output, device="cpu")
                    self.assertFalse(output.exists())
                else:
                    report = predict_model(manifest, self.model, output, device="cpu")
                    self.assertEqual([scene["id"] for scene in report["scenes"]], ["predict", "predict2"])

    def test_evaluation_rejects_all_ignored_labels_but_prediction_still_works(self):
        import shutil
        prepared = self.directory / "prepared"
        shutil.copytree(self.prepared, prepared)
        document = copy.deepcopy(self.document)
        arrays = copy.deepcopy(self.arrays)
        arrays["voxel_coords"][:, 0] += 40
        arrays["labels"][:] = -1
        arrays["point_labels"][:] = -1
        path = prepared / "scenes/test.npz"
        path.unlink()
        write_npz(path, arrays)
        document["scenes"][2]["sha256"] = sha256_file(path)
        manifest = prepared / "prepared.json"
        manifest.write_text(json.dumps(document))
        output = self.directory / "unscorable"
        with self.assertRaisesRegex(MLRuntimeError, "no valid nonignored point labels"):
            evaluate_model(manifest, self.model, output, device="cpu")
        self.assertFalse(output.exists())
        report = predict_model(manifest, self.model, self.directory / "predicted", device="cpu", split="test")
        self.assertEqual(len(report["scenes"]), 1)


if __name__ == "__main__":
    unittest.main()
