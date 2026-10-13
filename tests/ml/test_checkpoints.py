"""Real Adam interruption/resume and durable run integrity regressions."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from rptu_ai_toolkit import ml
from rptu_ai_toolkit.artifacts import write_json
from rptu_ai_toolkit.data_commands import preprocess_dataset
from rptu_ai_toolkit.datasets import sha256_file
from rptu_ai_toolkit.models import MLRuntimeError
from rptu_ai_toolkit.synthetic import generate_synthetic


class TestTrainingCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        manifest = generate_synthetic(cls.root / "raw", seed=12)
        preprocess_dataset(manifest, cls.root / "prepared", 0.5)
        cls.prepared = cls.root / "prepared/prepared.json"
        cls.reference = cls.root / "reference"
        cls.report = ml.train_model(cls.prepared, cls.reference, device="cpu", backend="torch", epochs=3, seed=9)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)
        cls.temporary.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "run"

    def train(self, **kwargs):
        return ml.train_model(self.prepared, self.output, device="cpu", backend="torch", **kwargs)

    def interrupt_after(self, epoch=1):
        original = ml.save_checkpoint
        def save(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(args[3]) == epoch:
                raise KeyboardInterrupt("simulated interruption")
            return result
        with patch.object(ml, "save_checkpoint", side_effect=save):
            with self.assertRaises(KeyboardInterrupt):
                self.train(epochs=3, seed=9)

    def checkpoint(self):
        pointer = json.loads((self.output / "latest.json").read_text())
        path = self.output / pointer["path"]
        return pointer, path, json.loads(path.read_text())

    def rehash(self, pointer, path, metadata):
        path.write_text(json.dumps(metadata))
        pointer["sha256"] = sha256_file(path)
        (self.output / "latest.json").write_text(json.dumps(pointer))

    def hashes(self):
        return {p.relative_to(self.output).as_posix(): sha256_file(p) for p in self.output.rglob("*") if p.is_file()}

    def test_completed_epoch_resume_matches_uninterrupted_adam_history_weights_and_rng(self):
        import torch
        self.interrupt_after()
        torch.manual_seed(991)
        before = torch.random.get_rng_state().clone()
        report = self.train(resume=True)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(report, self.report)
        self.assertEqual(sha256_file(self.output / "weights.safetensors"), sha256_file(self.reference / "weights.safetensors"))
        attempts = [json.loads(p.read_text()) for p in (self.output / "attempts").glob("*/attempt.json")]
        self.assertEqual(len(attempts), 2)
        self.assertTrue((self.output / "complete.json").is_file())

    def test_mid_epoch_interruption_restarts_from_previous_checkpoint(self):
        import torch
        original = torch.optim.Adam.step
        calls = 0
        def step(optimizer, *args, **kwargs):
            nonlocal calls
            result = original(optimizer, *args, **kwargs)
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("after an unsaved optimizer step")
            return result
        with patch.object(torch.optim.Adam, "step", step):
            with self.assertRaises(KeyboardInterrupt):
                self.train(epochs=3, seed=9)
        self.assertLess(self.checkpoint()[2]["epoch"], 2)
        self.assertEqual(self.train(resume=True), self.report)
        self.assertEqual(sha256_file(self.output / "weights.safetensors"), sha256_file(self.reference / "weights.safetensors"))

    def test_completed_retry_is_read_only_and_rejects_changed_settings(self):
        shutil.copytree(self.reference, self.output)
        before = self.hashes()
        with patch.object(ml, "save_checkpoint", side_effect=AssertionError("retraining")):
            self.assertEqual(self.train(resume=True), self.report)
        for settings in ({"epochs": 4}, {"seed": 7}, {"learning_rate": 0.02}, {"max_voxels": 200000}):
            with self.subTest(settings=settings), self.assertRaisesRegex(MLRuntimeError, next(iter(settings))):
                self.train(resume=True, **settings)
        self.assertEqual(before, self.hashes())

    def test_dataset_identity_and_runtime_changes_fail_without_modifying_run(self):
        self.interrupt_after()
        before = self.hashes()
        other = self.directory / "different.json"
        document = json.loads(self.prepared.read_text())
        document["source_manifest_sha256"] = "f" * 64
        # Keep scene paths valid while changing only the manifest provenance.
        other = self.prepared.parent / "different.json"
        write_json(other, document)
        self.addCleanup(other.unlink)
        with self.assertRaisesRegex(MLRuntimeError, "prepared_sha256"):
            ml.train_model(other, self.output, resume=True)
        original = ml._runtime
        with patch.object(ml, "_runtime", side_effect=lambda *args: {**original(*args), "torch_version": "0.0"}):
            with self.assertRaisesRegex(MLRuntimeError, "torch_version"):
                self.train(resume=True)
        self.assertEqual(before, self.hashes())

    def test_corrupted_and_rehashed_malformed_checkpoint_fail_cleanly(self):
        self.interrupt_after()
        pointer, path, metadata = self.checkpoint()
        state_path = path.parent / "state.safetensors"
        state_path.write_bytes(b"invalid checkpoint tensors")
        with self.assertRaisesRegex(MLRuntimeError, "checksum mismatch"):
            self.train(resume=True)
        metadata["state"]["sha256"] = sha256_file(state_path)
        self.rehash(pointer, path, metadata)
        with self.assertRaisesRegex(MLRuntimeError, "restore checkpoint"):
            self.train(resume=True)

    def test_rehashed_wrong_tensor_names_shapes_dtypes_and_values_fail(self):
        import torch
        from safetensors.torch import load_file, save_file
        self.interrupt_after()
        pointer, path, metadata = self.checkpoint()
        state_path = path.parent / "state.safetensors"
        # Clone before intentional overwrite so no mmap aliases the file.
        original = {name: value.clone() for name, value in load_file(str(state_path)).items()}
        model_key = next(name for name in original if name.startswith("model.") and original[name].is_floating_point())
        adam_step = next(name for name in original if name.startswith("adam.") and name.endswith(".step"))
        bad = [
            {key: value for key, value in original.items() if key != model_key},
            {**original, model_key: original[model_key].flatten()[:1]},
            {**original, model_key: original[model_key].double()},
            {**original, model_key: torch.full_like(original[model_key], float("nan"))},
            {**original, adam_step: torch.tensor(99.0)},
            {**original, "rng.cpu": original["rng.cpu"][:1]},
        ]
        for tensors in bad:
            with self.subTest(keys=list(tensors)):
                save_file(tensors, str(state_path))
                metadata["state"]["sha256"] = sha256_file(state_path)
                self.rehash(pointer, path, metadata)
                with self.assertRaises(MLRuntimeError):
                    self.train(resume=True)

    def test_unknown_duplicate_fields_and_pointer_escape_are_rejected(self):
        self.interrupt_after()
        pointer, path, metadata = self.checkpoint()
        self.rehash(pointer, path, {**metadata, "unknown": True})
        with self.assertRaises(MLRuntimeError):
            self.train(resume=True)
        path.write_text('{"epoch":1,"epoch":1}')
        pointer["sha256"] = sha256_file(path)
        (self.output / "latest.json").write_text(json.dumps(pointer))
        with self.assertRaises(MLRuntimeError):
            self.train(resume=True)
        pointer["path"] = "../outside/checkpoint.json"
        (self.output / "latest.json").write_text(json.dumps(pointer))
        with self.assertRaisesRegex(MLRuntimeError, "path"):
            self.train(resume=True)

    def test_missing_completion_is_repaired_without_retraining_and_inference_rejects_it(self):
        shutil.copytree(self.reference, self.output)
        (self.output / "complete.json").unlink()
        (self.output / "training.json").write_text("interrupted publication")
        with self.assertRaisesRegex(MLRuntimeError, "incomplete"):
            ml.predict_model(self.prepared, self.output, self.directory / "predictions", device="cpu")
        with patch.object(ml, "save_checkpoint", side_effect=AssertionError("must not retrain")):
            self.assertEqual(self.train(resume=True), self.report)
        self.assertEqual(sha256_file(self.output / "weights.safetensors"), sha256_file(self.reference / "weights.safetensors"))

    def test_failed_first_checkpoint_can_resume_and_completed_hash_corruption_cannot(self):
        with patch.object(ml, "save_checkpoint", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.train(epochs=3, seed=9)
        self.assertFalse((self.output / "latest.json").exists())
        self.assertEqual(self.train(resume=True), self.report)
        (self.output / "training.json").write_text("corrupt completed output")
        before = self.hashes()
        with self.assertRaises(MLRuntimeError):
            self.train(resume=True)
        self.assertEqual(before, self.hashes())

    def test_missing_latest_after_progress_and_malformed_runtime_fail_cleanly(self):
        self.interrupt_after()
        (self.output / "latest.json").unlink()
        with self.assertRaisesRegex(MLRuntimeError, "pointer is missing"):
            self.train(resume=True)
        run_path = self.output / "run.json"
        original = json.loads(run_path.read_text())
        for key in ("backend", "runtime"):
            document = copy.deepcopy(original)
            if key == "backend":
                document["spec"][key] = []
            else:
                document["spec"][key]["device"] = []
            run_path.write_text(json.dumps(document))
            with self.assertRaises(MLRuntimeError):
                ml.train_model(self.prepared, self.output, resume=True)

    def test_stochastic_model_restores_rng_for_identical_resumed_training(self):
        import torch
        class StochasticModel(torch.nn.Module):
            def __init__(self, channels, classes):
                super().__init__()
                self.network = torch.nn.Sequential(torch.nn.Linear(channels, 8), torch.nn.ReLU(),
                                                   torch.nn.Dropout(0.5), torch.nn.Linear(8, classes))
            def forward(self, coordinates, features):
                return self.network(features)
        def build(backend, channels, classes, device):
            return StochasticModel(channels, classes).to(device)
        with patch.object(ml, "build_model", side_effect=build):
            reference = self.directory / "stochastic-reference"
            report = ml.train_model(self.prepared, reference, device="cpu", backend="torch", epochs=3, seed=9)
            self.interrupt_after()
            self.assertEqual(self.train(resume=True), report)
            self.assertEqual(sha256_file(self.output / "weights.safetensors"), sha256_file(reference / "weights.safetensors"))

    def test_rehashed_completed_report_and_summary_are_strict(self):
        shutil.copytree(self.reference, self.output)
        report_path = self.output / "training.json"
        complete_path = self.output / "complete.json"
        completion = json.loads(complete_path.read_text())
        report = json.loads(report_path.read_text())
        bad = ["not JSON", json.dumps({**report, "schema_version": True}),
               json.dumps({**report, "train": {**report["train"], "accuracy": -1}})]
        for content in bad:
            report_path.write_text(content)
            completion["files"]["training.json"] = sha256_file(report_path)
            complete_path.write_text(json.dumps(completion))
            before = self.hashes()
            with self.assertRaises(MLRuntimeError):
                self.train(resume=True)
            self.assertEqual(before, self.hashes())
        report_path.write_text(json.dumps(report))
        completion["files"]["training.json"] = sha256_file(report_path)
        completion["summary"] = {"epochs": 999}
        complete_path.write_text(json.dumps(completion))
        with self.assertRaisesRegex(MLRuntimeError, "summary"):
            self.train(resume=True)

    def test_no_resume_and_invalid_inputs_do_not_replace_outputs(self):
        shutil.copytree(self.reference, self.output)
        before = self.hashes()
        with self.assertRaisesRegex(MLRuntimeError, "output already exists"):
            self.train(epochs=3)
        self.assertEqual(before, self.hashes())
        missing = self.directory / "missing"
        with self.assertRaises(MLRuntimeError):
            ml.train_model(self.prepared, missing, resume=True)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
