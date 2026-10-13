"""Exercise the installed ML command from outside the source checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rptu_ai_toolkit.datasets import sha256_file


class TestInstalledModelCLI(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.command = Path(sys.executable).with_name("rptu")
        self.assertTrue(self.command.is_file(), "Install the package before running CLI tests")
        self.environment = os.environ.copy()
        self.environment.pop("PYTHONPATH", None)
        self.environment["OMP_NUM_THREADS"] = "1"
        self.environment["MKL_NUM_THREADS"] = "1"

    def run_command(self, *arguments):
        return subprocess.run(
            [str(self.command), *map(str, arguments)], cwd=self.directory,
            env=self.environment, capture_output=True, text=True, timeout=60,
        )

    def json_command(self, *arguments):
        result = self.run_command(*arguments, "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        # json.loads rejects any model progress or import output on stdout.
        document = json.loads(result.stdout)
        self.assertIsInstance(document, dict)
        return document

    def test_installed_train_evaluate_predict_json_and_bad_model_error(self):
        generated = self.json_command("synthetic", "--output", "raw", "--seed", "11")
        self.assertEqual(Path(generated["manifest"]), self.directory / "raw/manifest.json")
        prepared = self.json_command(
            "preprocess", "--manifest", "raw/manifest.json", "--output", "prepared",
            "--voxel-size", "0.1",
        )
        self.assertEqual(len(prepared["scenes"]), 4)
        training = self.json_command(
            "train", "--prepared", "prepared/prepared.json", "--output", "model",
            "--backend", "torch", "--device", "cpu", "--epochs", "2", "--seed", "11",
        )
        self.assertEqual(training["kind"], "rptu-training-report")
        self.assertEqual(training["backend"], "torch")
        self.assertEqual(training["device"], "cpu")
        self.assertEqual(len(training["epochs"]), 2)
        model = self.directory / "model"
        bundle = json.loads((model / "bundle.json").read_text())
        self.assertEqual(bundle["weights"]["sha256"], sha256_file(model / "weights.safetensors"))
        self.assertEqual(json.loads((model / "training.json").read_text()), training)

        evaluation = self.json_command(
            "evaluate", "--prepared", "prepared/prepared.json", "--model", "model",
            "--output", "evaluation", "--device", "cpu",
        )
        self.assertEqual(evaluation["kind"], "rptu-evaluation")
        self.assertEqual(evaluation["split"], "test")
        self.assertGreater(evaluation["metrics"]["scored_points"], 0)
        self.assertEqual(json.loads((self.directory / "evaluation/evaluation.json").read_text()), evaluation)
        predictions = self.json_command(
            "predict", "--prepared", "prepared/prepared.json", "--model", "model",
            "--output", "predictions", "--device", "cpu",
        )
        self.assertEqual(predictions["kind"], "rptu-predictions")
        self.assertEqual(predictions["split"], "predict")
        self.assertEqual(predictions["model_weights_sha256"], evaluation["model_weights_sha256"])
        self.assertEqual(json.loads((self.directory / "predictions/predictions.json").read_text()), predictions)
        scene = predictions["scenes"][0]
        artifact = self.directory / "predictions" / scene["path"]
        self.assertEqual(scene["sha256"], sha256_file(artifact))
        with np.load(artifact, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["point_rows"], np.arange(45))
            np.testing.assert_array_equal(result["point_indices"], np.arange(45))
            self.assertEqual(int(result["valid_mask"].sum()), 42)
            self.assertTrue(np.all(result["labels"][~result["valid_mask"]] == -1))
            self.assertTrue(set(result["labels"].tolist()).issubset({-1, 0, 7, 42}))

        # Corruption must be an ordinary CLI error, with no partially created output.
        with (model / "weights.safetensors").open("ab") as stream:
            stream.write(b"corrupt")
        failure = self.run_command(
            "predict", "--prepared", "prepared/prepared.json", "--model", "model",
            "--output", "failed-prediction", "--device", "cpu", "--json",
        )
        self.assertEqual(failure.returncode, 2)
        self.assertEqual(failure.stdout, "")
        self.assertIn("checksum mismatch", failure.stderr)
        self.assertNotIn("Traceback", failure.stderr)
        self.assertFalse((self.directory / "failed-prediction").exists())


if __name__ == "__main__":
    unittest.main()
