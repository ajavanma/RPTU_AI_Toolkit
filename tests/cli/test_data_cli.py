"""Exercise the installed CPU data workflow outside the source checkout."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class DatasetCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rptu-data-cli-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.environment = os.environ.copy()
        self.environment.pop("PYTHONPATH", None)
        self.environment["PYTHONNOUSERSITE"] = "1"

    def run_cli(self, *args, guard=False):
        invocation = [sys.executable, "-m", "rptu_ai_toolkit"]
        if guard:
            script = """
import importlib.abc
import runpy
import sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'warpconvnet', 'MinkowskiEngine', 'open3d'}:
            raise AssertionError('Unexpected ML import: ' + fullname)
sys.meta_path.insert(0, NoML())
sys.argv = ['rptu', *sys.argv[1:]]
runpy.run_module('rptu_ai_toolkit', run_name='__main__')
"""
            invocation = [sys.executable, "-c", script]
        return subprocess.run(
            [*invocation, *map(str, args)], cwd=self.directory,
            env=self.environment, capture_output=True, text=True, timeout=30,
        )

    def success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def usage_error(self, result):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("error", result.stderr.lower())
        self.assertNotIn("Traceback", result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")

    def generate(self, name="raw", **kwargs):
        report = self.success(self.run_cli("synthetic", "--output", name, "--seed", "42", "--json", **kwargs))
        return Path(report["manifest"])

    def mutate_scene(self, manifest_path, change, index=0):
        document = json.loads(manifest_path.read_text())
        scene = document["scenes"][index]
        path = manifest_path.parent / scene["path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        change(arrays)
        np.savez(path, **arrays)
        scene["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(document), encoding="utf-8")

    def test_generate_inspect_preprocess_without_ml_imports(self):
        manifest = self.generate("runs/raw", guard=True)
        inspection = self.success(self.run_cli("inspect", "--manifest", manifest, "--json", guard=True))
        self.assertEqual({scene["split"] for scene in inspection["scenes"]}, {"train", "val", "test", "predict"})
        self.assertGreater(inspection["total_invalid_points"], 0)
        self.assertEqual([label["dataset_id"] for label in inspection["label_mapping"]], [0, 7, 42])
        prepared = self.success(self.run_cli(
            "preprocess", "--manifest", manifest, "--output", "runs/prepared",
            "--voxel-size", "0.25", "--json", guard=True,
        ))
        saved = json.loads((self.directory / "runs/prepared/prepared.json").read_text())
        self.assertEqual(saved, prepared)
        self.assertEqual(prepared["coordinate_unit"], "m")
        self.assertEqual(prepared["source_manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        raw_document = json.loads(manifest.read_text())
        for scene in prepared["scenes"]:
            output = self.directory / "runs/prepared" / scene["path"]
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), scene["sha256"])
            source = next(item for item in raw_document["scenes"] if item["id"] == scene["id"])
            with np.load(output, allow_pickle=False) as result, np.load(manifest.parent / source["path"], allow_pickle=False) as raw:
                mask = np.isfinite(raw["coords"]).all(1) & np.isfinite(raw["features"]).all(1)
                mapping = result["point_to_voxel"]
                np.testing.assert_array_equal(result["point_indices"], raw["point_indices"])
                np.testing.assert_array_equal(mapping >= 0, mask)
                np.testing.assert_array_equal(result["voxel_coords"][mapping[mask]], np.floor(raw["coords"][mask] / 0.25).astype(np.int64))
                self.assertEqual(len(mapping), scene["input_points"])
                self.assertEqual(len(result["features"]), scene["voxel_count"])
                self.assertEqual("labels" in result.files, scene["split"] != "predict")
                if "labels" in result.files:
                    self.assertLessEqual(set(result["labels"].tolist()), {-1, 0, 1, 2})
                    expected_labels = np.array([{0: 0, 7: 1, 42: 2, -1: -1}[int(label)] for label in raw["labels"]])
                    np.testing.assert_array_equal(result["point_labels"], expected_labels)
                else:
                    self.assertNotIn("point_labels", result.files)

    def test_human_readable_commands(self):
        manifest = self.generate()
        for args in (
            ("inspect", "--manifest", manifest),
            ("preprocess", "--manifest", manifest, "--output", "prepared", "--voxel-size", "0.25"),
        ):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.strip())

    def test_same_seed_and_inputs_produce_identical_outputs(self):
        first = self.generate("first")
        second = self.generate("second")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        for manifest, output in ((first, "prepared-one"), (second, "prepared-two")):
            self.success(self.run_cli("preprocess", "--manifest", manifest, "--output", output, "--voxel-size", "0.25", "--json"))
        for path in (self.directory / "prepared-one").rglob("*"):
            if path.is_file():
                other = self.directory / "prepared-two" / path.relative_to(self.directory / "prepared-one")
                self.assertEqual(path.read_bytes(), other.read_bytes())

    def test_invalid_sizes_and_missing_arguments_fail_cleanly(self):
        manifest = self.generate()
        for size in ("0", "-1", "nan", "inf", "1e-320"):
            with self.subTest(size=size):
                result = self.run_cli("preprocess", "--manifest", manifest, "--output", "bad-output", "--voxel-size", size, "--json")
                self.usage_error(result)
                self.assertFalse((self.directory / "bad-output").exists())
        for args in (("synthetic",), ("inspect",), ("preprocess",), ("inspect", "--manifest", "absent.json")):
            self.usage_error(self.run_cli(*args))

    def test_invalid_seed_does_not_write(self):
        for seed in ("-1", "4294967296", "1.5"):
            self.usage_error(self.run_cli("synthetic", "--output", "absent", "--seed", seed))
            self.assertFalse((self.directory / "absent").exists())

    def test_existing_outputs_are_preserved(self):
        manifest = self.generate()
        before = manifest.read_bytes()
        self.usage_error(self.run_cli("synthetic", "--output", manifest.parent))
        self.assertEqual(manifest.read_bytes(), before)
        self.usage_error(self.run_cli("preprocess", "--manifest", manifest, "--output", manifest.parent, "--voxel-size", "0.25"))
        self.assertEqual(manifest.read_bytes(), before)
        empty = self.directory / "empty"
        empty.mkdir()
        self.usage_error(self.run_cli("preprocess", "--manifest", manifest, "--output", empty, "--voxel-size", "0.25"))
        self.assertTrue(empty.is_dir())
        self.assertEqual(list(empty.iterdir()), [])

    def test_checksum_failure_and_partial_preprocessing_cleanup(self):
        manifest = self.generate()
        document = json.loads(manifest.read_text())
        broken = manifest.parent / document["scenes"][1]["path"]
        broken.write_bytes(b"changed input")
        self.usage_error(self.run_cli("inspect", "--manifest", manifest, "--json"))
        self.usage_error(self.run_cli("preprocess", "--manifest", manifest, "--output", "prepared", "--voxel-size", "0.25"))
        self.assertFalse((self.directory / "prepared").exists())

    def test_unknown_label_on_an_invalid_point_is_rejected(self):
        manifest = self.generate()
        def corrupt(arrays):
            arrays["coords"][0, 0] = np.nan
            arrays["labels"][0] = 12345
        self.mutate_scene(manifest, corrupt)
        self.usage_error(self.run_cli("inspect", "--manifest", manifest))
        self.usage_error(self.run_cli("preprocess", "--manifest", manifest, "--output", "prepared", "--voxel-size", "0.25"))
        self.assertFalse((self.directory / "prepared").exists())

    def test_train_labels_required_but_prediction_is_unlabeled(self):
        manifest = self.generate()
        inspection = self.success(self.run_cli("inspect", "--manifest", manifest, "--json"))
        predict = next(scene for scene in inspection["scenes"] if scene["split"] == "predict")
        self.assertFalse(predict["labeled"])
        self.mutate_scene(manifest, lambda arrays: arrays.pop("labels"))
        self.usage_error(self.run_cli("inspect", "--manifest", manifest, "--json"))

    def test_identical_content_cannot_cross_splits(self):
        manifest = self.generate()
        document = json.loads(manifest.read_text())
        document["scenes"][1]["sha256"] = document["scenes"][0]["sha256"]
        manifest.write_text(json.dumps(document), encoding="utf-8")
        self.usage_error(self.run_cli("inspect", "--manifest", manifest))


if __name__ == "__main__":
    unittest.main()
