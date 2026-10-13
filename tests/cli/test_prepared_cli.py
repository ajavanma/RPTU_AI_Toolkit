"""Inspect portable prepared bundles through the installed CPU CLI."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class PreparedDatasetCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rptu-prepared-cli-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.environment = os.environ.copy()
        self.environment.pop("PYTHONPATH", None)
        self.environment["PYTHONNOUSERSITE"] = "1"

    def run_cli(self, *args, guard=False, console=False):
        invocation = [sys.executable, "-m", "rptu_ai_toolkit"]
        if console:
            invocation = [str(Path(sys.executable).with_name("rptu"))]
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

    def usage_error(self, result, detail=None):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("error", result.stderr.lower())
        self.assertNotIn("Traceback", result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        if detail is not None:
            self.assertIn(detail, result.stderr)

    def prepare(self, guard=False):
        generated = self.success(self.run_cli(
            "synthetic", "--output", "raw", "--seed", "42", "--json", guard=guard,
        ))
        raw = Path(generated["manifest"])
        self.success(self.run_cli(
            "preprocess", "--manifest", raw, "--output", "prepared",
            "--voxel-size", "0.25", "--json", guard=guard,
        ))
        return raw, self.directory / "prepared" / "prepared.json"

    def test_generate_preprocess_inspect_prepared_without_ml_imports(self):
        raw, prepared = self.prepare(guard=True)
        original = json.loads(prepared.read_text(encoding="utf-8"))
        report = self.success(self.run_cli(
            "inspect", "--prepared", prepared, "--json", guard=True,
        ))
        self.assertEqual(report["kind"], "rptu-prepared-dataset")
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["coordinate_unit"], "m")
        self.assertEqual(report["voxel_size"], 0.25)
        self.assertEqual(report["feature_names"], original["feature_names"])
        self.assertEqual(report["label_mapping"], original["label_mapping"])
        self.assertEqual(report["ignore_label"], original["ignore_label"])
        self.assertEqual(report["training_ignore_label"], -1)
        self.assertEqual(
            report["source_manifest_sha256"], hashlib.sha256(raw.read_bytes()).hexdigest(),
        )
        self.assertEqual({scene["split"] for scene in report["scenes"]}, {"train", "val", "test", "predict"})
        expected_input = expected_valid = expected_voxels = 0
        for scene, saved in zip(report["scenes"], original["scenes"], strict=True):
            self.assertEqual(scene["id"], saved["id"])
            for field in ("input_points", "valid_points", "voxel_count", "labeled", "sha256"):
                self.assertEqual(scene[field], saved[field])
            self.assertEqual(scene["invalid_points"], saved["input_points"] - saved["valid_points"])
            expected_input += saved["input_points"]
            expected_valid += saved["valid_points"]
            expected_voxels += saved["voxel_count"]
            with np.load(prepared.parent / saved["path"], allow_pickle=False) as archive:
                for field, key in (("point_label_counts", "point_labels"), ("voxel_label_counts", "labels")):
                    if saved["labeled"]:
                        expected = {str(value): count for value, count in Counter(archive[key].tolist()).items()}
                        self.assertEqual(scene[field], expected)
                    else:
                        self.assertIsNone(scene[field])
        self.assertEqual(report["total_input_points"], expected_input)
        self.assertEqual(report["total_valid_points"], expected_valid)
        self.assertEqual(report["total_invalid_points"], expected_input - expected_valid)
        self.assertGreater(report["total_invalid_points"], 0)
        self.assertEqual(report["total_voxels"], expected_voxels)
        raw_report = self.success(self.run_cli("inspect", "--manifest", raw, "--json"))
        self.assertEqual(raw_report["kind"], "rptu-raw-dataset")
        self.assertEqual(raw_report["total_points"], report["total_input_points"])
        self.assertEqual(raw_report["total_invalid_points"], report["total_invalid_points"])

    def test_console_entrypoint_has_human_readable_prepared_summary(self):
        _, prepared = self.prepare()
        result = self.run_cli("inspect", "--prepared", prepared, console=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in ("Prepared dataset is valid", "input points", "valid points", "invalid rows removed", "voxels", "0.25 m"):
            self.assertIn(text, result.stdout)

    def test_inspection_requires_exactly_one_source(self):
        for args in (
            ("inspect",),
            ("inspect", "--manifest", "raw.json", "--prepared", "prepared.json"),
            ("inspect", "--prepared"),
            ("inspect", "--prepared", "missing.json"),
        ):
            with self.subTest(args=args):
                self.usage_error(self.run_cli(*args))
        help_result = self.run_cli("inspect", "--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--prepared", help_result.stdout)
        self.assertIn("--manifest", help_result.stdout)

    def test_invalid_prepared_metadata_reports_the_field(self):
        _, prepared = self.prepare()
        original = prepared.read_text(encoding="utf-8")
        for field, value in (("schema_version", 99), ("voxel_size", True), ("coordinate_unit", "inch")):
            with self.subTest(field=field):
                document = json.loads(original)
                document[field] = value
                prepared.write_text(json.dumps(document), encoding="utf-8")
                result = self.run_cli("inspect", "--prepared", prepared, "--json")
                self.usage_error(result, field)
                self.assertIn(prepared.name, result.stderr)

    def test_checksum_failure_in_last_scene_is_not_skipped(self):
        _, prepared = self.prepare()
        document = json.loads(prepared.read_text(encoding="utf-8"))
        scene = document["scenes"][-1]
        (prepared.parent / scene["path"]).write_bytes(b"changed prepared scene")
        result = self.run_cli("inspect", "--prepared", prepared, "--json")
        self.usage_error(result, "checksum")
        self.assertIn(scene["id"], result.stderr)

    def test_invalid_npz_with_correct_checksum_is_rejected_cleanly(self):
        _, prepared = self.prepare()
        document = json.loads(prepared.read_text(encoding="utf-8"))
        scene = document["scenes"][-1]
        path = prepared.parent / scene["path"]
        path.write_bytes(b"not an NPZ archive")
        scene["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared.write_text(json.dumps(document), encoding="utf-8")
        result = self.run_cli("inspect", "--prepared", prepared, "--json")
        self.usage_error(result)
        self.assertIn(scene["id"], result.stderr)

    def test_invalid_point_mapping_is_rejected_even_with_updated_checksum(self):
        _, prepared = self.prepare()
        document = json.loads(prepared.read_text(encoding="utf-8"))
        scene = document["scenes"][-1]
        path = prepared.parent / scene["path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["point_to_voxel"][0] = scene["voxel_count"]
        np.savez(path, **arrays)
        scene["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        prepared.write_text(json.dumps(document), encoding="utf-8")
        self.usage_error(
            self.run_cli("inspect", "--prepared", prepared, "--json"), "point_to_voxel",
        )

    def test_relocated_bundle_is_read_only_and_does_not_require_raw_sources(self):
        raw, prepared = self.prepare()
        expected = self.success(self.run_cli("inspect", "--prepared", prepared, "--json"))
        relocated = self.directory / "different" / "location"
        relocated.parent.mkdir()
        shutil.move(str(prepared.parent), relocated)
        shutil.rmtree(raw.parent)
        before = {path.relative_to(relocated): path.read_bytes() for path in relocated.rglob("*") if path.is_file()}
        report = self.success(self.run_cli("inspect", "--prepared", relocated / "prepared.json", "--json"))
        self.assertEqual(report, expected)
        after = {path.relative_to(relocated): path.read_bytes() for path in relocated.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(raw.parent.exists())

    def test_raw_and_prepared_inputs_are_not_interchangeable(self):
        raw, prepared = self.prepare()
        for args in (("--prepared", raw), ("--manifest", prepared)):
            with self.subTest(args=args):
                self.usage_error(self.run_cli("inspect", *args, "--json"))


if __name__ == "__main__":
    unittest.main()
