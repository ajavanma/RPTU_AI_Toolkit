import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from rptu_ai_toolkit.artifacts import new_output_directory, write_json, write_npz
from rptu_ai_toolkit.datasets import DatasetError, load_manifest, load_scene, sha256_file
from rptu_ai_toolkit.synthetic import generate_synthetic


class TestSyntheticDataset(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_same_seed_produces_identical_manifests_arrays_and_npz_bytes(self):
        first = generate_synthetic(self.root / "first", seed=23)
        # A clock lookup would make reproducibility depend on generation time.
        with patch("zipfile.time.localtime", side_effect=AssertionError("clock dependency")):
            second = generate_synthetic(self.root / "second", seed=23)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        generation = json.loads((first.parent / "generation.json").read_text())
        self.assertEqual(generation["seed"], 23)
        self.assertEqual(generation["generator"], "geometric-v1")
        self.assertEqual(generation["numpy_version"], np.__version__)
        self.assertEqual((first.parent / "generation.json").read_bytes(), (second.parent / "generation.json").read_bytes())
        first_manifest = load_manifest(first)
        second_manifest = load_manifest(second)
        for left, right in zip(first_manifest.scenes, second_manifest.scenes):
            self.assertEqual(left.sha256, right.sha256)
            self.assertEqual(left.path.read_bytes(), right.path.read_bytes())
            with np.load(left.path, allow_pickle=False) as a, np.load(right.path, allow_pickle=False) as b:
                self.assertEqual(a.files, b.files)
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name])

    def test_other_seeds_change_every_scene_and_maximum_seed_is_valid(self):
        first = load_manifest(generate_synthetic(self.root / "first", seed=0))
        second = load_manifest(generate_synthetic(self.root / "second", seed=2**32 - 1))
        for left, right in zip(first.scenes, second.scenes):
            self.assertNotEqual(left.sha256, right.sha256)

    def test_contract_integration_and_split_isolation(self):
        source = generate_synthetic(self.root / "nested" / "dataset")
        self.assertTrue(source.is_absolute())
        manifest = load_manifest(source)
        self.assertEqual(manifest.coordinate_unit, "m")
        self.assertEqual(manifest.feature_names, ("intensity",))
        self.assertEqual([(entry.id, entry.name) for entry in manifest.classes],
                         [(0, "floor"), (7, "wall"), (42, "object")])
        self.assertEqual(manifest.ignore_label, -1)
        self.assertEqual([scene.split for scene in manifest.scenes], ["train", "val", "test", "predict"])
        self.assertEqual([scene.id for scene in manifest.scenes],
                         ["train-000", "val-000", "test-000", "predict-000"])
        self.assertEqual(len({scene.sha256 for scene in manifest.scenes}), 4)
        point_sets = []
        for scene in manifest.scenes:
            self.assertEqual(scene.sha256, sha256_file(scene.path))
            sample = load_scene(manifest, scene)
            self.assertEqual(sample.coords.shape, (45, 3))
            self.assertEqual(sample.features.shape, (45, 1))
            self.assertEqual(sample.coords.dtype, np.dtype("float64"))
            self.assertEqual(sample.features.dtype, np.dtype("float64"))
            self.assertEqual(sample.point_indices.dtype, np.dtype("int64"))
            np.testing.assert_array_equal(sample.point_indices, np.arange(45))
            finite = np.isfinite(sample.coords).all(axis=1) & np.isfinite(sample.features).all(axis=1)
            self.assertEqual(int(finite.sum()), 42)
            self.assertTrue(np.any(sample.coords[finite] < 0))
            point_sets.append({tuple(row) for row in sample.coords[finite]})
            if scene.split == "predict":
                self.assertIsNone(sample.labels)
                with np.load(scene.path, allow_pickle=False) as arrays:
                    self.assertNotIn("labels", arrays.files)
            else:
                self.assertEqual(sample.labels.dtype, np.dtype("int64"))
                self.assertEqual(set(sample.labels.tolist()), {-1, 0, 7, 42})
                encoded = manifest.encode_labels(sample.labels)
                self.assertEqual(set(encoded.tolist()), {-1, 0, 1, 2})
                np.testing.assert_array_equal(manifest.decode_labels(encoded), sample.labels)
        for index, points in enumerate(point_sets):
            for other in point_sets[index + 1:]:
                self.assertFalse(points & other)

    def test_demo_contains_voxel_tie_ignored_label_and_negative_floor_cases(self):
        manifest = load_manifest(generate_synthetic(self.root / "dataset"))
        sample = load_scene(manifest, manifest.scenes[0])
        coords = sample.coords[36:39]
        np.testing.assert_array_equal(np.floor(coords / 0.1), [[-2, -2, 2]] * 3)
        np.testing.assert_array_equal(sample.labels[36:39], [0, 7, -1])
        self.assertEqual(np.floor(sample.coords[39, 0] / 0.1), -1)
        self.assertEqual(np.floor(sample.coords[40, 0] / 0.1), 0)

    def test_invalid_seeds_fail_before_creating_output_or_parents(self):
        for seed in (True, False, -1, 2**32, 0.0, "0", None, np.int64(0)):
            with self.subTest(seed=seed):
                output = self.root / "missing-parent" / "dataset"
                with self.assertRaisesRegex(DatasetError, "seed"):
                    generate_synthetic(output, seed=seed)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_outputs_and_symlinks_are_unchanged(self):
        directory = self.root / "directory"
        directory.mkdir()
        populated = self.root / "populated"
        populated.mkdir()
        marker = populated / "marker"
        marker.write_text("keep me", encoding="utf-8")
        plain_file = self.root / "file"
        plain_file.write_text("keep file", encoding="utf-8")
        symlink = self.root / "symlink"
        symlink.symlink_to(populated, target_is_directory=True)
        broken = self.root / "broken"
        broken.symlink_to(self.root / "missing-target", target_is_directory=True)
        for output in (directory, populated, plain_file, symlink, broken):
            with self.subTest(output=output.name), self.assertRaisesRegex(DatasetError, "already exists"):
                generate_synthetic(output)
        self.assertEqual(list(directory.iterdir()), [])
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep me")
        self.assertEqual(plain_file.read_text(encoding="utf-8"), "keep file")
        self.assertTrue(symlink.is_symlink())
        self.assertTrue(broken.is_symlink())
        self.assertFalse((self.root / "missing-target").exists())

    def test_failed_scene_write_removes_only_reserved_output(self):
        output = self.root / "parent" / "dataset"
        with patch("rptu_ai_toolkit.synthetic.write_npz", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(DatasetError, "disk full"):
                generate_synthetic(output)
        self.assertFalse(output.exists())
        self.assertTrue(output.parent.is_dir())

    def test_parent_creation_failure_is_actionable(self):
        parent = self.root / "file"
        parent.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "cannot create output"):
            generate_synthetic(parent / "dataset")
        self.assertEqual(parent.read_text(encoding="utf-8"), "keep")

    def test_cleanup_does_not_remove_a_replacement_directory(self):
        output = self.root / "output"
        original = self.root / "moved-original"
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with new_output_directory(output):
                output.rename(original)
                output.mkdir()
                (output / "marker").write_text("not owned", encoding="utf-8")
                raise RuntimeError("test failure")
        self.assertEqual((output / "marker").read_text(encoding="utf-8"), "not owned")
        self.assertTrue(original.is_dir())


class TestArtifactWriters(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_npz_is_deterministic_and_pickle_free(self):
        first = self.root / "first.npz"
        second = self.root / "second.npz"
        values = np.array([[1.5, -2.0]], dtype="<f8")
        write_npz(first, {"z": np.array([7], dtype="<i8"), "a": values})
        write_npz(second, {"a": values.astype(">f8"), "z": np.array([7], dtype=">i8")})
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(archive.namelist(), ["a.npy", "z.npy"])
            for entry in archive.infolist():
                self.assertEqual(entry.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(entry.compress_type, zipfile.ZIP_STORED)
        with np.load(first, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["a"], values)
            np.testing.assert_array_equal(archive["z"], [7])

    def test_npz_rejects_object_arrays_and_unsafe_names_before_writing(self):
        path = self.root / "invalid.npz"
        for arrays in ({"coords": np.array([object()], dtype=object)},
                       {"../escape": np.array([1])}, {1: np.array([1])}):
            with self.subTest(arrays=list(arrays)), self.assertRaises(DatasetError):
                write_npz(path, arrays)
            self.assertFalse(path.exists())

    def test_writers_do_not_overwrite_existing_files(self):
        path = self.root / "existing"
        path.write_bytes(b"original")
        with self.assertRaises(DatasetError):
            write_npz(path, {"coords": np.array([1])})
        with self.assertRaises(DatasetError):
            write_json(path, {"value": 1})
        self.assertEqual(path.read_bytes(), b"original")

    def test_json_order_is_stable_and_nonfinite_numbers_are_rejected(self):
        first = self.root / "first.json"
        second = self.root / "second.json"
        write_json(first, {"z": 2, "a": {"d": 4, "b": 3}})
        write_json(second, {"a": {"b": 3, "d": 4}, "z": 2})
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(json.loads(first.read_text(encoding="utf-8")), {"a": {"b": 3, "d": 4}, "z": 2})
        invalid = self.root / "invalid.json"
        with self.assertRaises(DatasetError):
            write_json(invalid, {"bad": float("nan")})
        self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
