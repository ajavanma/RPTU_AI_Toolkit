import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import numpy as np

from rptu_ai_toolkit.data_commands import preprocess_dataset
from rptu_ai_toolkit.datasets import DatasetError, load_manifest, load_scene, sha256_file
from rptu_ai_toolkit.prepared import load_prepared_manifest, load_prepared_scene
from rptu_ai_toolkit.preprocessing import voxelize
from rptu_ai_toolkit.synthetic import generate_synthetic


class TestPreparedDatasets(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "prepared"
        (self.directory / "scenes").mkdir(parents=True)
        self.manifest_path = self.directory / "prepared.json"
        self.scene_path = self.directory / "scenes" / "one.npz"
        # Dataset class zero is real; declared order differs from dataset IDs.
        # The invalid row's class must not influence the first voxel's tie.
        self.arrays = {
            "voxel_coords": np.array([[-2, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int64),
            "features": np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64),
            "labels": np.array([1, -1, 0], dtype=np.int64),
            "point_labels": np.array([1, 2, 2, -1, -1, 0, 1], dtype=np.int64),
            "point_indices": np.array([90, 20, 70, 10, 60, 40, 30], dtype=np.int64),
            "point_to_voxel": np.array([0, 0, -1, 1, 1, 2, 2], dtype=np.int64),
        }
        self.document = {
            "schema_version": 1,
            "kind": "rptu-prepared-dataset",
            "source_manifest_sha256": "a" * 64,
            "coordinate_unit": "m",
            "feature_names": ["intensity", "height"],
            "label_mapping": [
                {"dataset_id": 42, "train_id": 0, "name": "object"},
                {"dataset_id": 0, "train_id": 1, "name": "floor"},
                {"dataset_id": 7, "train_id": 2, "name": "wall"},
            ],
            "ignore_label": 255,
            "training_ignore_label": -1,
            "voxel_size": 0.5,
            "quantization": "floor",
            "feature_reduction": "mean",
            "label_reduction": "majority_nonignored_then_smallest_train_id",
            "scenes": [{
                "id": "one", "split": "train", "path": "scenes/one.npz",
                "sha256": "b" * 64, "source_sha256": "c" * 64,
                "input_points": 7, "valid_points": 6, "voxel_count": 3, "labeled": True,
            }],
        }
        self.save_arrays(self.arrays)

    def save_document(self, document=None):
        self.manifest_path.write_text(json.dumps(self.document if document is None else document), encoding="utf-8")

    def save_arrays(self, arrays):
        np.savez(self.scene_path, **arrays)
        self.document["scenes"][0]["sha256"] = sha256_file(self.scene_path)
        self.save_document()

    def load(self):
        dataset = load_prepared_manifest(self.manifest_path)
        return load_prepared_scene(dataset, dataset.scenes[0])

    def test_reads_metadata_and_arrays_without_changing_source_order(self):
        dataset = load_prepared_manifest(self.manifest_path)
        self.assertEqual(dataset.source, self.manifest_path.resolve())
        self.assertEqual(dataset.feature_names, ("intensity", "height"))
        self.assertEqual(tuple(entry.id for entry in dataset.classes), (42, 0, 7))
        self.assertEqual(dataset.ignore_label, 255)
        self.assertEqual(dataset.voxel_size, 0.5)
        self.assertEqual(dataset.source_manifest_sha256, "a" * 64)
        self.assertEqual(dataset.scenes[0].path, self.scene_path.resolve())
        loaded = load_prepared_scene(dataset, dataset.scenes[0])
        for name, expected in self.arrays.items():
            np.testing.assert_array_equal(getattr(loaded, name), expected)

    def test_synthetic_preprocess_reader_roundtrip(self):
        source = generate_synthetic(self.root / "raw", seed=29)
        target = self.root / "generated-prepared"
        preprocess_dataset(source, target, 0.5)
        raw = load_manifest(source)
        prepared = load_prepared_manifest(target / "prepared.json")
        self.assertEqual(len(prepared.scenes), 4)
        self.assertEqual(prepared.classes, raw.classes)
        self.assertEqual(prepared.feature_names, raw.feature_names)
        for before, after in zip(raw.scenes, prepared.scenes):
            with self.subTest(scene=before.id):
                sample = load_scene(raw, before)
                expected = voxelize(sample, raw, 0.5)
                loaded = load_prepared_scene(prepared, after)
                for name in ("voxel_coords", "features", "point_indices", "point_to_voxel"):
                    np.testing.assert_array_equal(getattr(loaded, name), getattr(expected, name))
                if sample.labels is None:
                    self.assertIsNone(loaded.labels)
                    self.assertIsNone(loaded.point_labels)
                else:
                    np.testing.assert_array_equal(loaded.labels, expected.labels)
                    np.testing.assert_array_equal(loaded.point_labels, raw.encode_labels(sample.labels))

    def test_manifest_is_metadata_only_until_scene_load(self):
        self.scene_path.unlink()
        dataset = load_prepared_manifest(self.manifest_path)
        with self.assertRaisesRegex(DatasetError, "cannot read scene"):
            load_prepared_scene(dataset, dataset.scenes[0])

    def test_relocated_bundle_loads_from_unrelated_working_directory(self):
        moved = self.root / "relocated"
        shutil.move(self.directory, moved)
        cwd = Path.cwd()
        try:
            os.chdir(self.root)
            dataset = load_prepared_manifest(moved / "prepared.json")
            loaded = load_prepared_scene(dataset, dataset.scenes[0])
        finally:
            os.chdir(cwd)
        np.testing.assert_array_equal(loaded.point_indices, self.arrays["point_indices"])
        self.assertTrue(dataset.scenes[0].path.is_relative_to(moved.resolve()))

    def test_metadata_fields_and_fixed_conventions_are_strict(self):
        changes = [
            ("schema_version", True), ("schema_version", 2), ("schema_version", 1.0),
            ("kind", "dataset"), ("coordinate_unit", "feet"), ("feature_names", []),
            ("feature_names", ["x", "x"]), ("feature_names", [""]),
            ("label_mapping", []), ("ignore_label", 0), ("ignore_label", True),
            ("training_ignore_label", -1.0), ("training_ignore_label", 0),
            ("voxel_size", True), ("voxel_size", 0), ("voxel_size", -0.1),
            ("voxel_size", "0.5"), ("voxel_size", 10**400),
            ("quantization", "round"), ("feature_reduction", "sum"),
            ("label_reduction", "first"), ("scenes", []),
            ("source_manifest_sha256", "A" * 64), ("source_manifest_sha256", "a" * 63),
        ]
        for name, value in changes:
            with self.subTest(field=name, value=value):
                document = copy.deepcopy(self.document)
                document[name] = value
                self.save_document(document)
                with self.assertRaisesRegex(DatasetError, name):
                    load_prepared_manifest(self.manifest_path)
        for path in ((), ("scenes", 0), ("label_mapping", 0)):
            for action in ("extra", "missing"):
                with self.subTest(path=path, action=action):
                    document = copy.deepcopy(self.document)
                    target = document
                    for part in path:
                        target = target[part]
                    if action == "extra":
                        target["unknown"] = 1
                    else:
                        target.pop(next(iter(target)))
                    self.save_document(document)
                    with self.assertRaisesRegex(DatasetError, "unknown fields|missing fields"):
                        load_prepared_manifest(self.manifest_path)

    def test_duplicate_malformed_nonfinite_and_nonobject_json_fail(self):
        cases = [
            '{"schema_version":1,"schema_version":1}', "{", "[]", "null",
            self.manifest_path.read_text().replace('"voxel_size": 0.5', '"voxel_size": NaN'),
            self.manifest_path.read_text().replace('"voxel_size": 0.5', '"voxel_size": 1e999'),
        ]
        for text in cases:
            with self.subTest(text=text[:50]):
                self.manifest_path.write_text(text)
                with self.assertRaises(DatasetError):
                    load_prepared_manifest(self.manifest_path)
        self.manifest_path.write_bytes(b"\xff\xfe")
        with self.assertRaisesRegex(DatasetError, "cannot read prepared manifest"):
            load_prepared_manifest(self.manifest_path)

    def test_label_mapping_order_identity_and_ignore_are_checked(self):
        mutations = [
            (0, "train_id", 1), (1, "train_id", 0), (0, "train_id", False),
            (1, "dataset_id", 42), (1, "dataset_id", 2**63),
            (1, "name", "object"), (1, "name", ""),
        ]
        for index, field, value in mutations:
            with self.subTest(index=index, field=field, value=value):
                document = copy.deepcopy(self.document)
                document["label_mapping"][index][field] = value
                self.save_document(document)
                with self.assertRaisesRegex(DatasetError, "label_mapping"):
                    load_prepared_manifest(self.manifest_path)

    def test_scene_metadata_counts_split_labels_and_checksums_are_strict(self):
        changes = [
            ("id", "../bad"), ("split", "validation"), ("labeled", 1), ("labeled", False),
            ("input_points", True), ("input_points", 0), ("input_points", 5),
            ("valid_points", 0), ("valid_points", 8), ("valid_points", 2),
            ("voxel_count", -1), ("voxel_count", 7), ("voxel_count", 1.0),
            ("sha256", "no"), ("source_sha256", "C" * 64), ("path", "scenes/one.ply"),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                document = copy.deepcopy(self.document)
                document["scenes"][0][field] = value
                self.save_document(document)
                with self.assertRaisesRegex(DatasetError, "scenes\\[0\\]"):
                    load_prepared_manifest(self.manifest_path)

    def test_paths_must_be_portable_and_contained(self):
        for path in ("../outside.npz", "/outside.npz", "C:/data.npz", "scenes\\one.npz", ""):
            with self.subTest(path=path):
                document = copy.deepcopy(self.document)
                document["scenes"][0]["path"] = path
                self.save_document(document)
                with self.assertRaisesRegex(DatasetError, "path"):
                    load_prepared_manifest(self.manifest_path)
        outside = self.root / "outside.npz"
        outside.write_bytes(self.scene_path.read_bytes())
        (self.directory / "escape.npz").symlink_to(outside)
        document["scenes"][0]["path"] = "escape.npz"
        self.save_document(document)
        with self.assertRaisesRegex(DatasetError, "inside"):
            load_prepared_manifest(self.manifest_path)

    def test_changed_symlink_is_rechecked_when_loading(self):
        dataset = load_prepared_manifest(self.manifest_path)
        outside = self.root / "outside"
        shutil.move(self.scene_path.parent, outside)
        self.scene_path.parent.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(DatasetError, "inside"):
            load_prepared_scene(dataset, dataset.scenes[0])

    def test_duplicate_ids_and_cross_split_duplicate_content_fail(self):
        for field in ("id", "path", "sha256", "source_sha256"):
            with self.subTest(field=field):
                document = copy.deepcopy(self.document)
                second = copy.deepcopy(document["scenes"][0])
                second.update(id="two", split="val", path="scenes/two.npz", sha256="d" * 64, source_sha256="e" * 64)
                second[field] = document["scenes"][0][field]
                document["scenes"].append(second)
                self.save_document(document)
                with self.assertRaisesRegex(DatasetError, "duplicate scene ID|different splits"):
                    load_prepared_manifest(self.manifest_path)

    def test_same_split_duplicate_content_can_have_distinct_scene_ids(self):
        second = copy.deepcopy(self.document["scenes"][0])
        second["id"] = "two"
        self.document["scenes"].append(second)
        self.save_document()
        dataset = load_prepared_manifest(self.manifest_path)
        self.assertEqual(len(dataset.scenes), 2)
        load_prepared_scene(dataset, dataset.scenes[1])

    def test_changed_bytes_fail_checksum_before_array_parsing(self):
        self.scene_path.write_bytes(b"not an archive")
        with self.assertRaisesRegex(DatasetError, "SHA-256 checksum mismatch"):
            self.load()

    def test_malformed_archive_errors_are_actionable(self):
        for contents in (b"garbage", b"PK\x03\x04truncated", b"\x93NUMPYbroken"):
            with self.subTest(contents=contents):
                self.scene_path.write_bytes(contents)
                self.document["scenes"][0]["sha256"] = sha256_file(self.scene_path)
                self.save_document()
                with self.assertRaisesRegex(DatasetError, "scene 'one'.*numeric prepared NPZ"):
                    self.load()

    def test_missing_unknown_duplicate_and_nonarray_npz_entries_fail(self):
        for action in ("missing", "unknown", "duplicate", "nonarray"):
            with self.subTest(action=action):
                if action in ("missing", "unknown"):
                    arrays = self.arrays.copy()
                    if action == "missing":
                        arrays.pop("point_to_voxel")
                    else:
                        arrays["extra"] = np.array([1])
                    self.save_arrays(arrays)
                elif action == "duplicate":
                    self.save_arrays(self.arrays)
                    with zipfile.ZipFile(self.scene_path, "a") as archive:
                        buffer = io.BytesIO()
                        np.save(buffer, self.arrays["features"])
                        with self.assertWarns(UserWarning):
                            archive.writestr("features.npy", buffer.getvalue())
                    self.document["scenes"][0]["sha256"] = sha256_file(self.scene_path)
                    self.save_document()
                else:
                    with zipfile.ZipFile(self.scene_path, "w") as archive:
                        for name, array in self.arrays.items():
                            if name == "features":
                                # NumPy returns bytes for entries without .npy.
                                archive.writestr(name, "not a numpy array")
                            else:
                                buffer = io.BytesIO()
                                np.save(buffer, array)
                                archive.writestr(f"{name}.npy", buffer.getvalue())
                    self.document["scenes"][0]["sha256"] = sha256_file(self.scene_path)
                    self.save_document()
                with self.assertRaises(DatasetError):
                    self.load()

    def test_object_payloads_are_never_unpickled(self):
        arrays = self.arrays.copy()
        arrays["features"] = np.array([["x", "y"]] * 3, dtype=object)
        self.save_arrays(arrays)
        with self.assertRaisesRegex(DatasetError, "allow_pickle=False"):
            self.load()

    def test_dtypes_are_exact_and_both_endian_orders_are_supported(self):
        for name in self.arrays:
            wrong_types = (np.float32, np.int64, np.complex128) if name == "features" else (np.int32, np.uint64, np.float64, np.bool_)
            for dtype in wrong_types:
                with self.subTest(name=name, dtype=dtype):
                    arrays = self.arrays.copy()
                    arrays[name] = arrays[name].astype(dtype)
                    self.save_arrays(arrays)
                    with self.assertRaisesRegex(DatasetError, f"{name}: expected .*dtype"):
                        self.load()
        for byte_order in (">", "<"):
            arrays = {name: values.astype(values.dtype.newbyteorder(byte_order)) for name, values in self.arrays.items()}
            self.save_arrays(arrays)
            loaded = self.load()
            for name, expected in self.arrays.items():
                np.testing.assert_array_equal(getattr(loaded, name), expected)
                self.assertTrue(getattr(loaded, name).dtype.isnative)

    def test_shapes_and_nonfinite_features_are_rejected(self):
        for name in self.arrays:
            with self.subTest(name=name):
                arrays = self.arrays.copy()
                arrays[name] = arrays[name][:-1]
                self.save_arrays(arrays)
                with self.assertRaisesRegex(DatasetError, f"{name}: expected shape"):
                    self.load()
        for value in (np.nan, np.inf, -np.inf):
            arrays = copy.deepcopy(self.arrays)
            arrays["features"][0, 0] = value
            self.save_arrays(arrays)
            with self.assertRaisesRegex(DatasetError, "features.*finite"):
                self.load()

    def test_voxel_coordinates_must_be_unique_and_lexicographic(self):
        for coords in (
            [[0, 1, 0], [-2, 0, 0], [1, 0, 0]],
            [[-2, 0, 0], [-2, 0, 0], [1, 0, 0]],
            [[0, 0, 2], [0, 0, 1], [0, 1, 0]],
        ):
            arrays = self.arrays.copy()
            arrays["voxel_coords"] = np.array(coords, dtype=np.int64)
            self.save_arrays(arrays)
            with self.assertRaisesRegex(DatasetError, "unique coordinates in lexicographic order"):
                self.load()

    def test_original_ids_must_be_nonnegative_and_unique(self):
        for replacement in (-1, 20):
            arrays = copy.deepcopy(self.arrays)
            arrays["point_indices"][0] = replacement
            self.save_arrays(arrays)
            with self.assertRaisesRegex(DatasetError, "unique nonnegative"):
                self.load()

    def test_mapping_bounds_counts_and_unreferenced_voxels_fail(self):
        for mapping, message in (
            ([-2, 0, -1, 1, 1, 2, 2], "existing voxel index"),
            ([3, 0, -1, 1, 1, 2, 2], "existing voxel index"),
            ([-1, 0, -1, 1, 1, 2, 2], "valid_points"),
            ([0, 0, -1, 1, 1, 1, 1], "every voxel"),
        ):
            arrays = self.arrays.copy()
            arrays["point_to_voxel"] = np.array(mapping, dtype=np.int64)
            self.save_arrays(arrays)
            with self.assertRaisesRegex(DatasetError, message):
                self.load()

    def test_training_labels_reject_unknown_ids_even_on_invalid_rows(self):
        for name, position, value in (("labels", 0, 3), ("labels", 0, -2), ("point_labels", 2, 3), ("point_labels", 2, -2)):
            arrays = copy.deepcopy(self.arrays)
            arrays[name][position] = value
            self.save_arrays(arrays)
            with self.assertRaisesRegex(DatasetError, f"{name}: expected -1"):
                self.load()

    def test_majority_tie_and_allignored_targets_are_verified(self):
        for position, value in ((0, 2), (1, 0), (2, 1)):
            arrays = copy.deepcopy(self.arrays)
            arrays["labels"][position] = value
            self.save_arrays(arrays)
            with self.assertRaisesRegex(DatasetError, "majority_nonignored"):
                self.load()
        # Make class 2 the nonignored majority rather than a tie at voxel zero.
        arrays = copy.deepcopy(self.arrays)
        arrays["point_labels"][0] = 2
        arrays["labels"][0] = 2
        self.save_arrays(arrays)
        self.load()

    def test_prediction_can_be_labeled_or_unlabeled_but_arrays_follow_flag(self):
        self.document["scenes"][0]["split"] = "predict"
        self.save_document()
        self.assertIsNotNone(self.load().labels)
        self.document["scenes"][0]["labeled"] = False
        arrays = {name: value for name, value in self.arrays.items() if name not in ("labels", "point_labels")}
        self.save_arrays(arrays)
        result = self.load()
        self.assertIsNone(result.labels)
        self.assertIsNone(result.point_labels)
        for name in ("labels", "point_labels"):
            partial = arrays.copy()
            partial[name] = self.arrays[name]
            self.save_arrays(partial)
            with self.assertRaisesRegex(DatasetError, "unknown arrays"):
                self.load()
        self.document["scenes"][0]["labeled"] = True
        for name in ("labels", "point_labels"):
            partial = self.arrays.copy()
            partial.pop(name)
            self.save_arrays(partial)
            with self.assertRaisesRegex(DatasetError, "missing arrays"):
                self.load()

    def test_loader_and_returned_array_mutation_do_not_modify_artifacts(self):
        before = (self.manifest_path.read_bytes(), self.scene_path.read_bytes())
        loaded = self.load()
        loaded.features[0, 0] = 999
        loaded.point_indices[0] = 999
        self.assertEqual(before, (self.manifest_path.read_bytes(), self.scene_path.read_bytes()))
        np.testing.assert_array_equal(self.load().features, self.arrays["features"])


if __name__ == "__main__":
    unittest.main()
