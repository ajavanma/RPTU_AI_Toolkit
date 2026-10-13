import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rptu_ai_toolkit.datasets import (
    DatasetError, Sample, load_manifest, load_scene, sha256_file, validate_sample,
)


class TestDatasetContract(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = self.root / "dataset.json"
        self.arrays = {
            "coords": np.array([[-1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
            "features": np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8),
            "labels": np.array([0, 42, 255], dtype=np.int32),
        }
        self.document = {
            "schema_version": 1,
            "coordinate_unit": "m",
            "feature_names": ["red", "green"],
            "classes": [{"id": 42, "name": "wall"}, {"id": 0, "name": "floor"}],
            "ignore_label": 255,
            "scenes": [{"id": "scene-1", "split": "train", "path": "scene.npz", "sha256": "0" * 64}],
        }
        self.write_scene()

    def write_manifest(self):
        self.path.write_text(json.dumps(self.document), encoding="utf-8")
        return load_manifest(self.path)

    def write_scene(self, arrays=None):
        np.savez(self.root / "scene.npz", **(self.arrays if arrays is None else arrays))
        self.document["scenes"][0]["sha256"] = sha256_file(self.root / "scene.npz")

    def read_scene(self):
        manifest = self.write_manifest()
        return load_scene(manifest, manifest.scenes[0])

    def assert_manifest_invalid(self, message):
        with self.assertRaisesRegex(DatasetError, message):
            self.write_manifest()

    def test_manifest_resolves_paths_and_retains_declared_order(self):
        manifest = self.write_manifest()
        self.assertEqual(manifest.source, self.path)
        self.assertEqual(manifest.coordinate_unit, "m")
        self.assertEqual(manifest.feature_names, ("red", "green"))
        self.assertEqual([value.id for value in manifest.classes], [42, 0])
        self.assertEqual(manifest.scenes[0].path, self.root / "scene.npz")

    def test_manifest_does_not_require_scene_files_or_create_outputs(self):
        (self.root / "scene.npz").unlink()
        self.write_manifest()
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_label_mapping_is_reversible_and_preserves_real_zero(self):
        manifest = self.write_manifest()
        original = np.array([[0, 42, 255], [42, 0, 255]], dtype=np.int64)
        encoded = manifest.encode_labels(original)
        np.testing.assert_array_equal(encoded, [[1, 0, -1], [0, 1, -1]])
        np.testing.assert_array_equal(manifest.decode_labels(encoded), original)

    def test_mapping_rejects_unknown_or_noninteger_labels(self):
        manifest = self.write_manifest()
        for values in (np.array([7]), np.array([0.0]), np.array([True]), np.array([2**64 - 1], dtype=np.uint64)):
            with self.subTest(values=values), self.assertRaises(DatasetError):
                manifest.encode_labels(values)
        for values in (np.array([-2]), np.array([2]), np.array([0.0])):
            with self.subTest(values=values), self.assertRaises(DatasetError):
                manifest.decode_labels(values)

    def test_int64_extreme_ids_round_trip(self):
        self.document["classes"] = [{"id": -(2**63), "name": "minimum"}, {"id": 2**63 - 1, "name": "maximum"}]
        manifest = self.write_manifest()
        labels = np.array([-(2**63), 2**63 - 1, 255], dtype=np.int64)
        np.testing.assert_array_equal(manifest.decode_labels(manifest.encode_labels(labels)), labels)

    def test_missing_unknown_and_wrong_manifest_fields_are_rejected(self):
        original = copy.deepcopy(self.document)
        for field in original:
            with self.subTest(missing=field):
                self.document = copy.deepcopy(original)
                del self.document[field]
                self.assert_manifest_invalid("missing fields")
        self.document = copy.deepcopy(original)
        self.document["extra"] = 1
        self.assert_manifest_invalid("unknown fields")
        self.document = []
        self.assert_manifest_invalid("expected an object")

    def test_schema_version_units_and_feature_names_are_strict(self):
        original = copy.deepcopy(self.document)
        cases = [
            ("schema_version", True), ("schema_version", 2), ("schema_version", 1.0),
            ("coordinate_unit", "meters"), ("coordinate_unit", []),
            ("feature_names", []), ("feature_names", "red"),
            ("feature_names", ["red", "red"]), ("feature_names", [" "]),
            ("feature_names", [2]),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.document = copy.deepcopy(original)
                self.document[field] = value
                self.assert_manifest_invalid(field)

    def test_labels_require_distinct_names_ids_and_ignore(self):
        original = copy.deepcopy(self.document)
        for classes in (
            [], [{"id": True, "name": "wall"}], [{"id": 2**63, "name": "wall"}],
            [{"id": -(2**63)-1, "name": "wall"}], [{"id": 1, "name": ""}],
            [{"id": 1, "name": "wall", "extra": 0}],
            [{"id": 1, "name": "wall"}, {"id": 1, "name": "floor"}],
            [{"id": 1, "name": "wall"}, {"id": 2, "name": "wall"}],
        ):
            with self.subTest(classes=classes):
                self.document = copy.deepcopy(original)
                self.document["classes"] = classes
                self.assert_manifest_invalid("classes")
        for ignored in (True, 0, 2**63):
            with self.subTest(ignore_label=ignored):
                self.document = copy.deepcopy(original)
                self.document["ignore_label"] = ignored
                self.assert_manifest_invalid("ignore_label")

    def test_scene_fields_ids_splits_and_hashes_are_strict(self):
        original = copy.deepcopy(self.document)
        cases = [
            ("id", "../scene"), ("id", "scene 1"), ("id", "_scene"), ("id", ""),
            ("split", "validation"), ("split", []), ("sha256", "A" * 64),
            ("sha256", "0" * 63), ("sha256", None), ("extra", "unknown"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                self.document = copy.deepcopy(original)
                self.document["scenes"][0][field] = value
                self.assert_manifest_invalid("scenes")
        self.document = copy.deepcopy(original)
        self.document["scenes"] = []
        self.assert_manifest_invalid("nonempty list")

    def test_duplicate_scene_ids_are_rejected_even_in_same_split(self):
        self.document["scenes"].append(copy.deepcopy(self.document["scenes"][0]))
        self.assert_manifest_invalid("duplicate scene ID")

    def test_same_path_or_content_cannot_cross_splits(self):
        original = copy.deepcopy(self.document)
        for kind in ("path", "checksum"):
            with self.subTest(kind=kind):
                self.document = copy.deepcopy(original)
                scene = copy.deepcopy(self.document["scenes"][0])
                scene.update(id="scene-2", split="val")
                if kind == "path":
                    scene["sha256"] = "1" * 64
                else:
                    scene["path"] = "copy.npz"
                self.document["scenes"].append(scene)
                self.assert_manifest_invalid("shared across different splits")

    def test_same_content_in_same_split_is_allowed_with_distinct_scene_ids(self):
        scene = copy.deepcopy(self.document["scenes"][0])
        scene["id"] = "scene-2"
        self.document["scenes"].append(scene)
        self.assertEqual(len(self.write_manifest().scenes), 2)

    def test_paths_must_remain_inside_portable_bundle(self):
        for path in (str(self.root / "scene.npz"), "../outside.npz", "C:/scene.npz", "folder\\scene.npz", ""):
            with self.subTest(path=path):
                self.document["scenes"][0]["path"] = path
                self.assert_manifest_invalid("path")

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "link").symlink_to(Path(outside), target_is_directory=True)
            self.document["scenes"][0]["path"] = "link/scene.npz"
            self.assert_manifest_invalid("inside the manifest directory")

    def test_json_duplicate_nonfinite_and_malformed_values_are_rejected(self):
        for text in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{'):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(DatasetError, "cannot read dataset manifest"):
                    load_manifest(self.path)

    def test_npz_normalizes_arrays_without_rescaling_or_reordering(self):
        sample = self.read_scene()
        np.testing.assert_array_equal(sample.coords, self.arrays["coords"])
        np.testing.assert_array_equal(sample.features, self.arrays["features"])
        np.testing.assert_array_equal(sample.labels, [0, 42, 255])
        np.testing.assert_array_equal(sample.point_indices, [0, 1, 2])
        self.assertEqual(sample.coords.dtype, np.float64)
        self.assertEqual(sample.features.dtype, np.float64)
        self.assertEqual(sample.labels.dtype, np.int64)
        self.assertEqual(sample.point_indices.dtype, np.int64)

    def test_nonfinite_rows_and_explicit_point_ids_remain_aligned(self):
        self.arrays["coords"][0, 0] = np.nan
        self.arrays["features"] = self.arrays["features"].astype(float)
        self.arrays["features"][1, 0] = np.inf
        self.arrays["point_indices"] = np.array([90, 20, 10], dtype=np.uint32)
        self.write_scene()
        sample = self.read_scene()
        self.assertTrue(np.isnan(sample.coords[0, 0]))
        self.assertTrue(np.isinf(sample.features[1, 0]))
        np.testing.assert_array_equal(sample.labels, [0, 42, 255])
        np.testing.assert_array_equal(sample.point_indices, [90, 20, 10])

    def test_unknown_labels_are_rejected_even_on_nonfinite_points(self):
        self.arrays["coords"][0, 0] = np.nan
        self.arrays["labels"][0] = 999
        self.write_scene()
        with self.assertRaisesRegex(DatasetError, "unknown dataset class IDs"):
            self.read_scene()

    def test_unlabeled_prediction_is_valid_but_supervised_splits_require_labels(self):
        del self.arrays["labels"]
        self.write_scene()
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                self.document["scenes"][0]["split"] = split
                with self.assertRaisesRegex(DatasetError, "labels: required"):
                    self.read_scene()
        self.document["scenes"][0]["split"] = "predict"
        self.assertIsNone(self.read_scene().labels)

    def test_checksums_missing_files_and_unsupported_extensions_fail(self):
        manifest = self.write_manifest()
        (self.root / "scene.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(DatasetError, "checksum mismatch"):
            load_scene(manifest, manifest.scenes[0])
        (self.root / "scene.npz").unlink()
        with self.assertRaisesRegex(DatasetError, "cannot read scene"):
            load_scene(manifest, manifest.scenes[0])
        (self.root / "scene.bin").write_bytes(b"data")
        self.document["scenes"][0].update(path="scene.bin", sha256=sha256_file(self.root / "scene.bin"))
        with self.assertRaisesRegex(DatasetError, "unsupported scene format"):
            self.read_scene()

    def test_npz_missing_unknown_or_pickled_arrays_are_rejected(self):
        original = copy.deepcopy(self.arrays)
        for case in ("missing", "unknown", "object"):
            with self.subTest(case=case):
                arrays = copy.deepcopy(original)
                if case == "missing":
                    del arrays["features"]
                elif case == "unknown":
                    arrays["extra"] = np.array([1])
                else:
                    arrays["features"] = np.array([[object(), object()]] * 3, dtype=object)
                self.write_scene(arrays)
                with self.assertRaisesRegex(DatasetError, "cannot load numeric NPZ"):
                    self.read_scene()

    def test_corrupt_npz_and_npy_disguised_as_npz_fail_clearly(self):
        scene = self.root / "scene.npz"
        for content in (b"not an archive", None):
            with self.subTest(content=content):
                if content is None:
                    with scene.open("wb") as stream:
                        np.save(stream, self.arrays["coords"], allow_pickle=False)
                else:
                    scene.write_bytes(content)
                self.document["scenes"][0]["sha256"] = sha256_file(scene)
                with self.assertRaisesRegex(DatasetError, "cannot load numeric NPZ"):
                    self.read_scene()

    def test_array_shape_type_and_point_id_violations_fail(self):
        cases = [
            ("coords", np.zeros((3, 2))), ("coords", np.array(1)),
            ("coords", np.full((3, 3), True)), ("coords", np.zeros((3, 3), dtype=complex)),
            ("features", np.zeros((2, 2))), ("features", np.zeros((3, 1))),
            ("features", np.full((3, 2), "2")),
            ("labels", np.array([[0], [42], [255]])),
            ("labels", np.array([0.0, 42.0, 255.0])),
            ("point_indices", np.array([0, 0, 1])), ("point_indices", np.array([-1, 0, 1])),
            ("point_indices", np.array([0, 1])), ("point_indices", np.array([0., 1., 2.])),
            ("point_indices", np.array([0, 1, 2**64 - 1], dtype=np.uint64)),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                arrays = copy.deepcopy(self.arrays)
                arrays[field] = value
                self.write_scene(arrays)
                with self.assertRaisesRegex(DatasetError, field):
                    self.read_scene()

    def test_empty_and_all_nonfinite_samples_fail(self):
        self.write_scene({"coords": np.zeros((0, 3)), "features": np.zeros((0, 2)), "labels": np.zeros(0, dtype=int)})
        with self.assertRaisesRegex(DatasetError, "empty scenes"):
            self.read_scene()
        self.arrays["coords"][:] = np.nan
        self.write_scene()
        with self.assertRaisesRegex(DatasetError, "no rows with finite"):
            self.read_scene()

    def test_sample_validator_rejects_bad_feature_count(self):
        sample = Sample(self.arrays["coords"], self.arrays["features"], self.arrays["labels"], np.arange(3))
        for count in (0, True, 2.0):
            with self.subTest(count=count), self.assertRaisesRegex(DatasetError, "feature_count"):
                validate_sample(sample, count)

    def write_ply(self, labels=True):
        label_header = "property int label\n" if labels else ""
        label_values = [" 0", " 42", " 255"] if labels else ["", "", ""]
        text = (
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar green\nproperty uchar red\n" + label_header +
            "property uint point_index\nend_header\n" +
            "-1 0 0 20 10" + label_values[0] + " 90\n" +
            "0 1 0 40 30" + label_values[1] + " 20\n" +
            "1 0 0 60 50" + label_values[2] + " 10\n"
        )
        path = self.root / "scene.ply"
        path.write_text(text, encoding="ascii")
        self.document["scenes"][0].update(path="scene.ply", sha256=sha256_file(path))

    def test_missing_ply_extra_has_an_actionable_error(self):
        self.write_ply()
        with patch.dict("sys.modules", {"plyfile": None}):
            with self.assertRaisesRegex(DatasetError, r"rptu-ai-toolkit\[io\]"):
                self.read_scene()

    @unittest.skipUnless(importlib.util.find_spec("plyfile"), "optional plyfile is not installed")
    def test_ply_properties_use_manifest_feature_order_and_preserve_ids(self):
        self.write_ply()
        sample = self.read_scene()
        np.testing.assert_array_equal(sample.coords, self.arrays["coords"])
        np.testing.assert_array_equal(sample.features, self.arrays["features"])
        np.testing.assert_array_equal(sample.labels, [0, 42, 255])
        np.testing.assert_array_equal(sample.point_indices, [90, 20, 10])

    @unittest.skipUnless(importlib.util.find_spec("plyfile"), "optional plyfile is not installed")
    def test_ply_without_labels_is_valid_for_prediction(self):
        self.write_ply(labels=False)
        self.document["scenes"][0]["split"] = "predict"
        self.assertIsNone(self.read_scene().labels)

    @unittest.skipUnless(importlib.util.find_spec("plyfile"), "optional plyfile is not installed")
    def test_binary_ply_defaults_to_vertex_row_point_ids(self):
        from plyfile import PlyData, PlyElement

        vertices = np.array(
            [(-1, 0, 0, 10, 20, 0), (0, 1, 0, 30, 40, 42), (1, 0, 0, 50, 60, 255)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("label", "i4")],
        )
        path = self.root / "binary.ply"
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))
        self.document["scenes"][0].update(path="binary.ply", sha256=sha256_file(path))
        sample = self.read_scene()
        np.testing.assert_array_equal(sample.features, self.arrays["features"])
        np.testing.assert_array_equal(sample.point_indices, [0, 1, 2])

    @unittest.skipUnless(importlib.util.find_spec("plyfile"), "optional plyfile is not installed")
    def test_ply_missing_or_list_valued_features_are_rejected(self):
        self.write_ply()
        self.document["feature_names"] = ["missing"]
        with self.assertRaisesRegex(DatasetError, "missing properties: missing"):
            self.read_scene()
        path = self.root / "scene.ply"
        path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property list uchar float red\nproperty int label\nend_header\n"
            "0 0 0 2 10 20 0\n", encoding="ascii",
        )
        self.document["feature_names"] = ["red"]
        self.document["scenes"][0]["sha256"] = sha256_file(path)
        with self.assertRaisesRegex(DatasetError, "features: expected a real numeric array"):
            self.read_scene()

    @unittest.skipUnless(importlib.util.find_spec("plyfile"), "optional plyfile is not installed")
    def test_malformed_ply_header_truncated_rows_and_bad_numbers_fail_clearly(self):
        for case in ("header", "truncated", "number", "overflow", "signed_overflow"):
            with self.subTest(case=case):
                self.write_ply()
                path = self.root / "scene.ply"
                text = path.read_text(encoding="ascii")
                if case == "header":
                    text = text.replace("format ascii 1.0", "bogus")
                elif case == "truncated":
                    text = text.rsplit("\n", 2)[0] + "\n"
                elif case == "number":
                    text = text.replace("-1 0 0 20", "nonnumeric 0 0 20")
                elif case == "overflow":
                    text = text.replace("-1 0 0 20 10", "-1 0 0 20 300")
                else:
                    text = text.replace("-1 0 0 20 10 0 90", "-1 0 0 20 10 9999999999999999999999999 90")
                path.write_text(text, encoding="ascii")
                self.document["scenes"][0]["sha256"] = sha256_file(path)
                with self.assertRaisesRegex(DatasetError, "cannot load PLY vertices"):
                    self.read_scene()


if __name__ == "__main__":
    unittest.main()
