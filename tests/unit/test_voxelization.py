from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from rptu_ai_toolkit.datasets import DatasetError, DatasetManifest, LabelClass, Sample
from rptu_ai_toolkit.preprocessing import voxelize


class TestVoxelization(unittest.TestCase):
    def setUp(self):
        self.manifest = DatasetManifest(
            source=Path("manifest.json"),
            coordinate_unit="m",
            feature_names=("intensity", "height"),
            classes=(LabelClass(0, "floor"), LabelClass(7, "wall"), LabelClass(42, "roof")),
            ignore_label=255,
            scenes=(),
        )

    def sample(self, coords, features=None, labels=None, point_indices=None):
        coords = np.asarray(coords, dtype=np.float64)
        count = len(coords)
        return Sample(
            coords=coords,
            features=np.ones((count, 2)) if features is None else np.asarray(features, dtype=np.float64),
            labels=None if labels is None else np.asarray(labels, dtype=np.int64),
            point_indices=np.arange(count) if point_indices is None else np.asarray(point_indices, dtype=np.int64),
        )

    def test_floor_boundaries_negative_coordinates_and_lexicographic_order(self):
        sample = self.sample(
            [[1, 0, 0], [-0.01, 0, 0], [0, 0, 0], [0.49, 0, 0], [-0.5, 0, 0], [-0.51, 0, 0]],
            features=[[10, 100], [2, 20], [4, 40], [6, 60], [8, 80], [12, 120]],
            labels=[42, 0, 7, 0, 7, 42],
            point_indices=[42, 4, 8, 12, 91, 3],
        )
        result = voxelize(sample, self.manifest, 0.5)
        np.testing.assert_array_equal(result.voxel_coords, [[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [2, 0, 0]])
        np.testing.assert_array_equal(result.features, [[12, 120], [5, 50], [5, 50], [10, 100]])
        np.testing.assert_array_equal(result.labels, [2, 0, 0, 2])
        np.testing.assert_array_equal(result.point_to_voxel, [3, 1, 2, 2, 1, 0])
        np.testing.assert_array_equal(result.point_indices, sample.point_indices)
        self.assertEqual(result.voxel_coords.dtype, np.dtype("int64"))
        self.assertEqual(result.features.dtype, np.dtype("float64"))
        self.assertEqual(result.labels.dtype, np.dtype("int64"))
        self.assertEqual(result.point_to_voxel.dtype, np.dtype("int64"))
        self.assertEqual(result.point_indices.dtype, np.dtype("int64"))

    def test_one_finite_mask_preserves_original_row_alignment(self):
        sample = self.sample(
            [[0, 0, 0], [np.nan, 0, 0], [1, 0, 0], [0, np.inf, 0], [2, 0, 0], [1.1, 0, 0]],
            features=[[1, 10], [2, 20], [3, 30], [4, 40], [np.inf, 50], [7, 70]],
            labels=[0, 42, 7, 42, 42, 7],
            point_indices=[99, 4, 87, 12, 81, 62],
        )
        result = voxelize(sample, self.manifest, 1.0)
        np.testing.assert_array_equal(result.voxel_coords, [[0, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(result.features, [[1, 10], [5, 50]])
        np.testing.assert_array_equal(result.labels, [0, 1])
        np.testing.assert_array_equal(result.point_to_voxel, [0, -1, 1, -1, -1, 1])
        np.testing.assert_array_equal(result.point_indices, [99, 4, 87, 12, 81, 62])

    def test_unknown_label_is_rejected_even_on_nonfinite_row(self):
        sample = self.sample([[0, 0, 0], [np.nan, 0, 0]], labels=[0, 999])
        with self.assertRaisesRegex(DatasetError, "unknown.*999"):
            voxelize(sample, self.manifest, 1.0)

    def test_majority_excludes_ignore_and_uses_training_id_for_ties(self):
        manifest = replace(self.manifest, classes=(LabelClass(42, "roof"), LabelClass(0, "floor"), LabelClass(7, "wall")))
        sample = self.sample(
            [[0, 0, 0]] * 5 + [[1, 0, 0]] * 2 + [[2, 0, 0]] * 2,
            labels=[255, 255, 0, 7, 7, 255, 255, 0, 42],
        )
        result = voxelize(sample, manifest, 1)
        # The ignored majority does not win, all-ignored remains ignored, and
        # the tie picks class 42 because declared training order is authoritative.
        np.testing.assert_array_equal(result.labels, [2, -1, 0])

    def test_unlabeled_prediction_sample_stays_unlabeled(self):
        sample = self.sample([[0, 0, 0], [0.5, 0, 0]], features=[[2, 4], [4, 8]])
        result = voxelize(sample, self.manifest, 1)
        self.assertIsNone(result.labels)
        np.testing.assert_array_equal(result.features, [[3, 6]])
        np.testing.assert_array_equal(result.point_to_voxel, [0, 0])

    def test_mean_of_large_features_does_not_overflow_naive_sum(self):
        maximum = np.finfo(np.float64).max
        sample = self.sample([[0, 0, 0]] * 2, features=[[maximum, -maximum]] * 2)
        result = voxelize(sample, self.manifest, 1)
        np.testing.assert_array_equal(result.features, [[maximum, -maximum]])

    def test_numerical_feature_mean_overflow_is_rejected(self):
        maximum = np.finfo(np.float64).max
        sample = self.sample([[0, 0, 0]] * 3, features=[[maximum, -maximum]] * 3)
        with self.assertRaisesRegex(DatasetError, "feature mean is not finite"):
            voxelize(sample, self.manifest, 1)

    def test_voxelization_does_not_mutate_or_alias_input_arrays(self):
        sample = self.sample([[0, 0, 0], [np.nan, 0, 0]], labels=[0, 255])
        originals = {name: getattr(sample, name).copy() for name in ("coords", "features", "labels", "point_indices")}
        result = voxelize(sample, self.manifest, 1)
        for name, original in originals.items():
            np.testing.assert_array_equal(getattr(sample, name), original)
        result.point_indices[0] = 99
        result.features[0, 0] = 99
        result.labels[0] = 99
        np.testing.assert_array_equal(sample.point_indices, originals["point_indices"])
        np.testing.assert_array_equal(sample.features, originals["features"])
        np.testing.assert_array_equal(sample.labels, originals["labels"])

    def test_invalid_voxel_sizes_are_rejected(self):
        sample = self.sample([[0, 0, 0]])
        for size in (True, False, np.bool_(True), 0, -1, np.nan, np.inf, -np.inf, "1", 1j, None, 10**400):
            with self.subTest(size=size), self.assertRaisesRegex(DatasetError, "voxel_size"):
                voxelize(sample, self.manifest, size)

    def test_int64_coordinate_limits_are_checked_before_cast(self):
        for coordinate in (float(2**63), np.nextafter(float(-(2**63)), -np.inf)):
            with self.subTest(coordinate=coordinate):
                sample = self.sample([[coordinate, 0, 0]])
                with self.assertRaisesRegex(DatasetError, "int64 range"):
                    voxelize(sample, self.manifest, 1)
        accepted = self.sample([[float(-(2**63)), 0, 0], [np.nextafter(float(2**63), 0), 0, 0]])
        result = voxelize(accepted, self.manifest, 1)
        np.testing.assert_array_equal(result.voxel_coords[:, 0], [-(2**63), 2**63 - 1024])

    def test_division_overflow_is_an_actionable_error(self):
        sample = self.sample([[1, 0, 0]])
        with self.assertRaisesRegex(DatasetError, "int64 range"):
            voxelize(sample, self.manifest, float.fromhex("0x0.0000000000001p-1022"))

    def test_empty_and_all_nonfinite_samples_are_rejected(self):
        for sample in (
            self.sample(np.empty((0, 3))),
            self.sample([[np.nan, 0, 0], [np.inf, 0, 0]]),
            self.sample([[0, 0, 0]], features=[[np.nan, 1]]),
        ):
            with self.subTest(sample=sample), self.assertRaises(DatasetError):
                voxelize(sample, self.manifest, 1)

    def test_public_api_rejects_misaligned_arrays(self):
        sample = self.sample([[0, 0, 0], [1, 1, 1]], labels=[0, 7])
        changes = (
            {"coords": np.ones((2, 2))},
            {"coords": np.ones((2, 3), dtype=complex)},
            {"features": np.ones((1, 2))},
            {"features": np.ones((2, 3))},
            {"labels": np.array([0])},
            {"labels": np.array([0.0, 7.0])},
            {"point_indices": np.array([0])},
            {"point_indices": np.array([2, 2])},
            {"point_indices": np.array([0.0, 1.0])},
            {"point_indices": np.array([0, 2**63], dtype=np.uint64)},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(DatasetError):
                voxelize(replace(sample, **change), self.manifest, 1)

    def test_repeated_voxelization_is_deterministic(self):
        sample = self.sample(
            [[2, 1, 0], [-1, 3, 0], [2, 0, 1], [2, 0, 0], [2.1, 0, 0]],
            labels=[0, 7, 42, 0, 7],
        )
        first = voxelize(sample, self.manifest, np.float64(1))
        second = voxelize(sample, self.manifest, 1)
        for name in ("voxel_coords", "features", "labels", "point_to_voxel", "point_indices"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        np.testing.assert_array_equal(first.voxel_coords, [[-1, 3, 0], [2, 0, 0], [2, 0, 1], [2, 1, 0]])


if __name__ == "__main__":
    unittest.main()
