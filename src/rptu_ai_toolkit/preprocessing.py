"""Deterministic CPU voxelization, independent of a sparse-convolution engine."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .datasets import DatasetError, DatasetManifest, Sample, validate_sample


@dataclass(frozen=True)
class PreparedSample:
    """Voxel arrays plus a mapping for every original input row.

    ``point_to_voxel[i]`` indexes the voxel for input row ``i``; -1 marks a
    removed, nonfinite row. ``point_indices[i]`` is that row's original point
    identifier, which need not equal ``i``. Labels are contiguous training IDs,
    with -1 reserved for ignored labels.
    """

    voxel_coords: np.ndarray
    features: np.ndarray
    labels: np.ndarray | None
    point_to_voxel: np.ndarray
    point_indices: np.ndarray


def voxelize(
    sample: Sample, manifest: DatasetManifest, voxel_size: float
) -> PreparedSample:
    """Aggregate points into voxels of ``voxel_size`` in manifest units.

    Quantization uses floor, including for negative coordinates; no origin
    offset or boundary epsilon is applied. Voxels are ordered lexicographically
    by integer coordinate. Features use an arithmetic mean, while labels use
    the most frequent nonignored training ID (smallest ID on a tie).

    One shared finite-coordinate/finite-feature mask preserves array alignment.
    Unknown labels are rejected before that mask is applied. This operation
    creates new arrays and does not modify the source sample.
    """
    if isinstance(voxel_size, bool) or not isinstance(voxel_size, Real):
        raise DatasetError("voxel_size: expected a finite positive real number")
    try:
        size = float(voxel_size)
    except (OverflowError, ValueError) as error:
        raise DatasetError("voxel_size: expected a finite positive real number") from error
    if not np.isfinite(size) or size <= 0:
        raise DatasetError("voxel_size: expected a finite positive real number")

    sample = validate_sample(sample, len(manifest.feature_names))
    training_labels = (
        None if sample.labels is None else manifest.encode_labels(sample.labels)
    )
    finite = np.isfinite(sample.coords).all(axis=1) & np.isfinite(sample.features).all(axis=1)
    if not np.any(finite):
        raise DatasetError("sample: no points have finite coordinates and features")

    # The upper bound is exclusive: converting int64.max to float64 rounds up
    # to 2**63, which would otherwise allow an out-of-range value to be cast.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        quantized = np.floor(sample.coords[finite] / size)
    if (
        not np.isfinite(quantized).all()
        or np.any(quantized < -(2**63))
        or np.any(quantized >= 2**63)
    ):
        raise DatasetError("voxel coordinates: quantized values exceed the int64 range")
    voxel_coords, inverse, counts = np.unique(
        quantized.astype(np.int64), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)

    # Dividing first avoids overflowing the sum of large, finite features when
    # the mean is representable. Still reject numerical overflow explicitly.
    features = np.zeros((len(voxel_coords), len(manifest.feature_names)), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        np.add.at(features, inverse, sample.features[finite] / counts[inverse, None])
    if not np.isfinite(features).all():
        raise DatasetError("features: voxel feature mean is not finite")

    labels = None
    if training_labels is not None:
        labels = np.full(len(voxel_coords), -1, dtype=np.int64)
        valid_labels = training_labels[finite]
        included = valid_labels != -1
        if np.any(included):
            pairs, frequencies = np.unique(
                np.column_stack((inverse[included], valid_labels[included])),
                axis=0,
                return_counts=True,
            )
            order = np.lexsort((pairs[:, 1], -frequencies, pairs[:, 0]))
            ranked = pairs[order]
            first = np.r_[True, ranked[1:, 0] != ranked[:-1, 0]]
            winners = ranked[first]
            labels[winners[:, 0]] = winners[:, 1]

    point_to_voxel = np.full(len(sample.coords), -1, dtype=np.int64)
    point_to_voxel[finite] = inverse
    return PreparedSample(
        voxel_coords, features, labels, point_to_voxel, sample.point_indices.copy()
    )
