"""Validated, engine-independent readers for prepared dataset schema version 1."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import struct
import zipfile
import zlib

import numpy as np

from .datasets import (
    DatasetError,
    LabelClass,
    _fields,
    _float,
    _int64,
    _name,
    _nonempty_list,
    _nonfinite,
    _object,
    _scene_path,
    sha256_file,
)


@dataclass(frozen=True)
class PreparedScene:
    id: str
    split: str
    path: Path
    sha256: str
    source_sha256: str
    input_points: int
    valid_points: int
    voxel_count: int
    labeled: bool


@dataclass(frozen=True)
class PreparedDataset:
    source: Path
    coordinate_unit: str
    feature_names: tuple[str, ...]
    classes: tuple[LabelClass, ...]
    ignore_label: int
    voxel_size: float
    source_manifest_sha256: str
    scenes: tuple[PreparedScene, ...]


@dataclass(frozen=True)
class PreparedSceneData:
    voxel_coords: np.ndarray
    features: np.ndarray
    labels: np.ndarray | None
    point_labels: np.ndarray | None
    point_indices: np.ndarray
    point_to_voxel: np.ndarray


def _checksum(value, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DatasetError(f"{location}: expected 64 lowercase hexadecimal characters")
    return value


def _positive_count(value, location: str) -> int:
    count = _int64(value, location)
    if count < 1:
        raise DatasetError(f"{location}: expected a positive integer")
    return count


def load_prepared_manifest(path: str | Path) -> PreparedDataset:
    """Read strict metadata and resolve paths without loading scene arrays."""
    try:
        source = Path(path).expanduser().resolve()
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object,
            parse_constant=_nonfinite,
            parse_float=_float,
        )
    except (OSError, RuntimeError, ValueError, TypeError, RecursionError) as error:
        raise DatasetError(f"{path}: cannot read prepared manifest: {error}") from error

    try:
        root = _fields(document, {
            "schema_version", "kind", "source_manifest_sha256", "coordinate_unit",
            "feature_names", "label_mapping", "ignore_label", "training_ignore_label",
            "voxel_size", "quantization", "feature_reduction", "label_reduction", "scenes",
        }, "prepared manifest")
        if type(root["schema_version"]) is not int or root["schema_version"] != 1:
            raise DatasetError("schema_version: expected integer 1")
        constants = {
            "kind": "rptu-prepared-dataset",
            "quantization": "floor",
            "feature_reduction": "mean",
            "label_reduction": "majority_nonignored_then_smallest_train_id",
        }
        for name, expected in constants.items():
            if root[name] != expected:
                raise DatasetError(f"{name}: expected {expected!r}")
        if type(root["training_ignore_label"]) is not int or root["training_ignore_label"] != -1:
            raise DatasetError("training_ignore_label: expected integer -1")
        source_checksum = _checksum(root["source_manifest_sha256"], "source_manifest_sha256")
        if root["coordinate_unit"] not in ("m", "cm", "mm"):
            raise DatasetError("coordinate_unit: expected m, cm or mm")
        features = tuple(
            _name(name, "feature_names")
            for name in _nonempty_list(root["feature_names"], "feature_names")
        )
        if len(features) != len(set(features)):
            raise DatasetError("feature_names: names must be unique and ordered")
        classes = []
        for index, value in enumerate(_nonempty_list(root["label_mapping"], "label_mapping")):
            location = f"label_mapping[{index}]"
            value = _fields(value, {"dataset_id", "train_id", "name"}, location)
            if _int64(value["train_id"], f"{location}.train_id") != index:
                raise DatasetError(f"{location}.train_id: expected contiguous ID {index} in declared order")
            classes.append(LabelClass(
                _int64(value["dataset_id"], f"{location}.dataset_id"),
                _name(value["name"], f"{location}.name"),
            ))
        if len({entry.id for entry in classes}) != len(classes):
            raise DatasetError("label_mapping: dataset class IDs must be unique")
        if len({entry.name for entry in classes}) != len(classes):
            raise DatasetError("label_mapping: class names must be unique")
        ignore_label = _int64(root["ignore_label"], "ignore_label")
        if ignore_label in {entry.id for entry in classes}:
            raise DatasetError("ignore_label: must differ from every dataset class ID")
        size = root["voxel_size"]
        if type(size) not in (int, float):
            raise DatasetError("voxel_size: expected a finite positive number")
        try:
            size = float(size)
        except (OverflowError, ValueError) as error:
            raise DatasetError("voxel_size: expected a finite positive number") from error
        if not math.isfinite(size) or size <= 0:
            raise DatasetError("voxel_size: expected a finite positive number")

        scenes = []
        ids = set()
        paths, hashes, source_hashes = {}, {}, {}
        for index, value in enumerate(_nonempty_list(root["scenes"], "scenes")):
            location = f"scenes[{index}]"
            value = _fields(value, {
                "id", "split", "path", "sha256", "source_sha256", "input_points",
                "valid_points", "voxel_count", "labeled",
            }, location)
            scene_id = _name(value["id"], f"{location}.id")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", scene_id):
                raise DatasetError(f"{location}.id: expected a safe scene ID using letters, digits, '_' or '-'")
            if scene_id in ids:
                raise DatasetError(f"{location}.id: duplicate scene ID {scene_id!r}")
            ids.add(scene_id)
            split = value["split"]
            if split not in ("train", "val", "test", "predict"):
                raise DatasetError(f"{location}.split: expected train, val, test or predict")
            scene_path = _scene_path(value["path"], source, f"{location}.path")
            if scene_path.suffix.lower() != ".npz":
                raise DatasetError(f"{location}.path: expected a .npz scene")
            checksum = _checksum(value["sha256"], f"{location}.sha256")
            source_hash = _checksum(value["source_sha256"], f"{location}.source_sha256")
            for seen, identity, kind in (
                (paths, scene_path, "path"),
                (hashes, checksum, "checksum"),
                (source_hashes, source_hash, "source checksum"),
            ):
                if identity in seen and seen[identity] != split:
                    raise DatasetError(f"{location}: scene {kind} is shared across different splits")
                seen[identity] = split
            counts = {
                name: _positive_count(value[name], f"{location}.{name}")
                for name in ("input_points", "valid_points", "voxel_count")
            }
            if not counts["voxel_count"] <= counts["valid_points"] <= counts["input_points"]:
                raise DatasetError(f"{location}: expected voxel_count <= valid_points <= input_points")
            labeled = value["labeled"]
            if type(labeled) is not bool:
                raise DatasetError(f"{location}.labeled: expected a boolean")
            if not labeled and split != "predict":
                raise DatasetError(f"{location}.labeled: labels are required for the {split!r} split")
            scenes.append(PreparedScene(
                scene_id, split, scene_path, checksum, source_hash,
                counts["input_points"], counts["valid_points"], counts["voxel_count"], labeled,
            ))
        return PreparedDataset(
            source, root["coordinate_unit"], features, tuple(classes),
            ignore_label, size, source_checksum, tuple(scenes),
        )
    except DatasetError as error:
        raise DatasetError(f"{source}: {error}") from error


def _read_arrays(path: Path, labeled: bool) -> dict[str, np.ndarray]:
    expected = {"voxel_coords", "features", "point_indices", "point_to_voxel"}
    if labeled:
        expected |= {"labels", "point_labels"}
    try:
        # Own the stream so malformed NPY/ZIP headers cannot leave a descriptor
        # open when NumPy fails before returning its archive context manager.
        with path.open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
            names = set(archive.files)
            if len(names) != len(archive.files):
                raise DatasetError("NPZ: duplicate array names")
            if expected - names:
                raise DatasetError(f"NPZ: missing arrays: {', '.join(sorted(expected - names))}")
            if names - expected:
                raise DatasetError(f"NPZ: unknown arrays: {', '.join(sorted(names - expected))}")
            arrays = {name: archive[name] for name in names}
            if any(not isinstance(array, np.ndarray) for array in arrays.values()):
                raise DatasetError("NPZ: each entry must contain a numeric NPY array")
            return arrays
    except (
        OSError, ValueError, TypeError, AttributeError, EOFError, KeyError,
        RuntimeError, NotImplementedError, zipfile.BadZipFile, zlib.error, struct.error,
    ) as error:
        raise DatasetError(f"cannot load numeric prepared NPZ: {error}") from error


def _typed_array(value: np.ndarray, kind: str, name: str, shape: tuple[int, ...]) -> np.ndarray:
    dtype = "float64" if kind == "f" else "signed int64"
    if value.dtype.kind != kind or value.dtype.itemsize != 8:
        raise DatasetError(f"{name}: expected {dtype} dtype")
    if value.shape != shape:
        raise DatasetError(f"{name}: expected shape {list(shape)}")
    return value.astype(np.float64 if kind == "f" else np.int64, copy=False)


def load_prepared_scene(dataset: PreparedDataset, scene: PreparedScene) -> PreparedSceneData:
    """Verify a scene's bytes and all cross-array invariants before returning it.

    Source row order and point IDs are preserved. Labels use contiguous training
    IDs; -1 is the training ignore value and also the removed-row mapping marker.
    A checksum identifies bytes, not a trusted publisher or source provenance.
    """
    try:
        if not scene.path.resolve().is_relative_to(dataset.source.parent):
            raise DatasetError("scene path must remain inside the manifest directory")
        if sha256_file(scene.path) != scene.sha256:
            raise DatasetError("SHA-256 checksum mismatch; scene bytes differ from the prepared manifest")
        arrays = _read_arrays(scene.path, scene.labeled)
        count, voxels = scene.input_points, scene.voxel_count
        coords = _typed_array(arrays["voxel_coords"], "i", "voxel_coords", (voxels, 3))
        features = _typed_array(arrays["features"], "f", "features", (voxels, len(dataset.feature_names)))
        indices = _typed_array(arrays["point_indices"], "i", "point_indices", (count,))
        mapping = _typed_array(arrays["point_to_voxel"], "i", "point_to_voxel", (count,))
        if not np.isfinite(features).all():
            raise DatasetError("features: all prepared feature values must be finite")
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        if not np.array_equal(order, np.arange(voxels)) or np.any(np.all(coords[1:] == coords[:-1], axis=1)):
            raise DatasetError("voxel_coords: expected unique coordinates in lexicographic order")
        if np.any(indices < 0) or len(np.unique(indices)) != count:
            raise DatasetError("point_indices: expected unique nonnegative original point IDs")
        if np.any((mapping < -1) | (mapping >= voxels)):
            raise DatasetError("point_to_voxel: expected -1 or an existing voxel index")
        valid = mapping >= 0
        if np.count_nonzero(valid) != scene.valid_points:
            raise DatasetError("point_to_voxel: retained row count differs from valid_points")
        if len(np.unique(mapping[valid])) != voxels:
            raise DatasetError("point_to_voxel: every voxel must be referenced by at least one source row")

        labels, point_labels = None, None
        if scene.labeled:
            labels = _typed_array(arrays["labels"], "i", "labels", (voxels,))
            point_labels = _typed_array(arrays["point_labels"], "i", "point_labels", (count,))
            for name, values in (("labels", labels), ("point_labels", point_labels)):
                if np.any((values < -1) | (values >= len(dataset.classes))):
                    raise DatasetError(f"{name}: expected -1 or a declared training class ID")
            expected = np.full(voxels, -1, dtype=np.int64)
            included = valid & (point_labels != -1)
            if np.any(included):
                pairs, frequencies = np.unique(
                    np.column_stack((mapping[included], point_labels[included])),
                    axis=0, return_counts=True,
                )
                order = np.lexsort((pairs[:, 1], -frequencies, pairs[:, 0]))
                ranked = pairs[order]
                first = np.r_[True, ranked[1:, 0] != ranked[:-1, 0]]
                expected[ranked[first, 0]] = ranked[first, 1]
            if not np.array_equal(labels, expected):
                raise DatasetError("labels: do not match majority_nonignored_then_smallest_train_id of retained point_labels")
        return PreparedSceneData(coords, features, labels, point_labels, indices, mapping)
    except (DatasetError, OSError, RuntimeError) as error:
        raise DatasetError(f"{scene.path} (scene {scene.id!r}): {error}") from error
