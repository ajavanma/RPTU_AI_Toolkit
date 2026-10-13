"""Versioned, CPU-only point-cloud dataset and array contracts."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np


class DatasetError(ValueError):
    """A dataset manifest or scene does not satisfy the public contract."""


@dataclass(frozen=True)
class LabelClass:
    id: int
    name: str


@dataclass(frozen=True)
class Scene:
    id: str
    split: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class Sample:
    coords: np.ndarray
    features: np.ndarray
    labels: np.ndarray | None
    point_indices: np.ndarray


@dataclass(frozen=True)
class DatasetManifest:
    source: Path
    coordinate_unit: str
    feature_names: tuple[str, ...]
    classes: tuple[LabelClass, ...]
    ignore_label: int
    scenes: tuple[Scene, ...]

    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """Map dataset IDs to contiguous IDs in declared class order; ignore is -1."""
        values = _integer_array(labels, "labels")
        result = np.full(values.shape, -1, dtype=np.int64)
        known = values == self.ignore_label
        for training_id, label_class in enumerate(self.classes):
            selected = values == label_class.id
            result[selected] = training_id
            known |= selected
        if not np.all(known):
            unknown = np.unique(values[~known]).tolist()
            raise DatasetError(f"labels: unknown dataset class IDs: {unknown}")
        return result

    def decode_labels(self, labels: np.ndarray) -> np.ndarray:
        """Restore dataset IDs, including the dataset's ignore label."""
        values = _integer_array(labels, "training labels")
        if np.any((values < -1) | (values >= len(self.classes))):
            raise DatasetError("training labels: expected -1 or a declared training class ID")
        result = np.full(values.shape, self.ignore_label, dtype=np.int64)
        for training_id, label_class in enumerate(self.classes):
            result[values == training_id] = label_class.id
        return result


def sha256_file(path: str | Path) -> str:
    """Hash file bytes without loading the entire scene into memory."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except (OSError, ValueError) as error:
        raise DatasetError(f"{path}: cannot read scene: {error}") from error
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _nonfinite(value):
    raise DatasetError(f"nonfinite JSON number {value!r} is not allowed")


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        _nonfinite(value)
    return parsed


def _fields(value: Any, fields: set[str], location: str) -> dict:
    if not isinstance(value, dict):
        raise DatasetError(f"{location}: expected an object")
    missing = fields - value.keys()
    unknown = value.keys() - fields
    if missing:
        raise DatasetError(f"{location}: missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise DatasetError(f"{location}: unknown fields: {', '.join(sorted(unknown))}")
    return value


def _int64(value: Any, location: str) -> int:
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise DatasetError(f"{location}: expected an integer in the int64 range")
    return value


def _name(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{location}: expected a nonempty string")
    return value


def _nonempty_list(value: Any, location: str) -> list:
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{location}: expected a nonempty list")
    return value


def _scene_path(value: Any, source: Path, location: str) -> Path:
    _name(value, location)
    # A bundle uses portable relative paths, rather than host-specific drive paths.
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise DatasetError(f"{location}: expected a portable relative path using '/' separators")
    path = Path(value)
    if path.is_absolute():
        raise DatasetError(f"{location}: expected a manifest-relative path")
    try:
        resolved = (source.parent / path).resolve()
        if not resolved.is_relative_to(source.parent):
            raise DatasetError(f"{location}: path must remain inside the manifest directory")
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, DatasetError):
            raise
        raise DatasetError(f"{location}: cannot resolve path: {error}") from error
    return resolved


def load_manifest(path: str | Path) -> DatasetManifest:
    """Validate schema-v1 JSON and resolve paths without loading scene arrays."""
    try:
        source = Path(path).expanduser().resolve()
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object,
            parse_constant=_nonfinite,
            parse_float=_float,
        )
    except (OSError, RuntimeError, ValueError, TypeError, RecursionError) as error:
        raise DatasetError(f"{path}: cannot read dataset manifest: {error}") from error

    try:
        root = _fields(document, {
            "schema_version", "coordinate_unit", "feature_names", "classes", "ignore_label", "scenes"
        }, "manifest")
        if type(root["schema_version"]) is not int or root["schema_version"] != 1:
            raise DatasetError("schema_version: expected integer 1")
        if root["coordinate_unit"] not in ("m", "cm", "mm"):
            raise DatasetError("coordinate_unit: expected m, cm or mm")
        features = tuple(
            _name(name, "feature_names")
            for name in _nonempty_list(root["feature_names"], "feature_names")
        )
        if len(features) != len(set(features)):
            raise DatasetError("feature_names: names must be unique and ordered")
        classes = []
        for index, value in enumerate(_nonempty_list(root["classes"], "classes")):
            location = f"classes[{index}]"
            value = _fields(value, {"id", "name"}, location)
            classes.append(LabelClass(
                _int64(value["id"], f"{location}.id"),
                _name(value["name"], f"{location}.name"),
            ))
        if len({entry.id for entry in classes}) != len(classes):
            raise DatasetError("classes: class IDs must be unique")
        if len({entry.name for entry in classes}) != len(classes):
            raise DatasetError("classes: class names must be unique")
        ignore_label = _int64(root["ignore_label"], "ignore_label")
        if ignore_label in {entry.id for entry in classes}:
            raise DatasetError("ignore_label: must differ from every class ID")

        scenes = []
        ids = set()
        paths = {}
        hashes = {}
        for index, value in enumerate(_nonempty_list(root["scenes"], "scenes")):
            location = f"scenes[{index}]"
            value = _fields(value, {"id", "split", "path", "sha256"}, location)
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
            checksum = value["sha256"]
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise DatasetError(f"{location}.sha256: expected 64 lowercase hexadecimal characters")
            for seen, identity, kind in ((paths, scene_path, "path"), (hashes, checksum, "checksum")):
                if identity in seen and seen[identity] != split:
                    raise DatasetError(f"{location}: scene {kind} is shared across different splits")
                seen[identity] = split
            scenes.append(Scene(scene_id, split, scene_path, checksum))
        return DatasetManifest(source, root["coordinate_unit"], features, tuple(classes), ignore_label, tuple(scenes))
    except DatasetError as error:
        raise DatasetError(f"{source}: {error}") from error


def _integer_array(value: np.ndarray, location: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in "iu":
        raise DatasetError(f"{location}: expected an integer array (booleans are not accepted)")
    if value.dtype.kind == "u" and value.size and np.any(value > np.iinfo(np.int64).max):
        raise DatasetError(f"{location}: values exceed the int64 range")
    return value.astype(np.int64, copy=False)


def _real_array(value: np.ndarray, location: str) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in "iuf":
        raise DatasetError(f"{location}: expected a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        return value.astype(np.float64, copy=False)


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"coords", "features"}
            allowed = required | {"labels", "point_indices"}
            names = set(archive.files)
            if len(names) != len(archive.files):
                raise DatasetError("NPZ: duplicate array names")
            if required - names:
                raise DatasetError(f"NPZ: missing arrays: {', '.join(sorted(required - names))}")
            if names - allowed:
                raise DatasetError(f"NPZ: unknown arrays: {', '.join(sorted(names - allowed))}")
            return {name: archive[name] for name in names}
    except (OSError, ValueError, TypeError, AttributeError, EOFError, zipfile.BadZipFile) as error:
        raise DatasetError(f"{path}: cannot load numeric NPZ: {error}") from error


def _ply_arrays(path: Path, features: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        from plyfile import PlyData, PlyParseError
    except ImportError as error:
        raise DatasetError("PLY support requires the optional IO dependency: pip install 'rptu-ai-toolkit[io]'") from error
    try:
        data = PlyData.read(str(path))
        vertex = data["vertex"].data
        names = set(vertex.dtype.names or ())
        required = {"x", "y", "z", *features}
        if required - names:
            raise DatasetError(f"PLY vertex: missing properties: {', '.join(sorted(required - names))}")
        arrays = {
            "coords": np.column_stack([vertex[name] for name in ("x", "y", "z")]),
            "features": np.column_stack([vertex[name] for name in features]),
        }
        for property_name, array_name in (("label", "labels"), ("point_index", "point_indices")):
            if property_name in names:
                arrays[array_name] = vertex[property_name]
        return arrays
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, PlyParseError) as error:
        raise DatasetError(f"{path}: cannot load PLY vertices: {error}") from error


def validate_sample(sample: Sample, feature_count: int) -> Sample:
    """Normalize array types without filtering or reordering any source rows."""
    if type(feature_count) is not int or feature_count < 1:
        raise DatasetError("feature_count: expected a positive integer")
    coords = _real_array(sample.coords, "coords")
    features = _real_array(sample.features, "features")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise DatasetError("coords: expected shape [N,3]")
    count = coords.shape[0]
    if count == 0:
        raise DatasetError("coords: empty scenes are not supported")
    if features.shape != (count, feature_count):
        raise DatasetError(f"features: expected shape [{count},{feature_count}]")
    labels = None
    if sample.labels is not None:
        labels = _integer_array(sample.labels, "labels")
        if labels.shape != (count,):
            raise DatasetError(f"labels: expected shape [{count}]")
    indices = _integer_array(sample.point_indices, "point_indices")
    if indices.shape != (count,):
        raise DatasetError(f"point_indices: expected shape [{count}]")
    if np.any(indices < 0) or len(np.unique(indices)) != count:
        raise DatasetError("point_indices: expected unique nonnegative original point IDs")
    valid = np.all(np.isfinite(coords), axis=1) & np.all(np.isfinite(features), axis=1)
    if not np.any(valid):
        raise DatasetError("scene has no rows with finite coordinates and features")
    return Sample(coords, features, labels, indices)


def load_scene(manifest: DatasetManifest, scene: Scene) -> Sample:
    """Load aligned arrays; retain invalid rows for the shared preprocessing mask."""
    try:
        # Recheck containment in case a symlink changed after manifest validation.
        if not scene.path.resolve().is_relative_to(manifest.source.parent):
            raise DatasetError("scene path must remain inside the manifest directory")
        if sha256_file(scene.path) != scene.sha256:
            raise DatasetError("SHA-256 checksum mismatch; scene bytes differ from the manifest")
        if scene.path.suffix.lower() == ".npz":
            arrays = _npz_arrays(scene.path)
        elif scene.path.suffix.lower() == ".ply":
            arrays = _ply_arrays(scene.path, manifest.feature_names)
        else:
            raise DatasetError("unsupported scene format; expected .npz or .ply")
        labels = arrays.get("labels")
        if "labels" in arrays:
            # Validate every row, including nonfinite points that will later be dropped.
            manifest.encode_labels(labels)
        elif scene.split != "predict":
            raise DatasetError(f"labels: required for the {scene.split!r} split")
        coords = arrays["coords"]
        count = coords.shape[0] if coords.ndim else 0
        indices = arrays.get("point_indices", np.arange(count, dtype=np.int64))
        return validate_sample(Sample(coords, arrays["features"], labels, indices), len(manifest.feature_names))
    except (DatasetError, OSError, RuntimeError) as error:
        raise DatasetError(f"{scene.path} (scene {scene.id!r}): {error}") from error
