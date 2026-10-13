"""Generate tiny geometric fixtures for data correctness, not model quality.

Each independent scene contains floor, wall, and object points, with explicit
edge cases for filtering and voxel aggregation. These scenes are a public CPU
demo; they provide no evidence of real-world segmentation performance.
"""

from pathlib import Path

import numpy as np

from rptu_ai_toolkit.artifacts import new_output_directory, write_json, write_npz
from rptu_ai_toolkit.datasets import DatasetError, sha256_file
from rptu_ai_toolkit import __version__


def _scene_arrays(generator: np.random.Generator, scene_number: int, labelled: bool):
    floor = np.column_stack((
        generator.uniform(-2.0, 2.0, (12, 2)),
        np.zeros(12),
    ))
    wall = np.column_stack((
        np.full(12, -2.4),
        generator.uniform(-2.0, 2.0, 12),
        generator.uniform(0.1, 2.0, 12),
    ))
    objects = generator.uniform(-0.3, 0.3, (12, 3)) + np.array([0.8, -0.8, 0.9])
    # At a 0.1 m voxel size, the first three rows collide: classes 0 and 7
    # tie after the ignored row is omitted. The next two straddle x = 0,
    # demonstrating why negative coordinates require floor, not truncation.
    edge_cases = np.array([
        [-0.19, -0.19, 0.21],
        [-0.18, -0.18, 0.22],
        [-0.17, -0.17, 0.23],
        [-0.09, 0.02, 0.02],
        [0.01, 0.02, 0.02],
        [-1.25, -0.25, -0.25],
        [np.nan, 0.0, 0.0],
        [0.0, np.inf, 0.0],
        [0.3, -0.3, 0.3],
    ], dtype=np.float64)
    coords = np.concatenate((floor, wall, objects, edge_cases))
    # Translation separates split scenes spatially; y still includes negative
    # coordinates in every scene. It is a whole multiple of the demo voxel size.
    coords[:, 0] += scene_number * 8.0
    features = generator.uniform(0.0, 1.0, (len(coords), 1)).astype(np.float64)
    features[-1, 0] = np.nan
    arrays = {
        "coords": coords.astype(np.float64),
        "features": features,
        "point_indices": np.arange(len(coords), dtype=np.int64),
    }
    if labelled:
        arrays["labels"] = np.concatenate((
            np.full(12, 0, dtype=np.int64),
            np.full(12, 7, dtype=np.int64),
            np.full(12, 42, dtype=np.int64),
            np.array([0, 7, -1, 0, 0, -1, 7, 42, 0], dtype=np.int64),
        ))
    return arrays


def generate_synthetic(output: Path | str, seed: int = 0) -> Path:
    """Create four disjoint schema-v1 demo scenes and return their manifest.

    With the same implementation and dependency versions, the same seed produces
    identical bytes regardless of output directory or generation time. An
    existing output is never changed.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise DatasetError("seed: expected an integer in 0..4294967295; booleans are not accepted")

    generator = np.random.Generator(np.random.PCG64(seed))
    with new_output_directory(output) as directory:
        (directory / "scenes").mkdir()
        scenes = []
        for index, split in enumerate(("train", "val", "test", "predict")):
            scene_id = f"{split}-000"
            relative = f"scenes/{scene_id}.npz"
            scene_path = directory / relative
            write_npz(scene_path, _scene_arrays(generator, index, labelled=split != "predict"))
            scenes.append({
                "id": scene_id,
                "split": split,
                "path": relative,
                "sha256": sha256_file(scene_path),
            })
        write_json(directory / "generation.json", {
            "schema_version": 1,
            "generator": "geometric-v1",
            "seed": seed,
            "toolkit_version": __version__,
            "numpy_version": np.__version__,
        })
        manifest = directory / "manifest.json"
        write_json(manifest, {
            "schema_version": 1,
            "coordinate_unit": "m",
            "feature_names": ["intensity"],
            "classes": [
                {"id": 0, "name": "floor"},
                {"id": 7, "name": "wall"},
                {"id": 42, "name": "object"},
            ],
            "ignore_label": -1,
            "scenes": scenes,
        })
    return manifest
