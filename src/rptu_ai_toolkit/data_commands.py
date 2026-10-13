"""CPU dataset workflows shared by the command-line entry points."""

from pathlib import Path

import numpy as np

from .artifacts import new_output_directory, write_json, write_npz
from .datasets import load_manifest, load_scene, sha256_file
from .preprocessing import voxelize


def _label_mapping(manifest):
    return [
        {"dataset_id": label.id, "train_id": index, "name": label.name}
        for index, label in enumerate(manifest.classes)
    ]


def inspect_dataset(path: str | Path) -> dict:
    """Validate every scene and report raw point counts without modifying data."""
    manifest = load_manifest(path)
    scenes = []
    for scene in manifest.scenes:
        sample = load_scene(manifest, scene)
        finite = np.isfinite(sample.coords).all(axis=1) & np.isfinite(sample.features).all(axis=1)
        counts = None
        if sample.labels is not None:
            values, numbers = np.unique(sample.labels, return_counts=True)
            counts = {str(int(value)): int(count) for value, count in zip(values, numbers)}
        scenes.append({
            "id": scene.id,
            "split": scene.split,
            "points": len(sample.coords),
            "valid_points": int(finite.sum()),
            "invalid_points": int((~finite).sum()),
            "labeled": sample.labels is not None,
            "label_counts": counts,
            "sha256": scene.sha256,
        })
    return {
        "schema_version": 1,
        "coordinate_unit": manifest.coordinate_unit,
        "feature_names": list(manifest.feature_names),
        "label_mapping": _label_mapping(manifest),
        "ignore_label": manifest.ignore_label,
        "scenes": scenes,
        "total_points": sum(scene["points"] for scene in scenes),
        "total_invalid_points": sum(scene["invalid_points"] for scene in scenes),
    }


def preprocess_dataset(path: str | Path, output: str | Path, voxel_size: float) -> dict:
    """Write a new prepared dataset, keeping every source row's voxel mapping."""
    manifest = load_manifest(path)
    source_checksum = sha256_file(manifest.source)
    report = {
        "schema_version": 1,
        "kind": "rptu-prepared-dataset",
        "source_manifest_sha256": source_checksum,
        "coordinate_unit": manifest.coordinate_unit,
        "feature_names": list(manifest.feature_names),
        "label_mapping": _label_mapping(manifest),
        "ignore_label": manifest.ignore_label,
        "training_ignore_label": -1,
        "voxel_size": voxel_size,
        "quantization": "floor",
        "feature_reduction": "mean",
        "label_reduction": "majority_nonignored_then_smallest_train_id",
        "scenes": [],
    }
    with new_output_directory(output) as directory:
        (directory / "scenes").mkdir()
        for scene in manifest.scenes:
            sample = load_scene(manifest, scene)
            prepared = voxelize(sample, manifest, voxel_size)
            report["voxel_size"] = float(voxel_size)
            relative_path = f"scenes/{scene.id}.npz"
            target = directory / relative_path
            arrays = {
                "voxel_coords": prepared.voxel_coords,
                "features": prepared.features,
                "point_to_voxel": prepared.point_to_voxel,
                "point_indices": prepared.point_indices,
            }
            if prepared.labels is not None:
                arrays["labels"] = prepared.labels
                # Preserve original-row targets for later point-level evaluation;
                # voxel majority labels alone would erase collision information.
                arrays["point_labels"] = manifest.encode_labels(sample.labels)
            write_npz(target, arrays)
            report["scenes"].append({
                "id": scene.id,
                "split": scene.split,
                "path": relative_path,
                "sha256": sha256_file(target),
                "source_sha256": scene.sha256,
                "input_points": len(sample.coords),
                "valid_points": int((prepared.point_to_voxel >= 0).sum()),
                "voxel_count": len(prepared.voxel_coords),
                "labeled": prepared.labels is not None,
            })
        # Write completion metadata last. Failed runs remove their owned output;
        # successful output directories are never overwritten by a later call.
        write_json(directory / "prepared.json", report)
    return report
