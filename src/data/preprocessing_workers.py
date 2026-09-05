"""Dispatch preprocessing jobs without importing the point-cloud dependencies."""

from functools import partial

from tqdm.contrib.concurrent import process_map


def process_file_pairs(worker, matched_file_pairs, voxel_size, max_workers):
    """Bind voxel size before dispatch, preserving input order in the results."""
    return process_map(
        partial(worker, voxel_size=voxel_size),
        matched_file_pairs,
        chunksize=1,
        max_workers=max_workers,
        desc="Processing files",
        unit="file",
    )
