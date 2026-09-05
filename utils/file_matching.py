"""Match point clouds to label files without reading configuration or the filesystem."""

import os
from pathlib import Path


def files_match_making(pcd_files, asc_files):
    """Return pairs with identical stems, in deterministic PCD path order.

    Preserve the supplied path objects and use the first sorted ASC path if
    several label files have the same stem.
    """
    asc_by_stem = {}
    for asc_file in sorted(asc_files, key=os.fspath):
        asc_by_stem.setdefault(Path(asc_file).stem, asc_file)

    matched_pairs = []
    for pcd_file in sorted(pcd_files, key=os.fspath):
        asc_file = asc_by_stem.get(Path(pcd_file).stem)
        if asc_file is not None:
            matched_pairs.append((pcd_file, asc_file))
        else:
            print(f"Warning: No corresponding ASC file found for PCD file {pcd_file}.")
    return matched_pairs
