# Preprocess

The current CPU command validates scene data, removes unusable rows consistently,
and aggregates retained points into voxels:

```sh
rptu preprocess --manifest runs/demo-data/manifest.json --output runs/demo-prepared --voxel-size 0.1
```

Create the input using the [quickstart](quickstart.md), or supply your own versioned
[dataset manifest](dataset-contract.md). NPZ is supported by the base package;
ASCII and binary PLY require the optional `io` extra.

The command:

1. Validates the manifest and scene checksums, arrays, and class IDs.
2. Applies one shared finite-coordinate and finite-feature mask to aligned rows.
3. Computes voxel coordinates with `floor(coords / voxel_size)`.
4. Averages each voxel's retained features.
5. Chooses the most frequent nonignored training label, using the smallest training
   ID for a tie. Ignored-only voxels stay ignored.
6. Writes numeric scene archives and `prepared.json`, retaining original point IDs,
   original point labels when present, and the full point-to-voxel mapping.

Voxel size is expressed in the manifest's coordinate unit. Coordinates are not
implicitly normalized, translated, or converted to another unit. Negative
coordinates use floor quantization. Class zero is valid and is never implicitly
removed.

The output directory must be new. Completion metadata is written last, and failed
commands remove the output directory they created. Prepared metadata is distinct
from a raw manifest. Validate the complete output bundle with:

```sh
rptu inspect --prepared runs/demo-prepared/prepared.json
rptu inspect --prepared runs/demo-prepared/prepared.json --json
```

This checks output checksums, array types and shapes, counts, point-to-voxel
coverage, and label aggregation. The directory can be moved as one bundle, and
inspection does not need the raw files. Without those source arrays, inspection
cannot independently recompute voxel coordinates or mean features; recorded
source hashes are not proof of provenance.

The [dataset contract](dataset-contract.md) specifies numeric types, feature and
label reduction, invalid-row markers, and reconstruction rules.

Earlier documentation described PCD files paired with ASC labels, normalization,
nearest-neighbor labeling, and pickle output. Those research formats and behaviors
are not the current command's contract. A future adapter must establish point
correspondence before producing supported scene arrays.
