# Dataset contract, version 1

The CPU data pipeline uses a JSON manifest and numeric scene files. It does not
load Python pickles, initialize CUDA, or require a model. Synthetic fixtures
exercise data correctness; they do not establish useful segmentation accuracy.

## Manifest

Each dataset is a portable directory containing its manifest and scene files.
Paths are relative to the manifest, independent of the command's working
directory. Absolute paths, Windows drive paths, backslash separators, and paths
or symlinks escaping the directory are rejected.

```json
{
  "schema_version": 1,
  "coordinate_unit": "m",
  "feature_names": ["intensity"],
  "classes": [
    {"id": 0, "name": "floor"},
    {"id": 7, "name": "wall"},
    {"id": 42, "name": "object"}
  ],
  "ignore_label": -1,
  "scenes": [
    {
      "id": "train-000",
      "split": "train",
      "path": "scenes/train-000.npz",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
```

The example checksum is a placeholder. Replace it with the SHA-256 of the actual
file bytes; the synthetic generator does this automatically. Scene loading
checks the checksum before parsing arrays. Checksums detect changed bytes; they
do not certify dataset origin or trust.

All displayed fields are required, and unknown fields are rejected. JSON objects
must not repeat a field. Malformed JSON and nonfinite JSON numbers are rejected.

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1`; booleans and other versions are rejected. |
| `coordinate_unit` | `m`, `cm`, or `mm`. No automatic conversion or normalization. |
| `feature_names` | Nonempty ordered list of distinct nonempty strings. |
| `classes` | Nonempty ordered list; names and signed 64-bit integer IDs must each be unique. |
| `ignore_label` | Signed 64-bit integer outside the declared class IDs. |
| `scenes` | Nonempty list; each entry has exactly `id`, `split`, `path`, and `sha256`. |
| Scene `id` | Globally unique; matches `[A-Za-z0-9][A-Za-z0-9_-]*`. |
| Scene `split` | `train`, `val`, `test`, or `predict`. |
| Scene `sha256` | Exactly 64 lowercase hexadecimal characters. |

Assign whole source scenes to splits before producing spatial chunks. The same
resolved path or byte checksum cannot appear in different splits, including
`predict`. Repeated content within one split is allowed if scene IDs differ.
This check catches exact duplicates; re-encoding or changing a scene can change
its checksum, so it cannot prove that independently supplied scenes have no
overlapping physical content.

Manifest validation checks metadata and resolves paths. Loading a scene also
requires its file to exist and checks its bytes and arrays. This distinction lets
tools inspect a manifest before all data have been copied.

## Scene arrays

NPZ is the default interchange format. Each archive contains exactly the
required arrays and any supported optional arrays below. Object/pickled arrays,
unknown array names, and duplicate array names are rejected.

| Array | Shape | Meaning |
| --- | --- | --- |
| `coords` | `[N, 3]` | Real numeric x, y, z coordinates in the declared unit. |
| `features` | `[N, F]` | Real numeric values in `feature_names` order. |
| `labels` | `[N]` | Integer dataset class IDs, including the declared ignore label. |
| `point_indices` | `[N]` | Optional unique nonnegative integer point IDs; defaults to `0..N-1`. |

Coordinates and features normalize to float64; integer arrays normalize safely
to int64. Complex, boolean, string and object arrays are rejected. Floating-point
labels are rejected even when every value is mathematically an integer. Unsigned
integer labels/IDs must fit the signed int64 range. Features are neither scaled
nor interpreted as colors automatically.

Labels are mandatory for `train`, `val`, and `test`. They may be absent for
`predict`; if provided, they obey the same class contract. Unknown dataset class
IDs fail before filtering, including labels on nonfinite points. Empty scenes or
scenes with no row having finite coordinates and features fail clearly.

Loading preserves row order and nonfinite rows when at least one usable row
exists. Preprocessing applies one shared finite-coordinate and finite-feature
mask to all aligned arrays. Ignored labels do not cause points to be removed.
There is no implicit removal of class zero.

An explicit `point_indices` value is a point's external identity, not an array
offset. For example, IDs `[90, 20, 10]` remain in that order; data processing never
sorts rows by point ID.

## Label mapping

Class declaration order defines training IDs `0..C-1`. In the example, dataset
IDs `0`, `7`, and `42` map to training IDs `0`, `1`, and `2`. The dataset ignore
label maps separately to training ID `-1`. The ignore value cannot also be a
declared class, and no class has special meaning because of its numeric ID.

`DatasetManifest.encode_labels()` rejects unknown dataset IDs.
`DatasetManifest.decode_labels()` restores the original IDs, including the
ignore label, and rejects training IDs outside `-1..C-1`. Preserve manifest class
order alongside any future model or prepared artifact.

## PLY input

Install `rptu-ai-toolkit[io]` to enable the optional `plyfile` reader. Both ASCII
and binary PLY are supported by that reader. A scene must contain a `vertex`
element with scalar `x`, `y`, and `z` properties and every property named in
`feature_names`. Feature columns follow manifest order regardless of their order
in the PLY header. Additional properties/elements are ignored.

The optional scalar `label` property supplies dataset labels and the optional
integer `point_index` property supplies original point IDs. Without the latter,
the reader assigns IDs from vertex row order. The same dtype, alignment, split,
class and finite-row rules apply as for NPZ. List-valued properties cannot be
used as coordinates, features, labels, or point IDs.

PCD/ASC research data and the old pickle format are not accepted by this schema.
An adapter must first establish point correspondence and emit the documented
arrays; independently filtering coordinate and label files would lose alignment.

## Voxel mapping

Voxel size uses the manifest's coordinate unit. For example, `0.1` means ten
centimeters for a dataset in meters, but one millimeter for a dataset in
centimeters. Integer voxel coordinates are `floor(coords / voxel_size)`, with
the global origin at zero. No origin shift or boundary epsilon is added. Thus
`-0.01` at size `0.1` quantizes to `-1`, not `0`. Values outside the int64 voxel
coordinate range fail.

Voxel coordinates are sorted lexicographically. A voxel's features are the
arithmetic mean of its retained points. Its label is the most frequent
nonignored training ID; a tie chooses the smallest training ID. An ignored-only
voxel keeps label `-1`. A prediction sample stays unlabeled.

Prepared data retain a `point_to_voxel` entry for every original source row, and
the complete original `point_indices` array. `point_to_voxel[i]` selects that
row's voxel; `-1` marks a removed nonfinite row. Always mask out `-1` before
indexing predictions, because NumPy would otherwise select the final voxel.
This mapping supports restoring outputs to source order and explicitly marking
invalid points when prediction is implemented.

## Prepared artifacts

Preprocessing creates a new directory with `prepared.json` and one numeric NPZ
file per scene under `scenes/`. It refuses an existing output directory and
writes the completion metadata last. Failed commands remove the output they
created. This is a local single-command contract; durable distributed run
coordination is a later milestone.

Prepared metadata is a distinct schema, not a raw dataset manifest. Use
`rptu inspect --manifest PATH` for raw data or `rptu inspect --prepared PATH` for
prepared metadata. The options are mutually exclusive. Both commands validate
every listed scene before printing a successful report.

`prepared.json` has `schema_version: 1`, `kind: "rptu-prepared-dataset"`, and:

- `source_manifest_sha256`: checksum of the original manifest bytes.
- `coordinate_unit`, `feature_names`, and `ignore_label`: the source contract.
- `label_mapping`: ordered entries with `dataset_id`, `train_id`, and `name`.
- `training_ignore_label: -1` and the numeric `voxel_size` in the declared unit.
- `quantization: "floor"`, `feature_reduction: "mean"`, and
  `label_reduction: "majority_nonignored_then_smallest_train_id"`.
- `scenes`: entries containing `id`, `split`, relative output `path`, output
  `sha256`, `source_sha256`, `input_points`, `valid_points`, `voxel_count`, and
  the boolean `labeled`.

Each prepared NPZ contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `voxel_coords` | `[M, 3]` | Signed int64 voxel coordinates in lexicographic order. |
| `features` | `[M, F]` | Float64 mean features in manifest order. |
| `labels` | `[M]` | Optional int64 voxel training IDs, with `-1` for ignore. |
| `point_labels` | `[N]` | Optional int64 training IDs for every original source row. |
| `point_indices` | `[N]` | Original int64 point IDs, including removed rows. |
| `point_to_voxel` | `[N]` | Int64 voxel lookup, with `-1` for removed rows. |

`N` is the original point count and `M` is the number of retained voxels.
`point_labels` preserve conflicting point-level ground truth that majority
aggregation would otherwise discard. They also retain targets for nonfinite
rows; a future evaluator must use `point_to_voxel >= 0` and
`point_labels != -1` when selecting valid evaluation targets. Both label arrays
are absent for an unlabeled prediction scene.

The seeded synthetic generator produces identical array and manifest bytes for
the same implementation, seed, and dependency versions. This is not a guarantee
across changes to the generator, NumPy, or the serialization implementation.

## Reading and validating prepared data

```sh
rptu inspect --prepared runs/demo-prepared/prepared.json --json
```

The report identifies `kind: "rptu-prepared-dataset"` and includes per-scene and
total input, retained, removed, and voxel counts. Point and voxel label count maps
use training IDs, including `-1`; point counts include removed rows. Raw inspection
uses `kind: "rptu-raw-dataset"` and its label counts use original dataset IDs.

The Python API is independent of any ML engine:

```python
from rptu_ai_toolkit.prepared import load_prepared_manifest, load_prepared_scene

dataset = load_prepared_manifest("runs/demo-prepared/prepared.json")
for scene in dataset.scenes:
    sample = load_prepared_scene(dataset, scene)
    retained = sample.point_to_voxel >= 0
    original_ids = sample.point_indices[retained]
```

Manifest loading validates the schema, reductions, class mapping, scene identities,
counts, and contained relative paths. Scene loading additionally verifies the file
checksum, exact signed int64/float64 dtypes (either byte order), shapes, finite
features, sorted unique voxel coordinates, unique point IDs, and mapping bounds.
Every voxel must be referenced and counts must agree. Labeled scenes require both
label arrays; labels must lie in the declared training range, and voxel labels
must match the documented majority/tie rule applied to retained point labels.

The prepared directory can be moved as one bundle. Validation does not reopen
raw files. Source hashes are recorded provenance, not proof of publisher identity
or of the transformation against absent raw data. In particular, the reader
cannot recompute voxel coordinates or mean features without the source arrays.
