# CLI reference

The installed command is `rptu`. `python -m rptu_ai_toolkit` provides the same
interface. In a source checkout, `python src/main.py` delegates to this CLI;
without arguments it prints help instead of opening an interactive menu.

| Command | Behavior |
| --- | --- |
| `rptu --help` | Show commands and availability. |
| `rptu --version` | Show the installed toolkit version. |
| `rptu doctor [--json]` | Report Python, platform, installed package metadata, and capabilities. |
| `rptu config validate --config PATH [--json]` | Validate application YAML and show resolved configuration. |
| `rptu synthetic --output DIR [--seed N] [--json]` | Generate four small seeded scenes and their manifest. |
| `rptu inspect --manifest PATH [--json]` | Validate the raw dataset manifest, checksums, scene arrays, and labels. |
| `rptu inspect --prepared PATH [--json]` | Validate prepared metadata, every scene archive, and point/voxel consistency. |
| `rptu preprocess --manifest PATH --output DIR --voxel-size SIZE [--json]` | Filter invalid rows and voxelize the dataset on the CPU. |
| `rptu train`, `rptu evaluate`, `rptu predict`, `rptu benchmark` | Reserved commands; report that they are not implemented. |

Run `rptu COMMAND --help` for a command's arguments.

## Diagnostics and configuration

`doctor` reads distribution metadata without importing ML libraries. It reports
configuration validation, CPU dataset preparation, and prepared-dataset validation
as available, `ml_pipeline` as `false`, and `gpu_runtime` as `"not_checked"`. Finding a PyTorch or WarpConvNet
installation does not validate CUDA execution.

`config validate` checks structure and path rules without requiring a dataset,
creating directories, or initializing a GPU. Dataset commands use explicit CLI
arguments and a separate JSON manifest; they currently do not read application
YAML. See [configuration](configuration.md).

## Dataset commands

`synthetic` defaults to seed `0`. A seed must be an integer between `0` and
`4294967295`. Its output directory must not already exist.

`inspect --manifest` checks every raw scene's SHA-256 before parsing it and reports point counts,
nonfinite rows, and class statistics. A manifest can contain unusable rows when a
scene also contains usable points; preprocessing removes those rows consistently
from aligned data. Inspection never silently repairs unknown class IDs or
malformed arrays.

`inspect --prepared` validates a complete prepared bundle without reopening raw
files. It checks metadata, output checksums, exact array dtypes and shapes, sorted
voxel coordinates, point IDs, mapping bounds and coverage, counts, label ranges,
and majority/tie label aggregation. The raw `--manifest` and prepared `--prepared`
options are mutually exclusive. Neither command prints a successful report until
every listed scene has passed validation.

Prepared reports use `kind: "rptu-prepared-dataset"` and include per-scene and total
input, retained, removed, and voxel counts. Point and voxel label count maps use
training IDs; point-label counts include removed rows. Raw reports use
`kind: "rptu-raw-dataset"`, with label counts in original dataset IDs. Recorded
source hashes identify the claimed source bytes; prepared inspection does not
verify publisher identity or recompute voxel coordinates and feature means from
absent raw arrays.

`preprocess` requires a finite positive voxel size in the manifest's coordinate
unit and a new output directory. It preserves original point correspondence and
writes completion metadata last. On failure, it removes the output directory it
created. It does not train a model or reconstruct predictions.

The [dataset contract](dataset-contract.md) defines split rules, class mappings,
feature reductions, numeric limits, and the prepared output format. For a prepared
bundle, pass its `prepared.json` to `inspect --prepared`.

## Exit status and JSON

Successful commands and help return status `0`. Argument errors, configuration or
dataset errors, and unimplemented commands return status `2`, with errors on
stderr. A command's `--json` option emits JSON on stdout on success.

A successful `doctor` means diagnostics completed. A successful CPU preprocessing
run establishes a data artifact, not model quality or GPU support.
