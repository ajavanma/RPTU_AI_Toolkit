# RPTU AI Toolkit

[![CPU package tests](https://github.com/ajavanma/RPTU_AI_Toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/ajavanma/RPTU_AI_Toolkit/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RPTU AI Toolkit is a GPU-first point-cloud segmentation toolkit for NVIDIA RTX 4070 systems, with a portable CPU fallback. It combines synthetic data generation, dataset validation, voxel preprocessing, training, evaluation, and prediction. The project originated in LiDAR segmentation and BIM research and provides reusable public dataset contracts and reproducible workflows.

WarpConvNet MinkUNet18 provides the GPU model. A PyTorch voxel MLP provides
portable training and inference on CPU or CUDA. Both share the CLI, dataset
contracts, and original point mapping. Saved models retain their architecture;
WarpConvNet bundles require CUDA, while PyTorch bundles can move between devices.

## Install the GPU or CPU runtime

For NVIDIA RTX 4070 systems, use Linux x86_64 and Python 3.11 with an NVIDIA driver
compatible with CUDA 12.8:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/gpu-cu128.txt
python -m pip install '.[ml]'
```

For the CPU fallback, install `.[ml]`. On Linux, select the CPU PyTorch wheel first:

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[ml]'
```

On macOS, `python -m pip install '.[ml]'` is sufficient. The base installation
below remains useful for data preparation without any ML dependencies.

## Install and run the CPU demo

Use Python 3.11 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
rptu --help
rptu doctor
rptu config validate --config examples/config.yaml
rptu synthetic --output runs/demo-data --seed 42
rptu inspect --manifest runs/demo-data/manifest.json
rptu preprocess --manifest runs/demo-data/manifest.json --output runs/demo-prepared --voxel-size 0.1
rptu inspect --prepared runs/demo-prepared/prepared.json
```

The package depends on PyYAML and NumPy. Installation does not install PyTorch,
WarpConvNet, MinkowskiEngine, Open3D, or CUDA. Python 3.11 and 3.12 are covered by
the CPU CI matrix.

The demo has four small scenes assigned to train, validation, test, and prediction
splits. It deliberately includes invalid rows, ignored labels, class collisions,
and negative coordinates. It demonstrates data correctness, not real-world model
quality. Existing output directories are never overwritten; choose a fresh output
path for another run. Missing parent directories are created automatically.
`generation.json` records the seed, generator revision, and package versions.

NPZ input works with the base installation. To read ASCII or binary PLY vertices,
install the optional reader with `python -m pip install '.[io]'` from the repository.
PCD/ASC adapters remain future work.

The module entry point is equivalent to the installed command:

```sh
python -m rptu_ai_toolkit doctor --json
```

`python src/main.py` now delegates to the new CLI. Running it without arguments
prints help and exits; it no longer opens an interactive menu or imports PyTorch.

## Train, evaluate, and predict

With the `ml` dependencies installed and the demo prepared:

```sh
rptu train --prepared runs/demo-prepared/prepared.json --output runs/demo-model --device auto --epochs 20
rptu evaluate --prepared runs/demo-prepared/prepared.json --model runs/demo-model --output runs/demo-evaluation
rptu predict --prepared runs/demo-prepared/prepared.json --model runs/demo-model --output runs/demo-predictions
```

Automatic training uses WarpConvNet on CUDA when available, otherwise the CPU
fallback. Select explicitly with `--backend warpconvnet --device cuda` or
`--backend torch --device cpu`. Evaluation and prediction use the saved model's
architecture. The fallback is a per-voxel feature classifier, with a different
capacity from the spatial MinkUNet18 model.

Training writes a model bundle with safetensors weights, feature normalization,
class mapping, and checksums. Evaluation reports original-point metrics;
prediction exports original class IDs, point IDs, row positions, and validity
masks. Every output path must be new. The default bound is 100,000 voxels per
scene; `--max-voxels` changes it without silently truncating inputs.

See [training](docs/gitbook/train.md), [models](docs/gitbook/model.md), and
[container execution](docs/containers.md) for full instructions.

## Commands

| Command | Behavior |
| --- | --- |
| `rptu --help` | Show commands and their availability |
| `rptu --version` | Show the package version |
| `rptu doctor [--json]` | Report Python, platform, package metadata, and supported capabilities |
| `rptu config validate --config PATH [--json]` | Validate YAML and show the resolved configuration |
| `rptu synthetic --output DIR [--seed N] [--json]` | Generate a reproducible four-scene dataset |
| `rptu inspect --manifest PATH [--json]` | Validate the manifest, file checksums, arrays, and labels; report point counts |
| `rptu inspect --prepared PATH [--json]` | Validate every prepared scene, mapping, and label target; report point and voxel counts |
| `rptu preprocess --manifest PATH --output DIR --voxel-size SIZE [--json]` | Filter invalid rows and voxelize every scene on the CPU |
| `rptu train --prepared PATH --output DIR` | Train a model using the selected backend and device |
| `rptu evaluate --prepared PATH --model DIR --output DIR` | Evaluate a saved model at original point resolution |
| `rptu predict --prepared PATH --model DIR --output DIR` | Export original dataset labels and point IDs |
| `rptu benchmark` | Reserved for the benchmark workflow |

Diagnostics inspect installed package metadata without importing ML libraries.
Runtime and device selection happen when a model command starts.

Successful commands and help exit with status `0`. Invalid arguments,
configuration errors, and unimplemented commands exit with status `2`; error
messages go to stderr. `--json` emits JSON on stdout on success.

## Dataset contracts

Datasets use a versioned JSON manifest with scene IDs and splits, relative file
paths, SHA-256 checksums, coordinate units, ordered feature names, and explicit
class IDs. No specific university dataset or directory layout is assumed.
Each NPZ scene contains `coords[N,3]`, `features[N,F]`, optional `labels[N]`, and
optional original `point_indices[N]`. Labels are required for train/val/test and
optional for prediction. Arrays are numeric and are loaded without pickle.

Classes map to contiguous training IDs in their declared order. Class zero is
valid, the ignore label stays separate, and unknown IDs are errors. Manifest
validation rejects duplicate scene IDs and paths or identical file checksums shared
across splits. Assign each real scene to one split before producing chunks; these
checks cannot identify the same scene re-exported as different file bytes.

Preprocessing uses one finite-row mask for coordinates, features, and labels.
It computes `floor(coords / voxel_size)` in the declared coordinate unit, averages
voxel features, and chooses the most frequent nonignored label (smallest training
ID for a tie). A full-length point-to-voxel mapping marks removed rows with `-1`.
Prepared NPZ files also retain original point IDs and labels for point-level
evaluation. `prepared.json` records the schema, reductions, mappings, and checksums.

See [the dataset and prepared-artifact contract](docs/datasets.md) for the complete
schema, PLY property mapping, numeric limits, and reconstruction rules. The raw
dataset manifest and prepared metadata are distinct formats: select exactly one
of `inspect --manifest` and `inspect --prepared`. Prepared validation works after
moving the bundle and without access to the original dataset. It verifies the
saved artifacts' internal consistency; provenance hashes do not authenticate their
publisher or prove the transformation against unavailable raw data.

## Configuration

The application YAML schema is limited to paths and runtime settings. Dataset and ML
commands take explicit command-line arguments and their separate JSON
manifest; they do not read this YAML. Training options are recorded in the saved model bundle.

```yaml
schema_version: 1
paths:
  input_dir: ../data
  output_dir: ../runs
runtime:
  seed: 0
  num_workers: 0
```

See [examples/config.yaml](examples/config.yaml). In that file, `../data` resolves
to the repository's `data` directory, regardless of the command's working directory.

- `schema_version` must be integer `1`.
- `paths.input_dir` and `paths.output_dir` are required nonempty strings. Relative
  paths resolve against the configuration file's directory. Absolute paths remain
  absolute; `~` is expanded. Environment variables such as `$DATA_DIR` are literal
  text and are not expanded.
- Input and output must resolve to different directories.
- `runtime` and either of its fields can be omitted. `seed` defaults to `0` and
  must be an integer from `0` through `4294967295`. `num_workers` defaults to `0`
  and must be a nonnegative integer.
- Unknown fields, duplicate YAML keys, and booleans used as integers are rejected.
  YAML merge overrides that duplicate a field are also rejected.

Validation checks configuration structure. It does not require datasets to exist,
create directories, or initialize GPU libraries. Use JSON output to inspect
resolved paths and defaults:

```sh
rptu config validate --config examples/config.yaml --json
```

The old flat [config/config.yaml](config/config.yaml) belongs to the historical
scripts and is not accepted by the new CLI. Start with the new example rather
than renaming the old file. Historical `requirements.txt` is not the package's
installation specification; use `pyproject.toml` through pip or uv.

## Development and tests

For an editable install and the existing helper tests:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests/unit -v
python -m unittest discover -s tests/cli -v
```

Alternatively, use the committed uv lockfile:

```sh
uv sync --locked --extra test --extra io
uv run --locked --extra test --extra io python -m unittest discover -s tests/unit -v
uv run --locked --extra test --extra io python -m unittest discover -s tests/cli -v
```

The lockfile includes the optional ML dependencies. Install `.[ml]` and run
`python -m unittest discover -s tests/ml -v` for CPU model tests. The separate
`requirements/gpu-cu128.txt` profile pins the WarpConvNet CUDA wheel and matching
PyTorch build.

CI runs the CPU unit and CLI tests, builds a source distribution and a wheel,
and exercises the wheel in a separate environment from outside the checkout.
Help, diagnostics, and data-command tests prevent accidental ML imports.
Separate CI jobs run CPU model tests and the complete CPU container workflow. See [tests/README.md](tests/README.md)
for the distinction between supported tests and historical research tests.

To build locally:

```sh
python -m pip install 'build>=1.2,<2'
python -m build
```

## Modernization direction

1. Installable package, noninteractive CLI, and validated configuration — implemented.
2. General dataset contracts, synthetic fixtures, and point/label alignment checks — implemented.
3. WarpConvNet and CPU models with training, evaluation, and prediction — implemented.
4. CPU/GPU containers — implemented; checkpoint resume and durable retries are next.
5. Optional Helm Jobs, followed by Fleet deployment examples and measured GPU benchmarks.

The new pipeline will establish its own data and model baselines. BIM export and
multi-GPU training are future work. The [current documentation](https://rptu.gitbook.io/rptu-ai-toolkit/)
covers the package, dataset, and model workflow. It is a manually published
snapshot; see [publication details](docs/gitbook-publication.md). The
[original research documentation](https://arash.gitbook.io/rptu-ai-toolkit/) remains
historical and does not describe the current package installation or CLI.

## License

MIT. See [LICENSE](LICENSE).
