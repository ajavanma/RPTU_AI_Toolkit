# CPU quickstart

Use Python 3.11 or newer. Python 3.11 and 3.12 are covered by the CPU test matrix.
Install from the source repository; these instructions do not require a PyPI release.

```sh
git clone https://github.com/ajavanma/RPTU_AI_Toolkit.git
cd RPTU_AI_Toolkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
rptu --help
rptu doctor
```

These activation commands are for a POSIX shell. The base package installs NumPy
and PyYAML. It does not install PyTorch, WarpConvNet, MinkowskiEngine, Open3D, or CUDA.

## Run a complete CPU data demonstration

From the repository root, with the virtual environment active:

```sh
rptu config validate --config examples/config.yaml
rptu synthetic --output runs/demo-data --seed 42
rptu inspect --manifest runs/demo-data/manifest.json
rptu preprocess --manifest runs/demo-data/manifest.json --output runs/demo-prepared --voxel-size 0.1
rptu inspect --prepared runs/demo-prepared/prepared.json
```

The synthetic dataset contains four small scenes assigned to train, validation,
test, and prediction splits. It includes nonfinite rows, ignored labels, class
collisions, and negative coordinates to exercise alignment and voxelization.
It demonstrates data correctness, not model accuracy.

`runs/demo-data/manifest.json` describes the raw scenes. `generation.json` records
the seed, generator revision, and package versions. Preprocessing writes
`runs/demo-prepared/prepared.json` and numeric scene archives, including mappings
back to every original point. Inspect raw data with `--manifest` and prepared
metadata with `--prepared`; these options are mutually exclusive. Prepared
inspection validates all scene files and their mappings before reporting success.
The prepared directory can be moved as one bundle and inspected without the raw
files. Its recorded source hashes do not prove provenance or recompute the
transformation against absent source data.

Both output directories must be new. Existing output is never overwritten;
choose different output paths when repeating the demonstration. Missing parent
directories are created automatically.

The dataset uses meters, so `--voxel-size 0.1` means a ten-centimeter voxel edge.
For other datasets, voxel size uses the manifest's declared coordinate unit;
there is no automatic unit conversion.

## Optional PLY input

From the repository checkout:

```sh
python -m pip install '.[io]'
```

The optional `plyfile` reader accepts ASCII and binary PLY vertices. See the
[dataset contract](dataset-contract.md) for required properties and labels.
PCD, ASC, and historical pickle inputs are not supported by the new workflow.

## Inspect machine-readable output

```sh
rptu doctor --json
rptu inspect --manifest runs/demo-data/manifest.json --json
rptu inspect --prepared runs/demo-prepared/prepared.json --json
```

Prepared JSON reports input, retained, removed, and voxel counts per scene and in
total. Point and voxel label counts use training IDs, including `-1`; point-label
counts include removed rows. Raw reports use original dataset class IDs.

Diagnostics do not initialize ML libraries. `gpu_runtime: "not_checked"` and
`ml_pipeline: false` accurately describe this milestone; installed package
metadata alone does not establish GPU readiness.

Continue with the [CLI reference](cli-reference.md) or bring your own scenes using
the [dataset contract](dataset-contract.md).
