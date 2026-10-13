# Development and tests

Clone the [source repository](https://github.com/ajavanma/RPTU_AI_Toolkit) and create
a Python 3.11 or newer virtual environment using the [quickstart](quickstart.md).
Python 3.11 and 3.12 are covered by the CPU test matrix.

## Install and test

From the repository root:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests/unit -v
python -m unittest discover -s tests/cli -v
```

The test installation includes the optional PLY reader. Unit tests cover strict
configuration and dataset contracts, NPZ/PLY input, class mappings, aligned finite
row filtering, voxel aggregation, deterministic synthetic scenes, and preservation
or cleanup of outputs. Prepared-reader tests check strict metadata, exact array
dtypes and shapes, counts, voxel ordering, point mappings, and label aggregation.
They also exercise corruption, bundle relocation, and labeled/unlabeled scenes.
Existing configuration, file matching, and preprocessing helper regressions remain
covered.

CLI tests run the installed command and module entry point from temporary working
directories outside the checkout. They exercise the generate, inspect, and
preprocess sequence followed by prepared-bundle inspection, checksums, original
point alignment, useful failures, and reproducibility. Prepared inspection is also
tested after moving the bundle and removing its raw source directory. Import
guards catch accidental ML dependency imports.

The older tests directly under `tests/` reference incomplete research interfaces
and external data. They are outside the supported suite; use the explicit discovery
commands here instead of broad test collection.

## Locked development environment

If uv is installed:

```sh
uv sync --locked --extra test --extra io
uv run --locked --extra test --extra io python -m unittest discover -s tests/unit -v
uv run --locked --extra test --extra io python -m unittest discover -s tests/cli -v
```

The lockfile includes the optional ML dependencies. To exercise the CPU model:

```sh
uv sync --locked --extra test --extra io --extra ml
uv run --locked --extra test --extra io --extra ml python -m unittest discover -s tests/ml -v
```

Alternatively install `.[ml]` with pip and run the same `tests/ml` discovery.
The separate `requirements/gpu-cu128.txt` pins the WarpConvNet CUDA wheel and
matching PyTorch build. Keep it separate from the portable CPU dependency set.

## Build and CI

```sh
python -m pip install 'build>=1.2,<2'
python -m build
```

The CPU CI workflow runs the unit and CLI suites, builds a source distribution
and wheel, and repeats the CLI tests with the wheel installed in a separate
environment outside the checkout. The clean wheel check installs core dependencies
only, keeping PLY optional. Current run results are available in
[GitHub Actions](https://github.com/ajavanma/RPTU_AI_Toolkit/actions/workflows/tests.yml).

The CPU model suite runs real training, safetensors save/load, point-level
metrics, and prediction. It exercises corruption and contract-mismatch failures,
ignored and removed rows, deterministic behavior, and output preservation.
A separate CI job builds the CPU container and runs the CLI through mounted
storage. GPU kernels and cluster execution are separate execution environments.

Report defects through [GitHub issues](https://github.com/ajavanma/RPTU_AI_Toolkit/issues),
including the command, toolkit version, Python version, and a small reproducible
input where possible.
