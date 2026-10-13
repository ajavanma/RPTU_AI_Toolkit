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
and external data. They are outside the supported suite; use the two discovery
commands above instead of broad test collection.

## Locked development environment

If uv is installed:

```sh
uv sync --locked --extra test --extra io
uv run --locked --extra test --extra io python -m unittest discover -s tests/unit -v
uv run --locked --extra test --extra io python -m unittest discover -s tests/cli -v
```

The lockfile covers the CPU package and development environment. No validated GPU
dependency profile is provided yet.

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

Passing these CPU tests does not validate model training, inference, CUDA, or
Kubernetes execution. GPU, container, and deployment checks belong to their
respective future milestones.

Report defects through [GitHub issues](https://github.com/ajavanma/RPTU_AI_Toolkit/issues),
including the command, toolkit version, Python version, and a small reproducible
input where possible.
