# RPTU AI Toolkit

[![CPU package tests](https://github.com/ajavanma/RPTU_AI_Toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/ajavanma/RPTU_AI_Toolkit/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RPTU AI Toolkit is being modernized into a reproducible point-cloud semantic
segmentation toolkit using WarpConvNet. It originated in research on LiDAR
segmentation and BIM workflows.

**This development version provides the installable package, environment
diagnostics, and validated configuration.** Dataset processing, training,
evaluation, prediction, and benchmarking are not implemented in the new CLI yet.
The repository's older research scripts are incomplete and are not a supported
end-to-end workflow. No original datasets or trained model artifacts are included.

## Install the CPU foundation

Use Python 3.11 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
rptu --help
rptu doctor
rptu config validate --config examples/config.yaml
```

The package depends only on PyYAML. Installation does not install PyTorch,
WarpConvNet, MinkowskiEngine, Open3D, or CUDA. Python 3.11 and 3.12 are covered by
the CPU CI matrix. GPU runtime support will be established in a separate milestone.

The module entry point is equivalent to the installed command:

```sh
python -m rptu_ai_toolkit doctor --json
```

`python src/main.py` now delegates to the new CLI. Running it without arguments
prints help and exits; it no longer opens an interactive menu or imports PyTorch.

## Commands

| Command | Behavior |
| --- | --- |
| `rptu --help` | Show commands and their availability |
| `rptu --version` | Show the package version |
| `rptu doctor [--json]` | Report Python, platform, package metadata, and supported capabilities |
| `rptu config validate --config PATH [--json]` | Validate YAML and show the resolved configuration |
| `rptu inspect`, `preprocess`, `train`, `evaluate`, `predict`, `benchmark` | Reserved for later milestones; return an explicit not-implemented error |

Diagnostics inspect installed package metadata without importing ML libraries.
An installed WarpConvNet or PyTorch distribution does not establish that CUDA
works. `doctor` reports `gpu_runtime: "not_checked"` and `ml_pipeline: false`.
Its successful exit means diagnostics completed, not that GPU execution passed.

Successful commands and help exit with status `0`. Invalid arguments,
configuration errors, and unimplemented commands exit with status `2`; error
messages go to stderr. `--json` emits JSON on stdout on success.

## Configuration

The new schema is deliberately limited to application paths and runtime settings.
Model and dataset schemas will be added with their implementations.

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
uv sync --locked --extra test
uv run --locked --extra test python -m unittest discover -s tests/unit -v
uv run --locked --extra test python -m unittest discover -s tests/cli -v
```

The lockfile covers the CPU package and development environment. A GPU dependency
profile will be added after validating a specific WarpConvNet/PyTorch/CUDA combination.

CI runs the CPU unit and CLI tests, builds a source distribution and a wheel,
and exercises the wheel in a separate environment from outside the checkout.
The CLI tests prevent imports of ML dependencies. See [tests/README.md](tests/README.md)
for the distinction between supported tests and historical research tests.

To build locally:

```sh
python -m pip install 'build>=1.2,<2'
python -m build
```

## Modernization direction

1. Installable package, noninteractive CLI, and validated configuration — this change.
2. General dataset contracts, synthetic fixtures, and point/label alignment checks.
3. A fresh WarpConvNet model with training, evaluation, checkpoint resume, and prediction.
4. Reproducible containers and durable run outputs.
5. Optional Helm Jobs, followed by Fleet deployment examples and measured GPU benchmarks.

The new pipeline will establish its own data and model baselines. BIM export and
multi-GPU training are future work. The [original research documentation](https://arash.gitbook.io/rptu-ai-toolkit/)
is historical and does not describe the current package installation or CLI.

## License

MIT. See [LICENSE](LICENSE).
