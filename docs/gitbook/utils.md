# Utilities

Use `rptu doctor` for dependency metadata and `rptu COMMAND --help` for command
arguments. Diagnostics and data commands remain usable without loading PyTorch
or WarpConvNet.

`rptu config validate --config PATH` checks application YAML. Dataset and model
commands take explicit arguments; see [configuration](configuration.md) and the
[CLI reference](cli-reference.md).

The Python data API exposes `load_prepared_manifest` and `load_prepared_scene`
from `rptu_ai_toolkit.prepared`. They return validated metadata and numeric arrays,
independent of the model backend. See the [dataset contract](dataset-contract.md).

The model API in `rptu_ai_toolkit.ml` exposes `train_model`, `evaluate_model`, and
`predict_model`, with the same prepared inputs and new-output-directory rules as
the CLI. ML dependencies load only when a model command executes.

Use the repository's
[container guide](https://github.com/ajavanma/RPTU_AI_Toolkit/blob/main/docs/containers.md)
for CPU and NVIDIA execution through Docker. Input datasets are mounted read-only;
outputs and runtime caches use explicit writable mounts.
