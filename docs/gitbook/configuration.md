# Configuration

Application configuration uses YAML schema version `1`. It currently covers only
paths and runtime defaults. Dataset commands take explicit CLI arguments and a
separate JSON [dataset manifest](dataset-contract.md); they do not read this YAML.

The repository's `examples/config.yaml` contains:

```yaml
schema_version: 1
paths:
  input_dir: ../data
  output_dir: ../runs
runtime:
  seed: 0
  num_workers: 0
```

Validate it from the repository root:

```sh
rptu config validate --config examples/config.yaml
rptu config validate --config examples/config.yaml --json
```

Relative paths resolve against the configuration file's directory, regardless of
the working directory. Here `../data` resolves to the repository's `data` directory.
Validation shows resolved paths but does not require those directories to exist
or create them.

## Rules

- `schema_version` must be integer `1`.
- `paths.input_dir` and `paths.output_dir` are required nonempty strings. Their
  resolved directories must differ.
- Absolute paths stay absolute and `~` expands to the home directory. Environment
  variable text such as `$DATA_DIR` stays literal; it is not expanded.
- `runtime` and either field may be omitted. `seed` and `num_workers` then default
  to `0`.
- `seed` must be an integer from `0` through `4294967295`.
- `num_workers` must be a nonnegative integer.
- Unknown fields, duplicate YAML keys, and booleans used as integers are rejected.
  YAML merge overrides that duplicate a field are also rejected.

Configuration validation does not initialize a model or check GPU availability.
Model settings will be defined alongside the model implementation.

## Historical configuration

The flat `config/config.yaml` belongs to the older research scripts and is not
accepted by the new CLI. Start with `examples/config.yaml`; renaming the old file
is not a migration. The historical `requirements.txt` is also not the current
package installation specification. Install through `pyproject.toml` using pip
or the committed uv lockfile, as described in [development](development.md).
