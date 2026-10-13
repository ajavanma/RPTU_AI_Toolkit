# Tests

Use Python 3.11 or newer, and install the package before running its tests:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests/unit -v
python -m unittest discover -s tests/cli -v
```

`tests/unit` covers the new configuration schema and the existing configuration,
file-matching, preprocessing-result, and multiprocessing helpers. These tests do
not require Open3D, PyTorch, WarpConvNet, a GPU, or the original point-cloud data.

`tests/cli` runs the installed `rptu` command and module entry point from temporary
working directories outside the checkout. It checks version/help, diagnostics,
configuration validation, path resolution, and useful errors. An import guard
fails if help or diagnostics tries to import ML libraries. Tests also check the
legacy `src/main.py` wrapper.

CI builds the package and repeats the CLI suite with the wheel installed in a
fresh environment. This catches packaging errors that an editable install can hide.

The older tests directly under `tests/` reference incomplete research interfaces
and external data. They are not part of the supported suite; `tests/test_test.py`
is a placeholder. Run the two discovery commands above rather than broad test
collection. Passing the CPU suite does not validate model training, inference,
CUDA, or Kubernetes execution.
