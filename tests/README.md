# Tests

Run the lightweight unit suite from the repository root:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests/unit -v
```

These tests run without Open3D, PyTorch, MinkowskiEngine, a GPU, or the original
point-cloud dataset. CI runs this suite on pull requests and pushes to `main`.

The older tests directly under `tests/` require the training environment and
external data, and some still reference missing modules or outdated interfaces.
They are not part of this unit suite. `tests/test_test.py` is a placeholder and
does not validate application behavior.
