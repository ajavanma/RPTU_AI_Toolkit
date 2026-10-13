# Test and evaluation

The current toolkit has CPU correctness tests and dataset validation. Model
evaluation is planned; `rptu evaluate` returns an explicit not-implemented error
with exit status `2`.

To check raw data and its prepared output:

```sh
rptu inspect --manifest runs/demo-data/manifest.json
rptu inspect --prepared runs/demo-prepared/prepared.json
```

Generate and preprocess that demonstration dataset using the [quickstart](quickstart.md).
Prepared inspection checks every scene's checksum, dtypes, shapes, mapping counts,
voxel coverage, and label aggregation. It validates bundle consistency without
reopening the raw source files. To run the package's unit and CLI tests, follow
[development](development.md).

The dataset contract includes a `test` split, and prepared artifacts preserve
original point labels and point-to-voxel correspondence for future point-level
evaluation. A split name or preserved labels do not imply that a model evaluator
already exists.

Earlier research documentation listed accuracy, precision, recall, F1, and
confusion-matrix calculations from the research code. Those are not supported
commands in the current package. Synthetic scenes test data correctness and must
not be presented as evidence of real-world segmentation quality.
