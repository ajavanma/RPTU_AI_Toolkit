# Test and evaluation

## Evaluate a saved model

```sh
rptu evaluate --prepared runs/demo-prepared/prepared.json --model runs/demo-model --output runs/demo-evaluation --split test --json
```

Evaluation defaults to the `test` split. Select `val` for validation or `train` to
inspect training fit. Predictions are mapped from voxels back to original point
rows before scoring. Removed rows and ignored labels are excluded. Metrics include
a confusion matrix, per-class intersection-over-union, mean IoU, and accuracy.
Classes with no union receive `null` IoU and do not contribute to mean IoU.
Class ordering follows the declared label mapping.

The model bundle must match the dataset's ordered features, coordinate units,
voxel size, and class mapping. Scene files and model weights are checked before
use. Output is written into a new directory, with `evaluation.json` for later
analysis and per-scene prediction archives under `scenes/`. The report's `metrics`
object contains `confusion_matrix`, `classes` (including each class's `iou`),
`accuracy`, `mean_iou`, and counts of total, valid, scored, and excluded points.
Confusion rows are target training IDs and columns are predicted training IDs.
Unlabeled scenes contribute no scored points. Evaluation requires at least one
valid, nonignored labeled point in the selected split; otherwise it reports an
error and removes its incomplete output.

## Predict original dataset labels

```sh
rptu predict --prepared runs/demo-prepared/prepared.json --model runs/demo-model --output runs/demo-predictions --split predict --json
```

Prediction defaults to the `predict` split, which can be unlabeled. Each numeric
NPZ archive contains original point IDs, original row positions, predicted dataset
class IDs, and a validity mask, using the array names `point_indices`,
`point_rows`, `labels`, and `valid_mask`. Labeled inputs also include
`target_labels` in original dataset IDs, preserving targets even on removed rows.
Removed input rows retain their row and point ID,
receive the dataset's ignore label, and have a false validity flag.
`predictions.json` records the output files and checksums. No point is silently
dropped.

A saved PyTorch model can run on CPU or CUDA with `--device`. A saved WarpConvNet
model runs on CUDA and retains its spatial architecture. Model selection happens
during training; loading does not substitute an unrelated model.

## Software tests

See [development and tests](development.md) for core, CLI, and CPU model tests.
Synthetic metrics establish functional behavior; report real-data performance
against a held-out dataset with its preparation settings and model bundle.
