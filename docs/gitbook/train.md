# Train

Train on a prepared dataset with an explicit split assignment and label mapping:

```sh
rptu train --prepared runs/demo-prepared/prepared.json --output runs/model --backend warpconvnet --device cuda --epochs 20 --learning-rate 0.001 --seed 42
```

For the portable CPU model, use `--backend torch --device cpu`. Leaving both
options at `auto` chooses WarpConvNet when its CUDA runtime is available, otherwise
the CPU fallback. Explicit requests fail with an actionable error when unavailable;
an error during a GPU run does not silently start a different model.

## Training contract

- The training split supplies optimization samples and feature-normalization
  statistics. Validation scenes never contribute to the normalization fit.
- Training processes one scene at a time using Adam and masked cross-entropy.
  Ignored voxel labels (`-1`) do not contribute to loss.
- Feature order, units, voxel size, class mapping, and preprocessing choices are
  recorded with the model. Evaluation rejects incompatible datasets.
- `--epochs` defaults to `20`, `--learning-rate` to `0.001`, and `--seed` to `0`.
- `--max-voxels` defaults to `100000` per scene. Larger inputs fail clearly instead
  of losing points silently. This bound is not an automatic memory estimator.
- Loss and gradients must remain finite. Model artifacts are written only after
  successful training, and existing output directories are preserved.

WarpConvNet uses MinkUNet18 with fixed BatchNorm running statistics and trainable
affine parameters, allowing small sparse scenes. The CPU fallback is a two-hidden-
layer voxel MLP. See the [model guide](model.md) for their different roles.

## Model output

The output directory contains `bundle.json`, `weights.safetensors`, and
`training.json`. The report records each epoch's loss and point-level `train` and
`val` metrics; `val` is `null` when no validation split is present.
The bundle records the architecture, preprocessing contract, normalization,
weight checksum, and runtime information. It is portable as a whole directory.
Weights use the safetensors format; model loading does not deserialize Python
objects.

Use this directory with `evaluate --model` or `predict --model`. Checkpoint resume
and optimizer-state recovery are follow-up work. Train a new run into a new output
directory when changing settings.

The [quickstart](quickstart.md) creates public synthetic inputs. Their purpose is
to exercise the complete workflow; useful application models require representative
training data and a separate evaluation split.
