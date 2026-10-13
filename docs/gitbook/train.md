# Train

Train on a prepared dataset with an explicit split assignment and label mapping:

```sh
rptu train --prepared runs/demo-prepared/prepared.json --output runs/model --backend warpconvnet --device cuda --epochs 20 --learning-rate 0.001 --seed 42
```

For the portable CPU model, use `--backend torch --device cpu`. Leaving both
options at `auto` chooses WarpConvNet when its CUDA runtime is available, otherwise
the CPU fallback. Explicit requests fail with an actionable error when unavailable;
an error during a GPU run does not silently start a different model.

## Resume an interrupted run

Use the same prepared data and run directory:

```sh
rptu train --prepared runs/demo-prepared/prepared.json --output runs/model --resume
```

Training saves an initial checkpoint before optimization and another after every
completed epoch. Resume restores model weights, Adam optimizer state, Torch random
state, and loss history, then continues to the original total epoch count. An
interrupted epoch is repeated from its preceding checkpoint. Checkpoints contain
numeric safetensors and versioned JSON; they do not deserialize Python objects.

Omitted settings come from the saved run specification. Explicit settings must
match: `--epochs 20 --resume` means finish the original 20 epochs, not add 20 more.
The prepared manifest checksum, feature and label contracts, normalization,
architecture, runtime versions, device, and CPU thread count must remain
compatible. Use a new output directory when changing data or settings.

If the run already completed, `--resume` verifies the completion record, saved
artifacts, and final checkpoint, then returns the original report. It does not
train again or create another attempt. Without `--resume`, an existing output
is always rejected.

## Training contract

- The training split supplies optimization samples and feature-normalization
  statistics. Validation scenes never contribute to the normalization fit.
- Training processes one scene at a time using Adam and masked cross-entropy.
  Ignored voxel labels (`-1`) do not contribute to loss.
- Feature order, units, voxel size, class mapping, and preprocessing choices are
  recorded with the model. Evaluation rejects incompatible datasets.
- New runs default to 20 epochs, learning rate 0.001, and seed 0.
- `--max-voxels` defaults to `100000` per scene. Larger inputs fail clearly instead
  of losing points silently. This bound is not an automatic memory estimator.
- Loss, gradients, and checkpoint tensors must remain finite. Completed
  checkpoints and attempt records survive a training failure.

WarpConvNet uses MinkUNet18 with fixed BatchNorm running statistics and trainable
affine parameters, allowing small sparse scenes. The CPU fallback is a two-hidden-
layer voxel MLP. See the [model guide](model.md) for their different roles.

## Run directory and completion

The successful output retains the existing inference files `bundle.json`,
`weights.safetensors`, and `training.json`. The report records each epoch's loss
and point-level `train` and `val` metrics; `val` is `null` when no validation split
is present. Use this same directory with `evaluate --model` or `predict --model`.

The directory layout is:

```text
model/
  run.json                         # Immutable run identity and settings
  .run.lock                        # Kernel-held writer lock
  attempts/<attempt-id>/attempt.json
  checkpoints/epoch-<number>-<id>/checkpoint.json
  checkpoints/epoch-<number>-<id>/state.safetensors
  latest.json                      # Pointer to the latest complete checkpoint
  weights.safetensors              # Final inference weights
  training.json                    # Final training report
  bundle.json                      # Final inference contract
  complete.json                    # Completion and artifact checksums
```

The run keeps its immutable specification, epoch checkpoints, latest checkpoint
pointer, attempt records, and a checksummed completion record. Each
attempt has its own identity. A failed attempt leaves the last complete checkpoint
available. The latest pointer advances only after the new checkpoint is fully
written. Final model artifacts are staged before publication, and completion is
recorded last. Resume can finish an interrupted publication. Model loading rejects
an incomplete training run.

A process holds an exclusive lock while writing a run. A second writer fails
instead of competing for the directory; the operating system releases the lock
when the first process exits or is killed. Do not delete lock files manually.
Use a local POSIX filesystem, such as a Linux or macOS local disk, including local
container bind mounts or volumes. This mechanism does not coordinate independent
copies of a run, object storage, or shared network filesystems.

Keep the complete run directory to retain recovery information. For inference
alone, the versioned model bundle remains independent of optimizer and checkpoint
state. Step-level resume, changing settings during resume, and distributed run
coordination are future extensions.

The [quickstart](quickstart.md) creates public synthetic inputs. Their purpose is
to exercise the complete workflow; useful application models require representative
training data and a separate evaluation split.
