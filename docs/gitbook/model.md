# Model

The toolkit provides two concrete backends through the same train, evaluate, and
predict commands.

| Backend | Architecture | Device | Role |
| --- | --- | --- | --- |
| `warpconvnet` | Native WarpConvNet MinkUNet18 | CUDA | Spatial point-cloud segmentation, with NVIDIA RTX 4070 as the reference GPU |
| `torch` | Voxel MLP, two hidden layers of 64 units with ReLU | CPU or CUDA | Portable fallback and a lightweight baseline |

WarpConvNet receives sparse voxel coordinates and normalized features. Its output
is aligned back to the prepared voxel order before labels are reconstructed at
original point resolution. BatchNorm running statistics are frozen for small
scenes; affine parameters remain trainable. The adapter checks coordinate bounds
before constructing the sparse tensor.

The PyTorch fallback classifies each voxel's feature vector independently. It does
not reproduce the spatial receptive field of MinkUNet18. Both models use training-
split feature normalization and the same reversible class mapping.

## Runtime selection

`train --device auto --backend auto` selects WarpConvNet with CUDA when available,
otherwise PyTorch on CPU. `--backend torch` can use either CPU or CUDA.
`--backend warpconvnet --device cuda` requires the GPU profile. Explicit device or
backend choices are honored; kernel or out-of-memory failures are reported rather
than switching models partway through a run.

Evaluation and prediction read the saved backend. PyTorch bundles can move between
CPU and CUDA. WarpConvNet bundles require CUDA; their weights cannot be interpreted
as a voxel MLP.

## Model bundles

A bundle contains fixed-name `bundle.json` and `weights.safetensors` artifacts.
Metadata stores the schema version, architecture, feature normalization, ordered
features, label mapping, voxel size, units, training settings, runtime versions,
and weight checksum. Loading checks the contract and exact parameter structure.
Keep both files together when moving a model.

The GPU installation profile pairs WarpConvNet 1.8.2 with PyTorch 2.10.0 and CUDA
12.8 on Linux x86_64/Python 3.11. See the [quickstart](quickstart.md) for installation
and the [training guide](train.md) for producing a new bundle.

The project uses fresh training runs. Original research weights are not included,
and conversion of historical MinkowskiEngine checkpoints is not provided.
