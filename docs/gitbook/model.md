# Model

No supported model is included in the modernized CLI yet. WarpConvNet is the
planned sparse-convolution backend for a fresh point-cloud semantic-segmentation
pipeline. GPU dependencies and execution still need validation on a supported
GPU host.

The CPU package currently establishes input contracts, label mappings, voxelized
features, and correspondence back to original points. See the
[dataset contract](dataset-contract.md) for these interfaces. The future model
must preserve their class ordering and ignore-label semantics.

Training, evaluation, prediction, checkpoint resume, and benchmarking remain
future work. No pretrained weights, original datasets, or research performance
results are supplied. The new pipeline must establish and report its own baseline.

Earlier research documentation specified a MinkowskiEngine/PyTorch Lightning
architecture with global pooling and fully connected layers. It does not specify
a supported current model and must not be used as an installation or performance
claim for the new toolkit.

The base installation does not install PyTorch or WarpConvNet. `rptu doctor`
reports `gpu_runtime: "not_checked"`; distribution metadata alone cannot confirm
CUDA readiness. Follow the [roadmap](todo.md) for the planned sequence.
