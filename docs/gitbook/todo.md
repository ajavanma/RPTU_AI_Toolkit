# Roadmap

## Available workflow

- Installable package and CLI with optional ML and PLY dependencies.
- Versioned dataset contracts, reversible class mappings, and preserved point IDs.
- Seeded synthetic scenes, CPU voxel preprocessing, and prepared-bundle inspection.
- WarpConvNet MinkUNet18 for NVIDIA GPUs and a portable PyTorch CPU fallback.
- Training, original-point evaluation, and prediction in original dataset class IDs.
- Model bundles with safetensors weights, normalization, metadata, and checksums.
- Separate CPU and GPU container targets with explicit input, output, and cache mounts.

Use the [GPU and CPU quickstart](quickstart.md) for a complete run.

## Next stages

1. Add resumable training with optimizer, scheduler, and random-state recovery.
2. Add a real-data recipe and establish a fresh model-quality baseline.
3. Add durable run-state handling for interruption and retries.
4. Package explicit batch runs as Helm Jobs with preserved dataset and result storage.
5. Add optional Rancher Fleet examples for existing cluster operators.
6. Add repeatable GPU performance benchmarks with fixed inputs and model bundles.

Helm and Fleet will package the same CLI. Configuration reconciliation must not
silently launch training runs. A GPU host, cluster provisioner, or experiment server
is not a required dependency of the local package.

BIM export, multi-GPU training, and additional source formats remain follow-up
work. The existing project history preserves the original research implementation.
