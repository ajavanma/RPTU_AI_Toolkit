# Roadmap

## Available workflow

- Installable package and CLI with optional ML and PLY dependencies.
- Versioned dataset contracts, reversible class mappings, and preserved point IDs.
- Seeded synthetic scenes, CPU voxel preprocessing, and prepared-bundle inspection.
- WarpConvNet MinkUNet18 for NVIDIA GPUs and a portable PyTorch CPU fallback.
- Training, original-point evaluation, and prediction in original dataset class IDs.
- Model bundles with safetensors weights, normalization, metadata, and checksums.
- Epoch checkpoints with Adam and random state, explicit resume, durable attempt
  records, and verified completed retries.
- Separate CPU and GPU container targets with explicit input, output, and cache mounts.

Use the [GPU and CPU quickstart](quickstart.md) for a complete run.

## Next stages

1. Add a public-data recipe and spatial chunking for larger scenes.
2. Publish versioned packages and images with schema compatibility guidance.
3. Package explicit batch runs as Helm Jobs with preserved dataset and result storage.
4. Add optional Rancher Fleet examples for existing cluster operators.
5. Add repeatable GPU performance benchmarks with fixed inputs and model bundles.

Training currently resumes at completed epoch boundaries using the original run
settings. Step-level checkpoints, changing settings while resuming, scheduler
state, and distributed storage coordination remain separate future extensions.

Helm and Fleet will package the same CLI. Configuration reconciliation must not
silently launch training runs. A GPU host, cluster provisioner, or experiment server
is not a required dependency of the local package.

BIM export, multi-GPU training, and additional source formats remain follow-up
work. The existing project history preserves the original research implementation.
