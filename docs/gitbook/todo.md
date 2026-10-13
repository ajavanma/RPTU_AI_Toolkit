# Roadmap

The toolkit is being modernized in independently testable stages. This page
separates supported behavior from proposed work.

## Implemented

- Installable CPU package, locked development dependencies, and a noninteractive CLI.
- Strict application configuration with clear validation errors.
- Versioned dataset manifests, SHA-256 checksums, numeric NPZ input, and optional PLY input.
- Explicit splits, reversible class mappings, and preserved point identities.
- Deterministic synthetic scenes and CPU voxel preprocessing.
- Portable prepared-dataset reading through the Python API and
  `rptu inspect --prepared`, with checksums and cross-array validation.
- Unit and CLI acceptance tests, source/wheel builds, and clean installation checks.

Use the [quickstart](quickstart.md) for the current runnable workflow.

## Next work without a GPU

- Package the CPU workflow in a reproducible container and test its input/output
  mounts, nonprivileged execution, and artifact preservation.
- Establish durable run outputs and consistent execution metadata for future
  training and evaluation.
- Keep documentation and examples aligned with the supported CLI.

These items are planned and are not additional current commands.

## After establishing a GPU host

- Implement a fresh WarpConvNet model and validate a specific Python, PyTorch,
  WarpConvNet, and CUDA combination.
- Add training, evaluation, checkpoint resume, and prediction.
- Establish new quality and runtime baselines from reproducible inputs.
- Add optional Helm Jobs, then Fleet examples and measured GPU benchmarks.

GPU work is deferred. Helm and Fleet examples must wrap a useful, tested workflow;
installing a chart or reconciling configuration should not silently launch an
expensive training run.

BIM export, multi-GPU training, and additional source formats remain future work.
The old TensorBoard, scheduler, dropout, and weight-decay checklist was a research
note, not a statement that a supported model pipeline exists today.
