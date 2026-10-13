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

## Development transition

The public RPTU repository's default `main` branch retains the available workflow
above. Its [`research-legacy` branch](https://github.com/ajavanma/RPTU_AI_Toolkit/tree/research-legacy)
preserves the code before modernization. Existing RPTU installation instructions,
commands, and containers remain applicable to modern `main`.

Future feature development moves to
[PointCloud Workflows](https://github.com/ajavanma/PointCloud-Workflows), an
independent **private** successor with the full Git history. Owner-granted access
is required. This public documentation continues to describe the RPTU baseline;
successor implementation and documentation are maintained separately.

## Possible successor work

These are candidate stages for PointCloud Workflows, not scheduled RPTU releases:

1. Add a public-data recipe and spatial chunking for larger scenes.
2. Publish versioned packages and images with schema compatibility guidance.
3. Package explicit batch runs as Helm Jobs with preserved dataset and result storage.
4. Add optional Rancher Fleet examples for existing cluster operators.
5. Add repeatable GPU performance benchmarks with fixed inputs and model bundles.

Training currently resumes at completed epoch boundaries using the original run
settings. Step-level checkpoints, changing settings while resuming, scheduler
state, and distributed storage coordination remain separate future extensions.

Helm and Fleet packaging should preserve explicit run submission. Configuration reconciliation must not
silently launch training runs. A GPU host, cluster provisioner, or experiment server
is not a required dependency of the local package.

BIM export, multi-GPU training, and additional source formats are further
successor candidates. The
[project lineage record](https://github.com/ajavanma/RPTU_AI_Toolkit/blob/main/docs/project-lineage.md)
describes the preserved research and modernization history.
