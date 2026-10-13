# RPTU AI Toolkit

RPTU AI Toolkit provides a CPU workflow for generating synthetic point clouds,
validating raw and prepared datasets, and preprocessing points into voxels. The project
originated in LiDAR segmentation and BIM research and is being modernized for
reusable public datasets and reproducible workflows.

The current development version supports:

- An installable Python package and noninteractive `rptu` command.
- Strict configuration and dataset validation, including SHA-256 file checksums.
- Numeric NPZ scenes and an optional PLY reader.
- Seeded synthetic scenes for checking data correctness.
- CPU voxelization with preserved point IDs, original labels, and point-to-voxel mappings.
- A portable prepared-dataset reader and CPU validation of complete prepared bundles.

Training, evaluation, prediction, and benchmarking are not implemented in the new
CLI yet. WarpConvNet is the planned sparse-convolution backend; the GPU runtime
has not been validated. No original research datasets or trained model artifacts
are included. The older research scripts are incomplete and are not a supported
end-to-end workflow.

Start with the [CPU quickstart](quickstart.md), then read the
[CLI reference](cli-reference.md) and [dataset contract](dataset-contract.md).
Use the [configuration reference](configuration.md) for application YAML and the
[development guide](development.md) for testing and package builds.

The [roadmap](todo.md) distinguishes implemented features from planned work.
The Train, Test, Utils, Preprocess, and Model pages explain the current status of
the former research workflow.

Source: [ajavanma/RPTU_AI_Toolkit on GitHub](https://github.com/ajavanma/RPTU_AI_Toolkit).
The project is MIT licensed.
