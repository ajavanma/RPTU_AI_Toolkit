# RPTU AI Toolkit

RPTU AI Toolkit is a GPU-first point-cloud segmentation toolkit for NVIDIA RTX 4070 systems, with a portable CPU fallback. It combines synthetic data generation, dataset validation, voxel preprocessing, training, evaluation, and prediction. The project originated in LiDAR segmentation and BIM research and provides reusable public dataset contracts and reproducible workflows.

WarpConvNet's MinkUNet18 provides spatial segmentation on NVIDIA GPUs. The
PyTorch voxel MLP provides a lightweight CPU fallback and can also run on CUDA.
Both use the same dataset contracts, CLI, original point mapping, and model-bundle
format. Saved models retain their architecture; a WarpConvNet model requires
CUDA, while a PyTorch model can move between CPU and CUDA.

Start with the [GPU and CPU quickstart](quickstart.md), then read the
[CLI reference](cli-reference.md), [dataset contract](dataset-contract.md), and
[model guide](model.md). The [training guide](train.md) explains model selection;
[test and evaluation](test.md) covers metrics and point-label exports.

The toolkit includes small seeded synthetic scenes so the complete workflow can
run without an external dataset account. These scenes demonstrate behavior, not
real-world segmentation quality. Train a fresh model on your own labeled data
for an application; original research datasets and weights are not distributed.

Training supports epoch checkpoints and `--resume`, with durable attempt records
and verified completed retries. Use `rptu` for the current workflow.

## Project lineage

The public RPTU repository's default `main` branch retains this modern workflow.
The [`research-legacy` branch](https://github.com/ajavanma/RPTU_AI_Toolkit/tree/research-legacy)
preserves the research-era code before modernization. These pages continue to
document the working RPTU package, commands, and containers on `main`.

Future feature development continues in
[PointCloud Workflows](https://github.com/ajavanma/PointCloud-Workflows), an
independent **private** successor with the full Git history. Owner-granted access
is required. The [roadmap](todo.md) records possible successor work; this public
site continues to cover the RPTU baseline. See the
[project lineage record](https://github.com/ajavanma/RPTU_AI_Toolkit/blob/main/docs/project-lineage.md)
for branch and release details.

Source: [ajavanma/RPTU_AI_Toolkit on GitHub](https://github.com/ajavanma/RPTU_AI_Toolkit).
The project is MIT licensed.
