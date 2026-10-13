# GPU and CPU quickstart

Use a source checkout and a Python virtual environment. The GPU profile uses
Linux x86_64, Python 3.11, and an NVIDIA driver compatible with CUDA 12.8. The
reference GPU is the NVIDIA RTX 4070. The CPU fallback uses Python 3.11 or 3.12
on Linux or macOS.

```sh
git clone https://github.com/ajavanma/RPTU_AI_Toolkit.git
cd RPTU_AI_Toolkit
python3.11 -m venv .venv
source .venv/bin/activate
```

## Install the GPU profile

```sh
python -m pip install -r requirements/gpu-cu128.txt
python -m pip install '.[ml]'
```

This profile pairs PyTorch 2.10.0 with CUDA 12.8 and WarpConvNet 1.8.2. The
WarpConvNet wheel is pinned by release URL and SHA-256. It is specific to Linux
x86_64 and CPython 3.11.

## Install the CPU fallback

For CPU-only Linux installations, install the CPU PyTorch wheel first:

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[ml]'
```

On macOS, use `python -m pip install '.[ml]'`. For data preparation alone,
`python -m pip install .` installs only NumPy and PyYAML. ASCII and binary PLY
input is available with `python -m pip install '.[io]'`.

## Prepare the demonstration data

```sh
rptu doctor
rptu synthetic --output runs/demo-data --seed 42
rptu inspect --manifest runs/demo-data/manifest.json
rptu preprocess --manifest runs/demo-data/manifest.json --output runs/demo-prepared --voxel-size 0.1
rptu inspect --prepared runs/demo-prepared/prepared.json
```

The four synthetic scenes belong to train, validation, test, and prediction
splits. They include invalid rows, ignored labels, class collisions, and negative
coordinates. The dataset uses meters, so voxel size `0.1` is ten centimeters.
Use `--manifest` for raw data and `--prepared` for prepared metadata.

## Train, evaluate, and predict

```sh
rptu train --prepared runs/demo-prepared/prepared.json --output runs/demo-model --epochs 20 --device auto
rptu evaluate --prepared runs/demo-prepared/prepared.json --model runs/demo-model --output runs/demo-evaluation
rptu predict --prepared runs/demo-prepared/prepared.json --model runs/demo-model --output runs/demo-predictions
```

Automatic training selects WarpConvNet when CUDA and its dependencies are
available, otherwise the CPU PyTorch fallback. To select explicitly, add
`--backend warpconvnet --device cuda` or `--backend torch --device cpu` to `train`.
Evaluation and prediction use the architecture saved in the model bundle.

Outputs contain model weights and metadata, point-level evaluation metrics, or
numeric prediction archives, respectively. Predictions use the original dataset
class IDs and preserve source point IDs. Removed rows have the dataset's ignore
label and a false validity flag. See [test and evaluation](test.md).

New runs, evaluation, and prediction require new output directories. To resume
an interrupted training run, keep its prepared dataset and output directory:

```sh
rptu train --prepared runs/demo-prepared/prepared.json --output runs/demo-model --resume
```

Omitted training settings come from the run. Resume continues to the original
total epoch count, using the latest completed epoch. A completed run returns its
verified saved report without retraining. Use a new output directory to change
settings. See [training](train.md) for checkpoint and filesystem details.
Existing successful artifacts are never overwritten. The default limit is 100,000 voxels per
scene; `--max-voxels` changes it explicitly. A scene above the limit is rejected
rather than truncated. Reduce voxel density or prepare smaller scene chunks to
fit the available memory.

The workflow commands accept `--json` for machine-readable output. Model training uses
only the training split for optimization and feature normalization. Synthetic
results are a functional demonstration, not a real-data accuracy benchmark.

For container execution, use the repository's
[container guide](https://github.com/ajavanma/RPTU_AI_Toolkit/blob/main/docs/containers.md).
