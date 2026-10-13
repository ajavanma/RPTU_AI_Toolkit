# GPU and CPU containers

Both images expose the same `rptu` command. The GPU image uses WarpConvNet on
NVIDIA hardware, with an RTX 4070 as the reference configuration. The CPU image
provides the PyTorch fallback. Data preparation works with either image.

| Image target | Platform | Python | PyTorch | Sparse backend |
| --- | --- | --- | --- | --- |
| `gpu` | Linux amd64 | 3.11 | 2.10.0, CUDA 12.8 | WarpConvNet 1.8.2 |
| `cpu` | Linux amd64 or arm64 | 3.11 | 2.10.0, CPU | PyTorch |

The GPU host needs an NVIDIA driver supporting CUDA 12.8 and the NVIDIA Container
Toolkit configured for Docker. The image includes the CUDA userspace libraries;
a host CUDA compiler is not required.

## Build

From the repository root:

```sh
docker build --platform linux/amd64 --target gpu -t rptu-ai-toolkit:gpu .
docker build --target cpu -t rptu-ai-toolkit:cpu .
```

Images contain the installed package and its dependencies. Datasets, credentials,
model weights, and the legacy research environment are excluded from the build
context. No dataset or checkpoint is downloaded when a container starts.

The GPU install profile pins the WarpConvNet wheel by release URL and SHA-256 in
[`requirements/gpu-cu128.txt`](../requirements/gpu-cu128.txt). Its Python,
PyTorch, and CUDA versions form one ABI-compatible set. The CPU image installs
PyTorch from its CPU wheel index. Build dependencies and OS packages can still
change between builds; retain image digests when recording a particular run.

## Mount inputs and outputs

Containers run as UID/GID `10001:10001`. For bind mounts on Linux, run with your
host UID/GID so new files remain writable by your account. Use these locations:

| Container path | Mount | Contents |
| --- | --- | --- |
| `/input` | Read-only | Raw or prepared dataset bundle |
| `/models` | Read-only | Saved model bundle when evaluating or predicting |
| `/output` | Writable | Parent directory for a new result directory |
| `/cache` | Writable | PyTorch, Triton, CUDA, and WarpConvNet caches |

Create mount directories before running. Each command's `--output` names a new
directory underneath `/output`; existing results are never overwritten. Training
with `--resume` reuses its run directory and restores the saved settings. Keep the
complete writable run directory between container invocations. The checkpoint
and locking contract targets local POSIX filesystems, including local bind
mounts and local Docker volumes; it does not provide distributed coordination
for NFS, object storage, or multiple copies of a run directory.

For example, after preparing a dataset on the host:

```sh
mkdir -p runs/models runs/cache
docker run --rm --gpus all --read-only --tmpfs /tmp:rw,nosuid,size=256m \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/runs/demo-prepared,dst=/input,readonly" \
  --mount "type=bind,src=$PWD/runs/models,dst=/output" \
  --mount "type=bind,src=$PWD/runs/cache,dst=/cache" \
  rptu-ai-toolkit:gpu train \
  --prepared /input/prepared.json --output /output/baseline \
  --device auto --backend auto --epochs 20 --seed 42
```

For CPU execution, use `rptu-ai-toolkit:cpu`, omit `--gpus all`, and set
`--device cpu --backend torch`. The GPU image can also run the CPU fallback with
those command options and without `--gpus all`.

To recover an interrupted run, repeat the same mounts and image, replacing the
training arguments with:

```sh
train --prepared /input/prepared.json --output /output/baseline --resume
```

Do not change the prepared inputs. Resume restores the latest complete epoch;
an incomplete epoch is repeated. A completed retry checks its saved artifacts
and returns the original report without training again.

Evaluate the saved model on the test split:

```sh
mkdir -p runs/results
docker run --rm --gpus all --read-only --tmpfs /tmp:rw,nosuid,size=256m \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/runs/demo-prepared,dst=/input,readonly" \
  --mount "type=bind,src=$PWD/runs/models,dst=/models,readonly" \
  --mount "type=bind,src=$PWD/runs/results,dst=/output" \
  --mount "type=bind,src=$PWD/runs/cache,dst=/cache" \
  rptu-ai-toolkit:gpu evaluate \
  --prepared /input/prepared.json --model /models/baseline \
  --output /output/evaluation --device auto --split test
```

Use the same mounts for `predict`, selecting a new output directory and
`--split predict`.

## CPU container smoke workflow

```sh
sh scripts/container-smoke.sh rptu-ai-toolkit:cpu
```

This creates synthetic data, preprocesses it, inspects the prepared contract,
trains for two epochs, evaluates, predicts, and retries the completed training run. It uses a read-only container
filesystem and explicit writable mounts, checks that repeated outputs are
rejected without changing existing artifact hashes and input mounts cannot be
written, checks report and checkpoint hashes and output ownership, then removes
its temporary files.
The same script runs in CI. The separate CPU model test job exercises model
artifacts without adding ML dependencies to the core package tests.

## Install the GPU profile without Docker

In a Linux x86_64 Python 3.11 virtual environment, from the repository root:

```sh
python -m pip install -r requirements/gpu-cu128.txt
python -m pip install '.[ml]'
```

The package's `ml` extra pins PyTorch and safetensors. WarpConvNet is deliberately
in the platform-specific profile, keeping the base installation and CPU fallback
free of CUDA extensions. The versions follow the official
[PyTorch 2.10 installation instructions](https://pytorch.org/get-started/previous-versions/#v2100)
and [WarpConvNet 1.8.2 release assets](https://github.com/NVlabs/WarpConvNet/releases/tag/v1.8.2).
