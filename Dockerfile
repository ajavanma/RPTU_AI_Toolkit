# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS wheel
WORKDIR /build
COPY pyproject.toml MANIFEST.in README.md LICENSE ./
COPY src/rptu_ai_toolkit/ src/rptu_ai_toolkit/
RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/cache \
    TORCH_HOME=/cache/torch \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    TRITON_CACHE_DIR=/cache/triton \
    CUDA_CACHE_PATH=/cache/cuda \
    WARPCONVNET_BENCHMARK_CACHE_DIR=/cache/warpconvnet \
    HOME=/home/rptu
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rptu \
    && useradd --uid 10001 --gid rptu --create-home rptu \
    && mkdir -p /work /input /output /cache \
    && chown rptu:rptu /work /output /cache
WORKDIR /work
ENTRYPOINT ["rptu"]
CMD ["--help"]

# Portable fallback, with CPU-only PyTorch on Linux amd64 and arm64.
FROM runtime AS cpu
RUN python -m pip install --no-cache-dir 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
COPY --from=wheel /wheels/ /wheels/
RUN for wheel in /wheels/*.whl; do python -m pip install --no-cache-dir "${wheel}[ml]"; done \
    && rm -rf /wheels \
    && python -m pip check
USER 10001:10001

# The pinned WarpConvNet wheel targets Linux amd64, CPython 3.11, CUDA 12.8.
FROM runtime AS gpu
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY requirements/gpu-cu128.txt /tmp/gpu-cu128.txt
RUN test "$(uname -m)" = "x86_64" \
    && python -m pip install --no-cache-dir -r /tmp/gpu-cu128.txt \
    && rm /tmp/gpu-cu128.txt
COPY --from=wheel /wheels/ /wheels/
RUN for wheel in /wheels/*.whl; do python -m pip install --no-cache-dir "${wheel}[ml]"; done \
    && rm -rf /wheels \
    && python -m pip check
USER 10001:10001
