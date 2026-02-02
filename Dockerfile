# Multi-arch GPU base with PyTorch preinstalled.
# Using NVIDIA NGC PyTorch container for consistent CUDA support across platforms.
# See: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# IMPORTANT: For NVIDIA Arm platforms (e.g., GB10 / DX Spark), the CUDA-enabled
# PyTorch build is shipped with the NGC base image. Using a separate Python (e.g.,
# via micromamba) often pulls a CPU-only torch wheel. We use the base image Python.

COPY requirements.txt /app/requirements.txt

# Install requirements
# Unlike the previous Fairseq-based setup, transformers installs cleanly without
# any special handling or patches.
# flash-attn compiles CUDA kernels during installation (takes several minutes)
RUN python -m pip install --no-cache-dir packaging ninja && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# Sanity check: ensure we're still on a CUDA-enabled torch build.
RUN python - <<"PY"
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
if torch.version.cuda is None:
    raise SystemExit("ERROR: torch.version.cuda is None (CPU-only torch installed); refusing to build")
PY

# Final sanity check: transformers should import cleanly
RUN python - <<"PY"
import transformers
print("transformers", transformers.__version__)
PY

COPY main.py translation_service.py /app/

# HuggingFace cache location (mount a volume here in Docker/K8s)
ENV HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    HF_CACHE_DIR=/data/hf \
    TRANSFORMERS_CACHE=/data/hf/transformers

RUN mkdir -p /data/hf && chmod 0777 /data/hf

EXPOSE 8000

# Single worker to avoid duplicating the large model (9GB) in memory.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
