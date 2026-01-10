# GPU-friendly default base. Override with --build-arg BASE_IMAGE=python:3.11-slim for CPU-only.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for building/installing fairseq and friends.
# (Some wheels may not be available for your platform; keep build tools present.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# If you use a different CUDA version than the base image, install torch yourself.
# For the pytorch/pytorch CUDA runtime base image, torch is already present.
RUN python -m pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY main.py translation_service.py /app/

# HuggingFace cache location (mount a volume here in Docker/K8s)
ENV HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    HF_CACHE_DIR=/data/hf \
    TRANSFORMERS_CACHE=/data/hf/transformers

RUN mkdir -p /data/hf && chmod 0777 /data/hf

EXPOSE 8000

# Single worker to avoid duplicating large models in memory.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
