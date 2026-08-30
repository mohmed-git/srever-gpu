# =====================================================================
# Lingua Buds translation server -- container image
# =====================================================================
# Two targets, one file:
#
#   gpu  (default) -- CUDA 12.4 + cuDNN 9 runtime, for the RTX 3060 host.
#   cpu            -- slim Debian, no CUDA. For CI and for verifying the
#                     code path without a GPU. It WILL run and every number
#                     it reports will be real; it will NOT meet 150 ms.
#
# Build:
#   docker build -t lingua-server:gpu .
#   docker build -t lingua-server:cpu --target cpu .
#
# Run (GPU):
#   docker run --gpus all -p 8080:8080 \
#     -v hf-cache:/cache/huggingface \
#     --env-file .env lingua-server:gpu
#
# Run (CPU, for verification only):
#   docker run -p 8080:8080 -e DEVICE=cpu -e ASR_MODEL=tiny \
#     -v hf-cache:/cache/huggingface lingua-server:cpu
#
# WHY --gpus all AND NOT just nvidia-docker: faster-whisper reaches CUDA
# through CTranslate2, which links libcudnn at load time. Without the
# device *and* the cuDNN libraries in the image, ctranslate2 reports
# get_cuda_device_count() == 0 and app/config.py auto-detects "cpu" --
# the container starts successfully and serves 10x slower. That is the
# single most likely silent misconfiguration of this image, which is why
# the GPU target sets DEVICE=cuda explicitly (see below) so a missing GPU
# fails loudly at startup instead of degrading quietly.
# =====================================================================


# ---------------------------------------------------------------------
# Shared: pip settings used by both targets
# ---------------------------------------------------------------------
# PIP_NO_CACHE_DIR keeps the layer small; the model cache is a volume, not
# a layer, so weights are never baked into the image.


# =====================================================================
# TARGET: cpu
# =====================================================================
FROM python:3.11-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    DEVICE=cpu

# ffmpeg/libsndfile: the audio module decodes Opus/OGG/WebM through PyAV,
# which needs the ffmpeg shared libraries. Without them, every compressed
# upload fails at decode -- and because audio.py raises rather than
# swallowing, it fails visibly, which is the intent.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY tests ./tests
COPY run.py .

# Non-root: the process needs no privileges beyond reading its own code
# and writing the model cache.
RUN useradd -m -u 10001 lingua \
    && mkdir -p /cache/huggingface \
    && chown -R lingua:lingua /cache/huggingface /app
USER lingua

EXPOSE 8080

# The health check asks /health, not /. /health reports ready=false with a
# startup_error when an engine failed to load, so a container whose model
# download failed is reported unhealthy instead of answering 200 with dead
# engines. start-period is generous because a cold model load is slow:
# MEASURED 8.32 s for tiny+M2M100 on 2 cores; large-v3-turbo from a cold
# cache is minutes (download), seconds (warm volume).
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health \
        | grep -q '"ready": *true' || exit 1

CMD ["python", "run.py"]


# =====================================================================
# TARGET: gpu  (default -- last stage in the file)
# =====================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface

# DEVICE is pinned rather than auto-detected. Rationale above: if this
# container is started without --gpus all, auto-detection would return
# "cpu" and the server would come up healthy and 10x slow. With
# DEVICE=cuda, CTranslate2 raises at model load, the lifespan handler
# records STARTUP_ERROR, /health reports ready=false, and the health
# check fails. A loud failure is the correct outcome for a GPU image
# with no GPU.
ENV DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# vLLM is installed HERE and not in requirements.txt on purpose: it is a
# ~2 GB CUDA-specific wheel that cannot install on a CPU-only box, so
# putting it in requirements.txt would break the cpu target and CI.
# This is the MT path the brief asks for (Continuous Batching +
# PagedAttention).
RUN python -m pip install "vllm>=0.6.0"

COPY app ./app
COPY tests ./tests
COPY run.py .

RUN useradd -m -u 10001 lingua \
    && mkdir -p /cache/huggingface \
    && chown -R lingua:lingua /cache/huggingface /app
USER lingua

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health \
        | grep -q '"ready": *true' || exit 1

CMD ["python", "run.py"]
