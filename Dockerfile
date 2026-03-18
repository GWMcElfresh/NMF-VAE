# =============================================================================
# NMF-VAE  —  GPU-capable multi-stage Docker image
#
# Stages
# ------
#   deps     — Python + CUDA runtime + all pip dependencies (heavy layer,
#              cached monthly via the GWMcElfresh/dockerDependencies workflow)
#   runtime  — thin layer that adds the application source code on top of deps
#
# Build-arg protocol (used by GWMcElfresh/dockerDependencies/docker-cache.yml)
# -----------------------------------------------------------------------
#   BASE_IMAGE      — CUDA base to start from (monthly base-deps image or the
#                     default nvidia/cuda image below)
#   SKIP_BASE_DEPS  — set to "true" when BASE_IMAGE already contains the pip
#                     deps layer (avoids re-installing on cache hits)
#   DEPS_IMAGE      — fully-built deps image to layer the runtime stage on top
#                     of (defaults to the deps stage in this file for local builds)
# =============================================================================

# --- Stage 1: deps -----------------------------------------------------------
ARG BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS deps

ARG SKIP_BASE_DEPS=false
ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python + C/C++ build tools required by some pip packages.
# We always run this step even on cache-hits because it is fast and the
# monthly base image may not pin a specific Python patch version.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        gcc g++ \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3   /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the dependency manifest so Docker can cache this layer
# independently of source code changes.
COPY requirements.txt .

# Install GPU-capable PyTorch (CUDA 12.1) and all project dependencies.
# Skipped when SKIP_BASE_DEPS=true because the base image already contains
# the installed packages (dependency-cache hit from docker-cache.yml).
RUN if [ "$SKIP_BASE_DEPS" != "true" ]; then \
      pip install --no-cache-dir \
          torch torchvision \
          --index-url https://download.pytorch.org/whl/cu121 \
      && pip install --no-cache-dir -r requirements.txt; \
    fi


# --- Stage 2: runtime --------------------------------------------------------
# When called by docker-cache.yml the DEPS_IMAGE build-arg is set to the
# fully-built (and potentially remote) deps image, bypassing the deps stage
# above.  Local/direct builds fall back to the deps stage in this file.
ARG DEPS_IMAGE=deps
FROM ${DEPS_IMAGE} AS runtime

WORKDIR /app

# Copy the application source and install in editable mode.
COPY . .
RUN pip install --no-cache-dir -e .

# Default command: run the test suite (CI/CD entrypoint).
# Override CMD or use `docker run … python scripts/train.py …` for real use.
CMD ["python", "-m", "pytest", "tests/"]
