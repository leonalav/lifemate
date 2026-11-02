# LifeMate — the offline inference image for the final submission.
#
# The competition requires a Docker image containing the training code, the inference
# code and the data, runnable with no network access (docs/VERIFIED_FACTS.md §1).
# Everything the system needs is baked in: the LLM weights, the ASR ONNX bundle and
# the Python packages.
#
# The base image provides a working CUDA build of torch; the packages installed on
# top deliberately exclude torch/torchvision/tensorflow/jax (docs/PRD.md §7).
#
# Build:
#   docker build -t lifemate:latest \
#     --build-arg BASE_MODEL=Qwen/Qwen3.5-4B \
#     --build-arg LLM_DIR=/models/llm \
#     --build-arg ASR_DIR=/models/gipformer .
#
# The model directories are copied in by the build context, so they must be present
# where the build runs. They are never downloaded at runtime.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

# python3.10+ is required by the competition kit's runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        ffmpeg libsndfile1 ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The dependency layer changes rarely, so it is cached separately from the code.
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements.txt \
    && python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# The application code.
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/scripts/*.sh \
    && python3 -m pip install --no-deps -e /app

# The models, baked in. The build context must contain them.
ARG LLM_DIR=/models/llm
ARG ASR_DIR=/models/gipformer
COPY ${LLM_DIR} /models/llm
COPY ${ASR_DIR} /models/gipformer

# The competition mounts the private test at /data and collects result/submission.jsonl.
RUN mkdir -p /data /app/result

WORKDIR /app

ENTRYPOINT ["/app/entrypoint.sh"]
