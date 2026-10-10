# Byte simulation for NVIDIA/CUDA machines (plan_evotorch.md, Step 9).
#
# The same stack as the validated Linux laptop: Ubuntu 24.04 (glibc 2.39, so the scalar
# simulator's math.tanh matches tests/refs/linux-x86_64/), Python 3.12, torch 2.13.0+cu130.
# The CUDA libraries come with the torch wheel; the host provides only the driver (NVIDIA
# Container Toolkit, `docker run --gpus all`). Build, move and run: see README.md.

# ubuntu:24.04, pinned (digest of 2026-10-10) so every rebuild starts from the same base.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

# python3.12-dev + build-essential: torch.compile (Triton) compiles a small C launcher at
# run time; without them `compile: true` fails. No Qt: mvb imports it only for viz.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Dependencies before the code, so editing code does not reinstall them. torch from the
# CUDA 13.0 index first; then requirements.txt without the GUI-only packages (PySide6,
# pynput), whose torch==2.13.0 pin the installed 2.13.0+cu130 already satisfies.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130 \
 && grep -v -E '^(PySide6|pynput)==' /tmp/requirements.txt > /tmp/requirements-docker.txt \
 && pip install --no-cache-dir -r /tmp/requirements-docker.txt \
 && python -c "import torch; assert torch.__version__ == '2.13.0+cu130', torch.__version__"

WORKDIR /app
COPY mvb/ mvb/
COPY mvb_torch/ mvb_torch/
COPY simulate/ simulate/
COPY configs/ configs/
COPY tests/ tests/

# The container runs as the host user (`--user $(id -u):$(id -g)`), who may not exist in
# /etc/passwd: give it a writable home and caches, and a writable data/ when none is
# mounted (the tests write to data/temp). Nothing here changes numerics.
RUN mkdir -p /app/data /cache && chmod 1777 /app/data /cache
ENV HOME=/cache \
    USER=byte \
    LOGNAME=byte \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton \
    MPLCONFIGDIR=/cache/matplotlib \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1

CMD ["python", "-m", "tests.gpu_profile"]
