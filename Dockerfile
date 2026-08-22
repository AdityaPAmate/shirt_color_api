# =============================================================================
# Dockerfile -- Shirt/Kurta Virtual Fabric Replacement API
#
# CPU-only image (project's hard constraint: device="cpu", no GPU dependency).
# Base: python:3.11-slim (matches project's Python 3.11 requirement).
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# -----------------------------------------------------------------------------
# System dependencies
#
# - build-essential : GroundingDINO's setup.py compiles a C++/PyTorch
#                      extension at install time; needs a compiler toolchain.
# - git              : requirements.txt has "-e git+https://..." (GroundingDINO)
#                      and "git+https://..." (SAM-2) VCS lines -- pip needs
#                      git on PATH to clone/install these.
# - libgl1, libglib2.0-0, libsm6, libxext6, libxrender1 :
#                      opencv-python (non-headless) needs these even without
#                      an actual display (cv2 import fails otherwise on slim
#                      Debian images).
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install PyTorch CPU wheels FIRST, explicitly, from the CPU index.
#
# WHY explicit --index-url: since PyTorch 2.11, plain `pip install torch`
# on Linux pulls the CUDA build by default (multi-GB, and useless here since
# the whole project runs with device="cpu"). The CPU-only wheel is much
# smaller and is what this project actually needs.
#
# Versions match exactly what `pip freeze` showed in the working local
# environment (torch==2.13.0, torchvision==0.28.0, torchaudio==2.11.0).
# -----------------------------------------------------------------------------
RUN pip install --upgrade pip && \
    pip install \
        torch==2.13.0 \
        torchvision==0.28.0 \
        torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cpu

# -----------------------------------------------------------------------------
# Install the rest of the requirements (Django, DRF, opencv, numpy, scipy,
# gunicorn, and the GroundingDINO / SAM-2 VCS installs).
#
# Copied separately from the rest of the app so Docker can cache this layer
# -- as long as requirements.txt doesn't change, this slow step (compiling
# GroundingDINO's extension, cloning SAM-2) won't re-run on every code edit.
# -----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Copy the rest of the project code.
#
# NOTE: ai_models/ (model checkpoints, ~811 MB total) is intentionally
# excluded here via .dockerignore for LOCAL testing -- it is mounted as a
# volume instead (see docker-compose.yml) so editing code doesn't force a
# días slow re-copy of the checkpoints on every rebuild. For an actual cloud
# deploy image later, ai_models/ would need to either be included (remove
# the .dockerignore exclusion) or fetched at container startup -- revisit
# this when the deployment-target step is reached.
# -----------------------------------------------------------------------------
COPY . .

# Directories the app writes to at runtime (uploads, outputs, debug images)
RUN mkdir -p media/uploads media/outputs test_images/debug

EXPOSE 8000

# -----------------------------------------------------------------------------
# --workers 1 is REQUIRED, not optional.
#
# AppConfig.ready() (api/apps.py) loads GroundingDINO + SAM 2.1 into memory
# once per worker PROCESS via pipeline_singleton.preload_pipeline(). Gunicorn
# workers are separate OS processes -- more than 1 worker means the models
# get loaded into memory multiple times simultaneously. Until memory usage
# is measured and confirmed safe for more, keep this at 1.
#
# --timeout 300 : CPU-only GroundingDINO+SAM2.1+RTV inference can take well
# over gunicorn's default 30s timeout; raised to avoid the worker being
# killed mid-request.
# -----------------------------------------------------------------------------
CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "300"]