# ============================================================
# Base image
# ============================================================
FROM python:3.11-slim


# ============================================================
# Environment
# ============================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8080


# ============================================================
# Working directory
# ============================================================
WORKDIR /app


# ============================================================
# System dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Copy requirements first
# ============================================================
COPY requirements.txt .


# ============================================================
# Install Python dependencies
# ============================================================
# ============================================================
# Install Python dependencies
# ============================================================

RUN pip install --upgrade pip setuptools wheel


# Install PyTorch first so GroundingDINO can use the installed package.
RUN pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies.
RUN pip install --no-build-isolation -r requirements.txt


# ============================================================
# Copy application source code
# ============================================================
COPY . .


# ============================================================
# Create runtime directories
# ============================================================
RUN mkdir -p \
    media/uploads \
    media/outputs \
    test_images/debug


# ============================================================
# Cloud Run listens on PORT environment variable
# ============================================================
EXPOSE 8080


# ============================================================
# Start Django application
#
# One worker is intentionally used because each worker would load
# GroundingDINO and SAM models into its own process memory.
# ============================================================
CMD ["python", "startup.py"]