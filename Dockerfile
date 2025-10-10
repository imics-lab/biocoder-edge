# BioCoder-Edge Dockerfile
# This Dockerfile supports both CPU and GPU (NVIDIA CUDA) deployments
# For GPU support, use docker-compose-gpu.yml

# -----------------------------------------------------------------------------
# Base Image Selection
# -----------------------------------------------------------------------------
# Default: CPU-only Python base image
# For GPU: Use nvidia/cuda base image (see docker-compose-gpu.yml)
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# -----------------------------------------------------------------------------
# Build Arguments and Environment Variables
# -----------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------------------------------------------------------
# Install System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Video and image processing
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    # For OpenCV
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    # For camera access and USB devices
    v4l-utils \
    # Networking utilities
    curl \
    wget \
    # Build tools (needed for some Python packages)
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Create Application User (security best practice)
# -----------------------------------------------------------------------------
RUN groupadd -r biocoder && useradd -r -g biocoder biocoder

# -----------------------------------------------------------------------------
# Set Working Directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Install Python Dependencies
# -----------------------------------------------------------------------------
# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Copy Application Code
# -----------------------------------------------------------------------------
COPY main.py .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docker-entrypoint.sh .

# -----------------------------------------------------------------------------
# Make Entrypoint Script Executable
# -----------------------------------------------------------------------------
RUN chmod +x /app/docker-entrypoint.sh

# -----------------------------------------------------------------------------
# Create Required Directories
# -----------------------------------------------------------------------------
RUN mkdir -p \
    /app/config \
    /app/data/pending_upload \
    /app/data/uploaded \
    /app/model_weight \
    /app/logs \
    /tmp/biocoder_edge_temp \
    && chown -R biocoder:biocoder /app /tmp/biocoder_edge_temp

# -----------------------------------------------------------------------------
# Set Ownership
# -----------------------------------------------------------------------------
# Change ownership of all application files to the biocoder user
RUN chown -R biocoder:biocoder /app

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
# Simple health check to verify the container is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep -f "python.*main.py" || exit 1

# -----------------------------------------------------------------------------
# Switch to Non-Root User
# -----------------------------------------------------------------------------
USER biocoder

# -----------------------------------------------------------------------------
# Expose Ports (if using live view web server)
# -----------------------------------------------------------------------------
# Port 8080 is used by the optional stream_server.py for remote viewing
EXPOSE 8080

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
# Default command runs the main application
# Can be overridden in docker-compose.yml or with docker run
CMD ["python", "main.py"]
