# BioCoder-Edge Docker Deployment Guide

This guide explains how to build and run BioCoder-Edge using Docker Compose on systems with NVIDIA GPU support.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Prerequisites](#prerequisites)
- [Quick Start - Generic GPU](#quick-start---generic-gpu)
- [Quick Start - Jetson Devices](#quick-start---jetson-devices)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Deployment Options

BioCoder-Edge provides two Docker Compose configurations to support different hardware platforms:

### 1. **docker-compose-gpu.yml** - Generic GPU Machines
- **Use case:** Desktop/server machines with NVIDIA GPUs (RTX, Tesla, A100, etc.)
- **Base image:** `nvcr.io/nvidia/pytorch:24.12-py3` (CUDA 12.6.3, PyTorch, torchvision)
- **Requirements:** NVIDIA GPU driver R560+ supporting CUDA 12.6
- **Optimizations:** Standard CUDA memory management, larger shared memory (2GB)
- **Camera support:** Standard USB cameras via V4L2

### 2. **docker-compose-jetson.yml** - NVIDIA Jetson Devices
- **Use case:** NVIDIA Jetson Nano, Jetson Orin, and other Jetson family devices
- **Base image:** `dustynv/l4t-ml:r36.2.0` (JetPack-optimized with L4T)
- **Requirements:** JetPack SDK installed on host
- **Optimizations:** `cudaMallocAsync` backend for Jetson memory management, privileged mode for V4L2
- **Camera support:** USB cameras and CSI cameras with full media controller access

**Which one should I use?**
- Use `docker-compose-gpu.yml` for x86_64 machines with standard NVIDIA GPUs
- Use `docker-compose-jetson.yml` for ARM64-based Jetson devices

## Prerequisites

1.  **Docker** (version 20.10 or later)
    ```bash
    # Install Docker on Ubuntu/Debian
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    # Add your user to docker group (logout/login required)
    sudo usermod -aG docker $USER
    ```

2.  **NVIDIA Docker Runtime**
    ```bash
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    ```

## Quick Start - Generic GPU

For desktop/server machines with standard NVIDIA GPUs (RTX, Tesla, A100, etc.):

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-username/biocoder-edge.git
    cd biocoder-edge
    ```

2.  **Prepare configuration**:
    Ensure `config/config.yaml` exists and is properly configured for your camera and device settings. See the main `README.md` for more details.

3.  **Ensure model weights are present**:
    Place your trained YOLO model in `model_weight/best.pt`.

4.  **Build and run**:
    ```bash
    docker compose -f docker-compose-gpu.yml up --build -d
    ```

5.  **View logs**:
    ```bash
    docker compose -f docker-compose-gpu.yml logs -f
    ```

6.  **Access Live View**:
    Once the container is running, the live view is available at `http://<YOUR_DEVICE_IP>:8080`.

7.  **Stop the application**:
    ```bash
    docker compose -f docker-compose-gpu.yml down
    ```

## Quick Start - Jetson Devices

For NVIDIA Jetson devices (Jetson Nano, Jetson Orin, etc.):

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-username/biocoder-edge.git
    cd biocoder-edge
    ```

2.  **Prepare configuration**:
    Ensure `config/config.yaml` exists and is properly configured for your camera and device settings. See the main `README.md` for more details.

3.  **Ensure model weights are present**:
    Place your trained YOLO model in `model_weight/best.pt`.

4.  **Build and run**:
    ```bash
    docker compose -f docker-compose-jetson.yml up --build -d
    ```

5.  **View logs**:
    ```bash
    docker compose -f docker-compose-jetson.yml logs -f
    ```

6.  **Access Live View**:
    Once the container is running, the live view is available at `http://<YOUR_JETSON_IP>:8080`.

7.  **Stop the application**:
    ```bash
    docker compose -f docker-compose-jetson.yml down
    ```

## Configuration

### Docker Compose
If you need to modify the Docker setup (e.g., change the camera device from `/dev/video0`), edit the appropriate compose file:
- `docker-compose-gpu.yml` for generic GPU machines
- `docker-compose-jetson.yml` for Jetson devices

### Application
All application-level settings (camera resolution, model confidence, uploader settings, etc.) are in `config/config.yaml`.

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA runtime is installed on the host
docker info | grep -i runtime

# Test GPU access inside the running container
docker exec -it biocoder-edge-gpu nvidia-smi
```

### Camera Not Found
```bash
# Verify camera is connected on the host
ls -la /dev/video*

# Ensure the correct device is mapped in your docker-compose file
# (docker-compose-gpu.yml or docker-compose-jetson.yml)
```

### Start an interactive shell inside the container
```bash
docker exec -it biocoder-edge-gpu /bin/bash
```

### Switching Between Configurations
If you need to switch from one configuration to another (e.g., from Jetson to generic GPU):

1. Stop the current container:
   ```bash
   docker compose -f docker-compose-jetson.yml down
   ```

2. Build and start with the new configuration:
   ```bash
   docker compose -f docker-compose-gpu.yml up --build -d
   ```

Note: The two configurations use the same container name, so they cannot run simultaneously.