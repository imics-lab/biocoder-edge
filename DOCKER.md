# BioCoder-Edge Docker Deployment Guide

This guide explains how to build and run BioCoder-Edge using Docker Compose on a system with an NVIDIA GPU.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

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

## Quick Start

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

## Configuration

### Docker Compose
If you need to modify the Docker setup (e.g., change the camera device from `/dev/video0`), edit `docker-compose-gpu.yml`.

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

# Ensure the correct device is mapped in docker-compose-gpu.yml
```
