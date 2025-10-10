# BioCoder-Edge Docker Deployment Guide

This guide explains how to build and run BioCoder-Edge using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [GPU Support](#gpu-support)
- [Camera Configuration](#camera-configuration)
- [Accessing Live View](#accessing-live-view)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Prerequisites

### Required Software

1. **Docker** (version 20.10 or later)
   ```bash
   # Install Docker on Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Add your user to docker group (logout/login required)
   sudo usermod -aG docker $USER
   ```

2. **Docker Compose** (version 2.0 or later)
   ```bash
   # Docker Compose v2 is included with Docker Desktop
   # Verify installation
   docker compose version
   ```

### For GPU Support (Optional)

If you plan to use NVIDIA GPU acceleration:

1. **NVIDIA Docker Runtime**
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker

   # Test GPU access
   docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

## Quick Start

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-username/biocoder-edge.git
   cd biocoder-edge
   ```

2. **Prepare configuration**:
   ```bash
   # Ensure config.yaml exists and is properly configured
   # Edit config/config.yaml with your settings
   nano config/config.yaml
   ```

3. **Ensure model weights are present**:
   ```bash
   # Place your YOLO model weights in model_weight/
   ls -la model_weight/best.pt
   ```

4. **Identify your camera device**:
   ```bash
   # List available video devices
   ls -la /dev/video*
   # or
   v4l2-ctl --list-devices
   ```

5. **Build and run** (CPU version):
   ```bash
   docker compose up --build -d
   ```

6. **View logs**:
   ```bash
   docker compose logs -f
   ```

## Configuration

### Before First Run

Edit `config/config.yaml` to configure:

1. **Video Source**: Set the camera device number (default is 2, but you may need 0 or 1)
   ```yaml
   motion_detector:
     video_source: 0  # Change to match your camera device
   ```

2. **Device ID and Location**: Update with your device's information
   ```yaml
   animal_analyzer:
     device_id: "YOUR_DEVICE_ID"
     location:
       latitude: YOUR_LATITUDE
       longitude: YOUR_LONGITUDE
   ```

3. **Model Path**: Verify YOLO model path is correct
   ```yaml
   animal_analyzer:
     yolo_model_path: "model_weight/best.pt"
   ```

4. **Uploader Settings** (if using cloud upload):
   ```yaml
   uploader:
     enabled: true  # Set to true when ready to upload
     sftp:
       host: "YOUR_SFTP_HOST"
       username: "YOUR_SFTP_USERNAME"
       ssh_key_path: "/path/to/your/ssh/private/key"
   ```

### Docker Compose Configuration

If you need to modify the Docker setup, edit `docker-compose.yml`:

- **Camera device**: Change the device mapping if your camera is not at `/dev/video0`
  ```yaml
  devices:
    - /dev/video1:/dev/video1  # For second camera
  ```

- **Port mapping**: Change the live view port if 8080 is already in use
  ```yaml
  ports:
    - "9090:8080"  # Access via http://localhost:9090
  ```

## Running the Application

### Automatic Startup of Both Scripts

**IMPORTANT:** When you run `docker compose up`, **both** `main.py` and `scripts/stream_server.py` start automatically in a single container. You don't need to manually start the stream server.

Both scripts are managed by the `docker-entrypoint.sh` script, which:
- Starts `main.py` (the main application)
- Starts `scripts/stream_server.py` (the live view web server)
- Manages both processes and ensures graceful shutdown

### CPU Version (Standard)

```bash
# Build and start in detached mode (runs BOTH main.py and stream_server.py)
docker compose up --build -d

# View logs in real-time (shows output from both scripts)
docker compose logs -f

# View logs for specific time period
docker compose logs --since 10m

# Stop the application (gracefully stops both scripts)
docker compose down

# Stop and remove volumes (cleans temporary data)
docker compose down -v
```

### GPU Version

For NVIDIA GPU-accelerated processing (Jetson Nano, Jetson Orin, or desktop GPUs):

```bash
# Build and start with GPU support (runs both scripts)
docker compose -f docker-compose-gpu.yml up --build -d

# View logs
docker compose -f docker-compose-gpu.yml logs -f

# Stop
docker compose -f docker-compose-gpu.yml down
```

### Check Application Status

```bash
# View running container
docker compose ps

# Check health status (verifies both processes are running)
docker inspect biocoder-edge | grep -A 10 Health

# View resource usage
docker stats biocoder-edge

# Verify both processes are running inside the container
docker exec -it biocoder-edge pgrep -f "main.py"
docker exec -it biocoder-edge pgrep -f "stream_server.py"

# View process tree
docker exec -it biocoder-edge ps aux | grep python
```

### Live View Access

Once the container is running, the live view is automatically available at:
```
http://localhost:8080
```

No additional commands needed - both scripts are already running!

### GPU-Specific Commands

For GPU deployments using `docker-compose-gpu.yml`:

```bash
# Check if GPU is accessible inside container
docker exec -it biocoder-edge-gpu nvidia-smi

# Monitor GPU usage continuously
watch -n 1 docker exec biocoder-edge-gpu nvidia-smi

# Verify both processes are running
docker exec -it biocoder-edge-gpu pgrep -f "main.py"
docker exec -it biocoder-edge-gpu pgrep -f "stream_server.py"
```

### Jetson-Specific Configuration

For NVIDIA Jetson devices, you may need to modify the base image in `docker-compose-gpu.yml`:

```yaml
build:
  args:
    # Use Jetson-specific base image
    BASE_IMAGE: nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

## Camera Configuration

### Finding Your Camera

```bash
# List all video devices
ls -la /dev/video*

# Get detailed information about each camera
v4l2-ctl --list-devices

# Test camera with gstreamer (if installed)
gst-launch-1.0 v4l2src device=/dev/video0 ! autovideosink
```

### Multiple Cameras

To use multiple cameras, add them to `docker-compose.yml`:

```yaml
devices:
  - /dev/video0:/dev/video0
  - /dev/video1:/dev/video1
  - /dev/video2:/dev/video2
```

Then configure which camera to use in `config/config.yaml`:
```yaml
motion_detector:
  video_source: 0  # Use /dev/video0
```

### Camera Permissions

If you encounter permission issues:

```bash
# Give camera device world-readable permissions (temporary)
sudo chmod 666 /dev/video0

# Or run container in privileged mode (edit docker-compose.yml)
privileged: true
```

## Accessing Live View

The BioCoder-Edge application includes a live view feature that **starts automatically** with Docker Compose.

### Automatic Live View Startup

When using `docker-compose.yml` or `docker-compose-gpu.yml`, the stream server (`scripts/stream_server.py`) starts automatically via the entrypoint script. **No manual intervention required.**

### Enable Live View in Configuration

To enable the live view feature, edit `config/config.yaml`:

```yaml
live_view:
  enabled: true
```

Then restart the container:
```bash
docker compose restart
```

### Access the Stream

The live view is automatically available at:
```
http://localhost:8080
```

Or from other devices on your network:
```
http://<HOST_IP_ADDRESS>:8080
```

### Verify Stream Server is Running

```bash
# Check if the container is running
docker compose ps

# View all logs (includes stream server output)
docker compose logs -f

# Check if stream server process is active
docker exec -it biocoder-edge pgrep -f "stream_server.py"

# View stream server output specifically
docker exec -it biocoder-edge ps aux | grep stream_server
```

## Troubleshooting

### Container Exits Immediately

```bash
# Check logs for errors
docker compose logs

# Common issues:
# 1. Camera not accessible - check device mapping
# 2. Config file missing - ensure config/config.yaml exists
# 3. Model weights missing - ensure model_weight/best.pt exists
```

### Camera Not Found

```bash
# Verify camera is connected
ls -la /dev/video*

# Check if camera works on host
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg

# Verify device mapping in docker-compose.yml
docker compose config
```

### Permission Denied Errors

```bash
# Run with privileged mode (temporary fix)
# Edit docker-compose.yml:
privileged: true

# Then rebuild:
docker compose up --build -d
```

### Out of Memory Errors

```bash
# Increase shared memory size in docker-compose.yml:
shm_size: '1g'  # Increase from 256m to 1g

# Or reduce RAM frame limit in config.yaml:
animal_analyzer:
  ram_frame_limit: 150  # Reduce from 300
```

### GPU Not Detected

```bash
# Verify NVIDIA runtime is installed
docker info | grep -i runtime

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker exec -it biocoder-edge-gpu nvidia-smi
```

### Application Running But Not Processing

```bash
# Check if all processes are running
docker exec -it biocoder-edge ps aux | grep python

# View detailed logs
docker compose logs -f --tail=100

# Check video source configuration
docker exec -it biocoder-edge cat /app/config/config.yaml | grep video_source
```

## Advanced Usage

### Custom Python Packages

If you need additional Python packages:

1. Edit `requirements.txt`:
   ```
   opencv-python==4.8.1.78
   numpy==1.26.4
   your-package-name
   ```

2. Rebuild the image:
   ```bash
   docker compose build --no-cache
   docker compose up -d
   ```

### Mount Additional Directories

Edit `docker-compose.yml` to add volume mappings:

```yaml
volumes:
  - ./custom_scripts:/app/custom_scripts
  - /path/to/external/storage:/app/external_storage
```

### Environment Variables

Add environment variables in `docker-compose.yml`:

```yaml
environment:
  - TZ=America/Chicago
  - CUSTOM_VAR=value
```

### Running Scripts Inside Container

```bash
# Execute a shell in the running container
docker exec -it biocoder-edge bash

# Run a specific script
docker exec -it biocoder-edge python scripts/view_output.py

# Run with custom arguments
docker exec -it biocoder-edge python main.py --video /path/to/video.mp4
```

### Building Custom Images

```bash
# Build with a specific tag
docker build -t biocoder-edge:v1.0 .

# Build with custom base image
docker build --build-arg BASE_IMAGE=python:3.11-bullseye -t biocoder-edge:custom .
```

### Networking

To access the application from other devices on your network:

1. Find your host IP address:
   ```bash
   ip addr show | grep inet
   ```

2. Configure firewall to allow port 8080:
   ```bash
   sudo ufw allow 8080/tcp
   ```

3. Access from other devices: `http://<HOST_IP>:8080`

### Data Backup

```bash
# Backup configuration and data
tar -czf biocoder-backup-$(date +%Y%m%d).tar.gz \
  config/ data/ model_weight/ logs/

# Restore from backup
tar -xzf biocoder-backup-20241009.tar.gz
```

### Updating the Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Performance Optimization

### For Jetson Devices

```bash
# Set Jetson to maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor temperature and performance
sudo tegrastats
```

### Resource Limits

Add resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
    reservations:
      cpus: '2'
      memory: 2G
```

## Support

For issues and questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review logs: `docker compose logs -f`
- Open an issue on GitHub
- Consult the main README.md for application-specific documentation
