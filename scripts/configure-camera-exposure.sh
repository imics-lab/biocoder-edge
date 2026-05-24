#!/bin/bash

# ==============================================================================
# Script Name: configure-camera-exposure.sh
# Description: Configures UVC camera exposure controls for stable frame rate.
#
# Usage:
#   ./scripts/configure-camera-exposure.sh
#   ./scripts/configure-camera-exposure.sh /dev/video0 100
#
# Arguments:
#   $1: Camera device path. Defaults to VIDEO_SOURCE from .env, or /dev/video0.
#   $2: Exposure time. Defaults to CAMERA_EXPOSURE_TIME from .env, or 100.
# ==============================================================================

set -euo pipefail

if ! command -v v4l2-ctl >/dev/null 2>&1; then
    echo "Error: v4l2-ctl is not installed. Install it with: sudo apt install v4l-utils"
    exit 1
fi

ENV_FILE=".env"
VIDEO_SOURCE_VALUE=""
EXPOSURE_TIME_VALUE=""

if [ -f "$ENV_FILE" ]; then
    VIDEO_SOURCE_VALUE=$(grep -E '^VIDEO_SOURCE=' "$ENV_FILE" | tail -n 1 | cut -d= -f2-)
    EXPOSURE_TIME_VALUE=$(grep -E '^CAMERA_EXPOSURE_TIME=' "$ENV_FILE" | tail -n 1 | cut -d= -f2-)
fi

DEVICE="${1:-}"
if [ -z "$DEVICE" ]; then
    if [ -n "$VIDEO_SOURCE_VALUE" ]; then
        if [[ "$VIDEO_SOURCE_VALUE" =~ ^[0-9]+$ ]]; then
            DEVICE="/dev/video$VIDEO_SOURCE_VALUE"
        else
            DEVICE="$VIDEO_SOURCE_VALUE"
        fi
    else
        DEVICE="/dev/video0"
    fi
fi

EXPOSURE_TIME="${2:-${EXPOSURE_TIME_VALUE:-100}}"

if [ ! -e "$DEVICE" ]; then
    echo "Error: Camera device not found: $DEVICE"
    exit 1
fi

echo "Configuring camera exposure controls for $DEVICE"
echo "Using exposure_time_absolute=$EXPOSURE_TIME"

v4l2-ctl -d "$DEVICE" -c exposure_dynamic_framerate=0
v4l2-ctl -d "$DEVICE" -c auto_exposure=1
v4l2-ctl -d "$DEVICE" -c exposure_time_absolute="$EXPOSURE_TIME"

echo ""
echo "Camera controls after configuration:"
v4l2-ctl -d "$DEVICE" --list-ctrls | grep -E 'auto_exposure|exposure_time_absolute|exposure_dynamic_framerate'

echo ""
echo "Streaming parameters:"
v4l2-ctl -d "$DEVICE" --get-parm
