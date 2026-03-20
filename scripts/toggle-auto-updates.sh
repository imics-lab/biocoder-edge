#!/bin/bash

# ==============================================================================
# Script Name: toggle-auto-updates.sh
# Description: Enables or disables background system updates (APT and Snap) to 
#              prevent unintended data consumption on metered connections.
#
# Context:     By renaming APT source lists and placing Snap packages on hold, 
#              this script physically prevents the system from querying update 
#              servers, saving background data.
#
# Usage: 
#   sudo ./toggle-auto-updates.sh enable   -> (Restores sources, unholds Snaps)
#   sudo ./toggle-auto-updates.sh disable  -> (Breaks sources, holds Snaps)
# ==============================================================================

# Ensure the script is run as root (required to change system files and snap settings)
if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run this script as root (use sudo)."
  exit 1
fi

MODE=$1
DIR="/etc/apt/sources.list.d"

# These are the third-party repositories we want to control. 
FILES=("cuda-ubuntu2204-arm64.list" "nvidia-l4t-apt-source.list" "docker.list" "tailscale.list")

if [ "$MODE" == "enable" ]; then
    echo "=== Enabling Automatic Updates ==="
    
    # 1. Restore apt sources
    for FILE in "${FILES[@]}"; do
        if [ -f "$DIR/$FILE.bak" ]; then
            mv "$DIR/$FILE.bak" "$DIR/$FILE"
            echo "[OK] Restored $FILE"
        elif [ -f "$DIR/$FILE" ]; then
            echo "[INFO] $FILE is already active."
        else
            echo "[WARNING] $FILE not found."
        fi
    done
    
    # 2. Unhold snap packages
    echo "Unholding Snap packages..."
    snap refresh --unhold
    
    echo "=== Done. You can now run 'apt update' and 'apt upgrade'. ==="

elif [ "$MODE" == "disable" ]; then
    echo "=== Disabling Automatic Updates ==="
    
    # 1. Break apt sources
    for FILE in "${FILES[@]}"; do
        if [ -f "$DIR/$FILE" ]; then
            mv "$DIR/$FILE" "$DIR/$FILE.bak"
            echo "[OK] Disabled $FILE"
        elif [ -f "$DIR/$FILE.bak" ]; then
            echo "[INFO] $FILE is already disabled."
        else
            echo "[WARNING] $FILE not found."
        fi
    done
    
    # 2. Hold snap packages
    echo "Holding Snap packages indefinitely..."
    snap refresh --hold=forever
    
    echo "=== Done. System updates are locked down. ==="

else
    echo "Usage: sudo ./toggle-auto-updates.sh [enable|disable]"
    echo "Example: sudo ./toggle-auto-updates.sh disable"
    exit 1
fi
