#!/bin/bash
# BioCoder-Edge Docker Entrypoint Script
# This script starts both the main application and the stream server

set -e

echo "==================================================================="
echo "BioCoder-Edge Docker Container Starting"
echo "==================================================================="

# Function to handle shutdown gracefully
shutdown() {
    echo ""
    echo "==================================================================="
    echo "Shutdown signal received. Stopping all processes..."
    echo "==================================================================="

    # Kill the stream server if it's running
    if [ -n "$STREAM_PID" ]; then
        echo "Stopping stream server (PID: $STREAM_PID)..."
        kill -TERM "$STREAM_PID" 2>/dev/null || true
        wait "$STREAM_PID" 2>/dev/null || true
    fi

    # Kill the main application if it's running
    if [ -n "$MAIN_PID" ]; then
        echo "Stopping main application (PID: $MAIN_PID)..."
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID" 2>/dev/null || true
    fi

    echo "All processes stopped. Goodbye!"
    exit 0
}

# Trap SIGTERM and SIGINT signals and call shutdown function
trap shutdown SIGTERM SIGINT

# Check if config file exists
if [ ! -f "/app/config/config.yaml" ]; then
    echo "ERROR: Configuration file not found at /app/config/config.yaml"
    echo "Please mount your config file as a volume."
    exit 1
fi

# Check if model weights exist
if [ ! -f "/app/model_weight/best.pt" ]; then
    echo "WARNING: Model weights not found at /app/model_weight/best.pt"
    echo "The application may fail if the YOLO model is not available."
fi

echo ""
echo "Starting BioCoder-Edge Main Application..."
echo "-------------------------------------------------------------------"

# Start the main application in the background
python main.py &
MAIN_PID=$!
echo "Main application started with PID: $MAIN_PID"

# Give the main app a moment to initialize
sleep 3

echo ""
echo "Starting Stream Server..."
echo "-------------------------------------------------------------------"

# Start the stream server in the background
python scripts/stream_server.py &
STREAM_PID=$!
echo "Stream server started with PID: $STREAM_PID"

echo ""
echo "==================================================================="
echo "BioCoder-Edge is now running!"
echo "==================================================================="
echo "Main Application PID: $MAIN_PID"
echo "Stream Server PID:    $STREAM_PID"
echo "Live View URL:        http://localhost:8080"
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop:      docker compose down"
echo "==================================================================="
echo ""

# Wait for both processes
# This keeps the container running and allows for graceful shutdown
wait -n

# If we get here, one of the processes has exited
EXIT_CODE=$?

echo ""
echo "==================================================================="
echo "A process has exited with code: $EXIT_CODE"
echo "==================================================================="

# Check which process exited
if ! kill -0 "$MAIN_PID" 2>/dev/null; then
    echo "Main application has stopped."
fi

if ! kill -0 "$STREAM_PID" 2>/dev/null; then
    echo "Stream server has stopped."
fi

# Trigger shutdown of remaining processes
shutdown
