import cv2
import time
import yaml
import argparse
import sys
import os
from flask import Flask, Response

# Adjust Python path to import from the root directory
sys.path.append('..')

# Global variable to track the number of connected viewers
viewer_count = 0

def load_config(config_path="config/config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Flask App Initialization ---
app = Flask(__name__)

# Load configuration once when the server starts
config = load_config()
live_view_config = config.get('live_view', {})
RAM_DISK_PATH = live_view_config.get('ram_disk_path', '/dev/shm/live_frame.jpg')
LOCK_FILE_PATH = live_view_config.get('lock_file_path', '/dev/shm/viewer_active.lock')

def frame_generator():
    """
    A generator function that reads the latest frame from the RAM disk,
    encodes it as part of an MJPEG stream, and yields it.
    It also manages the viewer lock file.
    """
    global viewer_count
    
    try:
        # --- On first connection, create the lock file ---
        if viewer_count == 0:
            print("First viewer connected. Creating lock file.")
            with open(LOCK_FILE_PATH, 'w') as f:
                pass # Create an empty file
        viewer_count += 1
        
        while True:
            try:
                # Read the latest frame from the RAM disk
                with open(RAM_DISK_PATH, 'rb') as f:
                    frame_bytes = f.read()
                
                # Yield the frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except FileNotFoundError:
                # If the frame file doesn't exist yet, wait a moment and try again
                time.sleep(0.1)
                continue
            
            # Control the streaming frame rate
            time.sleep(1/30) # Stream at ~30 FPS

    finally:
        # --- On last disconnection, remove the lock file ---
        viewer_count -= 1
        if viewer_count == 0:
            print("Last viewer disconnected. Removing lock file.")
            if os.path.exists(LOCK_FILE_PATH):
                os.remove(LOCK_FILE_PATH)

@app.route('/')
def index():
    """A simple homepage that displays the video stream."""
    return f"""
    <html>
      <head><title>BioCoder-Edge Live Stream</title></head>
      <body>
        <h1>Live Camera Feed</h1>
        <img src="/video_feed" width="480">
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """The video streaming route."""
    return Response(frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Stream Server for BioCoder-Edge")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="Host to bind to (0.0.0.0 for all interfaces)")
    parser.add_argument('--port', type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()
    
    print(f"Starting stream server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)