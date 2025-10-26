import cv2
import time
import yaml
import argparse
import sys
import os
import threading
import atexit
import signal
from flask import Flask, Response, abort
import piexif
import numpy as np
from datetime import datetime

# Adjust Python path to import from the root directory
sys.path.append('..')

# Global variable to track the number of connected viewers
viewer_count = 0
viewer_lock = threading.Lock()

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
LOCK_HEARTBEAT_SECONDS = float(live_view_config.get('lock_heartbeat_seconds', 0.5))
TARGET_FPS = float(live_view_config.get('target_fps', 15))
MIN_FPS = float(live_view_config.get('min_fps', 5))
MAX_VIEWERS = int(live_view_config.get('max_viewers', 1))

def frame_generator():
    """
    A generator function that reads the latest frame from the RAM disk,
    encodes it as part of an MJPEG stream, and yields it.
    It also manages the viewer lock file.
    """
    global viewer_count
    last_frame_bytes = None

    try:
        # On first connection, create the lock file
        with viewer_lock:
            if MAX_VIEWERS > 0 and viewer_count >= MAX_VIEWERS:
                raise RuntimeError('Max viewers reached')
            if viewer_count == 0:
                print("First viewer connected. Creating lock file.")
                try:
                    with open(LOCK_FILE_PATH, 'a'):
                        pass
                except Exception:
                    pass
            viewer_count += 1

        while True:
            frame_bytes = None
            ts_text = "No timestamp available"
            
            try:
                # Read the latest frame from the RAM disk.
                with open(RAM_DISK_PATH, 'rb') as f:
                    jpeg_data = f.read()

                # Extract timestamp from EXIF data.
                try:
                    exif_dict = piexif.load(jpeg_data)
                    timestamp_str = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("utf-8")
                    dt_object = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
                    ts_text = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                except (KeyError, ValueError, piexif.InvalidImageDataError):
                    # Handle cases where EXIF data is missing or corrupt.
                    pass 

                # Decode the image to draw the timestamp on it.
                img_np = np.frombuffer(jpeg_data, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Add a semi-transparent background for the text for better readability.
                    (text_width, text_height), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (5, 5), (10 + text_width, 10 + text_height + 5), (0, 0, 0), -1)
                    alpha = 0.6  # Transparency factor.
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    
                    # Put the timestamp text on the image.
                    cv2.putText(img, ts_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Re-encode the image to JPEG format for streaming.
                    ok, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok:
                        frame_bytes = buffer.tobytes()

            except (FileNotFoundError, OSError):
                # If the file doesn't exist, wait briefly.
                time.sleep(0.05)
                pass

            # Fallback to the last successfully processed frame to avoid stream gaps.
            if frame_bytes is None and last_frame_bytes is not None:
                frame_bytes = last_frame_bytes

            if frame_bytes is not None:
                last_frame_bytes = frame_bytes
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    + f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode('ascii')
                    + frame_bytes + b'\r\n'
                )
            else:
                # If there's no frame at all, wait before trying again.
                time.sleep(0.1)

            # Simple sleep to aim for the target FPS.
            time.sleep(1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.05)

            # Heartbeat: refresh lock file mtime so detector knows viewer is active.
            try:
                now = time.time()
                os.utime(LOCK_FILE_PATH, (now, now))
            except Exception:
                pass

    finally:
        # On last disconnection, remove the lock file
        with viewer_lock:
            viewer_count -= 1
            if viewer_count == 0:
                print("Last viewer disconnected. Removing lock file.")
                try:
                    if os.path.exists(LOCK_FILE_PATH):
                        os.remove(LOCK_FILE_PATH)
                except Exception:
                    pass

@app.route('/')
def index():
    """A simple homepage that displays the video stream."""
    return f"""
    <html>
      <head>
        <title>BioCoder-Edge Live Stream</title>
        <style>
            body {{ font-family: sans-serif; background-color: #282c34; color: white; margin: 0; padding: 0; display: flex; flex-direction: column; align-items: center; }}
            h1 {{ margin-top: 20px; }}
            img {{ max-width: 90%; margin-top: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }}
        </style>
      </head>
      <body>
        <h1>Live Camera Feed</h1>
        <img src="/video_feed">
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """The video streaming route."""
    # Enforce viewer limit at connection time
    with viewer_lock:
        if MAX_VIEWERS > 0 and viewer_count >= MAX_VIEWERS:
            abort(503, description='Max viewers reached')

    resp = Response(frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Stream Server for BioCoder-Edge")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="Host to bind to (0.0.0.0 for all interfaces)")
    parser.add_argument('--port', type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()
    
    # Graceful shutdown: remove lock on exit and handle signals
    def _cleanup(*_):
        try:
            with viewer_lock:
                if os.path.exists(LOCK_FILE_PATH) and viewer_count == 0:
                    os.remove(LOCK_FILE_PATH)
        except Exception:
            pass
    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), os._exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), os._exit(0)))

    print(f"Starting stream server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
