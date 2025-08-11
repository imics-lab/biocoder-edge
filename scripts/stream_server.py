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

        # adaptive sleep for network resilience
        frame_interval = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 1.0 / 15.0
        min_interval = 1.0 / MIN_FPS if MIN_FPS > 0 else 1.0 / 5.0

        last_sent_time = 0.0

        while True:
            frame_bytes = None
            try:
                # Read the latest frame from the RAM disk
                with open(RAM_DISK_PATH, 'rb') as f:
                    data = f.read()
                    # Simple JPEG SOI/EOI sanity check to avoid partial reads
                    if len(data) > 3 and data[:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9':
                        frame_bytes = data
            except FileNotFoundError:
                pass

            # Fallback to last good frame to avoid gaps
            if frame_bytes is None and last_frame_bytes is not None:
                frame_bytes = last_frame_bytes

            if frame_bytes is not None:
                last_frame_bytes = frame_bytes
                headers = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    + f'Content-Length: {len(frame_bytes)}\r\n'.encode('ascii')
                    + b'\r\n'
                )
                yield headers + frame_bytes + b'\r\n'
            else:
                # No new frame, back off more aggressively
                time.sleep(max(frame_interval, 0.1))

            # Adaptive pacing: if send loop lags (client/network slow), increase interval up to min_fps
            now = time.time()
            elapsed = now - last_sent_time if last_sent_time else frame_interval
            last_sent_time = now
            if elapsed > frame_interval * 2 and frame_interval < min_interval:
                frame_interval = min(min_interval, frame_interval * 1.5)
            elif elapsed < frame_interval * 0.75 and frame_interval > (1.0 / 60.0):
                frame_interval = max(1.0 / 60.0, frame_interval / 1.2)

            time.sleep(frame_interval)

            # Heartbeat: refresh lock file mtime so detector knows viewer is active
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