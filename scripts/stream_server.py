import time
import yaml
import argparse
import os
import threading
import atexit
import signal
from flask import Flask, Response, abort
import piexif
from datetime import datetime
import json

# No path adjustments required

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
MAX_VIEWERS = int(live_view_config.get('max_viewers', 1))
STREAM_TIMEOUT_SECONDS = float(live_view_config.get('stream_timeout_seconds', 60))

# Timing constants
FRAME_CHECK_INTERVAL = 0.2
MISSING_FRAME_RETRY = 0.1
SSE_HEARTBEAT_SECONDS = 15.0

def _read_exif_timestamp(jpeg_bytes: bytes):
    """Returns a formatted timestamp from EXIF or None if unavailable."""
    try:
        exif_dict = piexif.load(jpeg_bytes)
        ts_raw = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal]
        ts_str = ts_raw.decode('utf-8', errors='ignore') if isinstance(ts_raw, (bytes, bytearray)) else str(ts_raw)
        dt = datetime.strptime(ts_str, "%Y:%m:%d %H:%M:%S")
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return None

def frame_generator():
    """
    A generator function that reads the latest frame from the RAM disk,
    encodes it as part of an MJPEG stream, and yields it.
    It also manages the viewer lock file.
    """
    global viewer_count
    
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

        last_lock_heartbeat = 0.0
        start_time = time.time()
        while True:
            if time.time() - start_time > STREAM_TIMEOUT_SECONDS:
                print(f"Stream timeout reached ({STREAM_TIMEOUT_SECONDS}s). Stopping MJPEG stream.")
                break

            try:
                # This generator now simply reads the latest frame and streams it.
                # All decoding and rendering is handled by the browser and JavaScript.
                with open(RAM_DISK_PATH, 'rb') as f:
                    frame_bytes = f.read()
                
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    + f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode('ascii')
                    + frame_bytes + b'\r\n'
                )
            except (FileNotFoundError, OSError):
                # If the file doesn't exist, wait briefly and continue.
                time.sleep(MISSING_FRAME_RETRY)
                continue

            # Simple sleep to aim for the target FPS.
            time.sleep(1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.05)

            # Heartbeat: refresh lock file mtime so detector knows viewer is active.
            try:
                now = time.time()
                if now - last_lock_heartbeat >= LOCK_HEARTBEAT_SECONDS:
                    os.utime(LOCK_FILE_PATH, (now, now))
                    last_lock_heartbeat = now
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
    """A simple homepage that displays the video stream and timestamp."""
    return f"""
    <html>
      <head>
        <title>FaunaScope-Edge Live Stream</title>
        <style>
            body {{ 
                font-family: sans-serif; 
                background-color: #1a1a1a; 
                color: white; 
                margin: 0; 
                padding: 0; 
                display: flex; 
                flex-direction: column; 
                align-items: center; 
            }}
            h1 {{ margin-top: 20px; }}
            .stream-container {{
                position: relative;
                max-width: 90%;
                margin-top: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.7);
                background-color: #000;
            }}
            img {{ 
                display: block;
                width: 100%;
                border-radius: 8px;
            }}
            #timestamp {{
                position: absolute;
                top: 10px;
                left: 10px;
                background-color: rgba(0, 0, 0, 0.6);
                color: #fff;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
                text-shadow: 1px 1px 2px #000;
            }}
            #timer {{
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: rgba(220, 53, 69, 0.8);
                color: #fff;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }}
            .overlay {{
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(0,0,0,0.85);
                display: none;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                border-radius: 8px;
                z-index: 10;
                text-align: center;
                padding: 20px;
            }}
            .overlay p {{
                font-size: 18px;
                margin-bottom: 20px;
            }}
            .overlay button {{
                padding: 12px 24px;
                font-size: 16px;
                cursor: pointer;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                transition: background 0.2s;
            }}
            .overlay button:hover {{
                background: #0056b3;
            }}
        </style>
      </head>
      <body>
        <h1>Live Camera Feed</h1>
        <div class="stream-container">
            <img src="/video_feed">
            <div id="timestamp">Loading timestamp...</div>
            <div id="timer"></div>
            <div id="overlay" class="overlay">
                <p>Stream paused after {int(STREAM_TIMEOUT_SECONDS)}s to save data budget.</p>
                <button onclick="window.location.reload()">Refresh Stream</button>
            </div>
        </div>

        <script>
            const tsDiv = document.getElementById('timestamp');
            const timerDiv = document.getElementById('timer');
            const overlay = document.getElementById('overlay');
            const timeoutSeconds = {STREAM_TIMEOUT_SECONDS};
            let timeLeft = Math.floor(timeoutSeconds);

            const updateTimer = () => {{
                if (timeLeft <= 0) {{
                    timerDiv.textContent = 'Expired';
                    overlay.style.display = 'flex';
                    if (window.es) window.es.close();
                    return true;
                }}
                timerDiv.textContent = `Expires in: ${{timeLeft}}s`;
                return false;
            }};

            updateTimer();
            const countdown = setInterval(() => {{
                timeLeft--;
                if (updateTimer()) {{
                    clearInterval(countdown);
                }}
            }}, 1000);

            window.es = new EventSource('/timestamp_stream');
            window.es.onmessage = (e) => {{
                try {{
                    const data = JSON.parse(e.data);
                    if (data.timestamp) {{
                        tsDiv.textContent = data.timestamp;
                    }} else if (data.error) {{
                        tsDiv.textContent = data.error;
                        if (data.error === 'Stream expired') {{
                             overlay.style.display = 'flex';
                             window.es.close();
                        }}
                    }}
                }} catch (_) {{}}
            }};
            window.es.onerror = () => {{
                // Optional: display disconnected state
            }};
        </script>
      </body>
    </html>
    """

@app.route('/timestamp_stream')
def timestamp_stream():
    """Server-Sent Events endpoint that emits a timestamp when the frame changes."""
    def event_stream():
        last_mtime = 0.0
        last_heartbeat = 0.0
        start_time = time.time()
        while True:
            now = time.time()
            if now - start_time > STREAM_TIMEOUT_SECONDS:
                yield f"data: {json.dumps({'error': 'Stream expired'})}\n\n"
                break

            try:
                mtime = os.path.getmtime(RAM_DISK_PATH)
            except OSError:
                mtime = 0.0

            if mtime and mtime != last_mtime:
                try:
                    with open(RAM_DISK_PATH, 'rb') as f:
                        jpeg_data = f.read()
                    formatted = _read_exif_timestamp(jpeg_data)
                    payload = {"timestamp": formatted} if formatted else {"error": "No timestamp in frame"}
                except Exception:
                    payload = {"error": "No timestamp in frame"}
                yield f"data: {json.dumps(payload)}\n\n"
                last_mtime = mtime
                last_heartbeat = now
            elif now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                # Heartbeat to keep the connection alive
                yield ": keep-alive\n\n"
                last_heartbeat = now
            time.sleep(FRAME_CHECK_INTERVAL)

    headers = {
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Content-Type': 'text/event-stream'
    }
    return Response(event_stream(), headers=headers)

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
    parser = argparse.ArgumentParser(description="Live Stream Server for FaunaScope-Edge")
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
