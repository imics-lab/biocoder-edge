import cv2
import time
import yaml
import argparse
import sys
import os

# Adjust Python path to import from the root directory
sys.path.append('..')

def load_config(config_path="config/config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function to view the live frame from the RAM disk."""
    parser = argparse.ArgumentParser(description="Local Live Viewer for BioCoder-Edge")
    parser.add_argument('--config', type=str, default="config/config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    live_view_config = config.get('live_view', {})
    ram_disk_path = live_view_config.get('ram_disk_path', '/dev/shm/live_frame.jpg')

    print(f"Attempting to view stream from: {ram_disk_path}")
    print("Press 'q' in the window to quit.")
    
    # Create the lock file to signal the MotionDetector to start writing frames.
    # This is necessary because this script is also a "viewer".
    lock_file_path = live_view_config.get('lock_file_path', '/dev/shm/viewer_active.lock')
    with open(lock_file_path, 'w') as f: pass
    
    try:
        while True:
            if os.path.exists(ram_disk_path):
                try:
                    frame = cv2.imread(ram_disk_path)
                    if frame is not None:
                        cv2.imshow("BioCoder-Edge Local Live View", frame)
                except Exception as e:
                    print(f"Could not read or display frame: {e}")
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
    finally:
        # Clean up the lock file and close windows on exit
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
        cv2.destroyAllWindows()
        print("Live viewer stopped.")

if __name__ == "__main__":
    main()