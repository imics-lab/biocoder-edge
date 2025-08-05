import cv2
import time
import yaml
import argparse
import sys
import os
from dotenv import load_dotenv
import re

load_dotenv()

# Adjust Python path to import from the root directory
sys.path.append('..')

def load_config(config_path="config/config.yaml"):
    """
    Loads a YAML config file, manually expanding environment variables.
    """
    # Pattern to find all ${VAR_NAME} occurrences
    pattern = re.compile(r'\$\{(\w+)\}')
    try:
        with open(config_path, 'r') as f:
            raw_config_string = f.read()

        # This function looks up the env var for each match
        def replace_with_env(match):
            var_name = match.group(1)
            return os.environ.get(var_name, f'${{{var_name}}}') # Keep placeholder if var not found

        # Substitute all placeholders
        populated_config_string = pattern.sub(replace_with_env, raw_config_string)

        # Load the now-populated string as YAML
        return yaml.safe_load(populated_config_string)
        
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        print("Please ensure you are running this script from the project's root directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)
        
        
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