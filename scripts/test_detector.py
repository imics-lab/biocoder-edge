import yaml
import argparse
import sys
import os
from dotenv import load_dotenv
import os
import re

load_dotenv()

# Adjust the Python path to import from the 'src' directory
# Get the directory of this script and add the parent directory (project root) to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.motion_detector.detector import MotionDetector

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
    """
    Main function to test the MotionDetector class on a video file.
    This script instantiates the actual MotionDetector in debug mode.
    """
    parser = argparse.ArgumentParser(
        description="Test the MotionDetector class on a video file with visualization.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help="Path to the input video file to be processed."
    )
    parser.add_argument(
        '--config',
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)."
    )
    args = parser.parse_args()

    print("--- Starting MotionDetector Test Harness ---")

    # --- Load Configuration ---
    config = load_config(args.config)

    try:
        # --- Initialize the MotionDetector in debug mode with the video file as its source ---
        # We pass debug_mode=True to enable the visualization windows.
        # We pass the video path as the source instead of the default camera index.
        detector = MotionDetector(config=config, debug_mode=True, video_source=args.video)

        # --- Run the detector ---
        # Since this test is not part of the main application, we don't need a queue.
        # The detector's start() method will run until the video ends or 'q' is pressed.
        detector.start(shared_queue=None)

    except Exception as e:
        print(f"An error occurred during the test: {e}")

    finally:
        print("\n--- Test Harness Finished ---")

# Usage: python scripts/test_detector.py --video /path/to/video.mp4
if __name__ == "__main__":
    main()