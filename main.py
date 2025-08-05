# main.py

import yaml
import time
from multiprocessing import Process, Queue
import sys
import argparse
from dotenv import load_dotenv
import re
import os

load_dotenv()

# Import the main classes from the source directory
from src.motion_detector.detector import MotionDetector
from src.animal_analyzer.analyzer import AnimalAnalyzer
from src.data_uploader.uploader import DataUploader

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
    The main entry point for the BioCoder-Edge application.
    """
    # --- NEW: Add argument parsing ---
    parser = argparse.ArgumentParser(description="Run the BioCoder-Edge application.")
    parser.add_argument(
        '--video',
        type=str,
        default=0,
        help="Path to a video file to use as input. Defaults to camera index 0."
    )
    args = parser.parse_args()
    video_source = args.video if args.video == 0 else str(args.video)

    print("--- BioCoder-Edge Application Starting ---")
    print(f"Using video source: {video_source}")

    # 1. Load configuration from the YAML file
    config = load_config()

    # 2. Create the shared queue for communication between the
    #    MotionDetector and the AnimalAnalyzer.
    frame_queue = Queue()

    # 3. Instantiate the main module classes
    try:
        motion_detector = MotionDetector(config, video_source=video_source)
        animal_analyzer = AnimalAnalyzer(frame_queue, config)
        data_uploader = DataUploader(config)
    except KeyError as e:
        print(f"Error: A required key is missing from the configuration file: {e}")
        sys.exit(1)

    # 4. Create a process for each module.
    #    The 'target' is the 'start' method of each class instance.
    processes = [
        Process(target=motion_detector.start, args=(frame_queue,)),
        Process(target=animal_analyzer.start),
        Process(target=data_uploader.start)
    ]

    try:
        # 5. Start all processes
        print("Starting all modules in separate processes...")
        for p in processes:
            p.start()
        
        # The main process will now wait. The application will run until
        # it is interrupted by the user (e.g., with Ctrl+C).
        # We use a simple loop here to keep the main process alive.
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n--- Shutdown signal received (Ctrl+C) ---")
    
    finally:
        # 7. Graceful shutdown procedure
        print("Initiating graceful shutdown of all modules...")

        # The order of shutdown can matter. It's often best to stop
        # the producer (MotionDetector) first. However, a forceful
        # termination is simpler and generally effective here.
        for p in processes:
            print(f"Terminating process: {p.name} (PID: {p.pid})")
            if p.is_alive():
                p.terminate()  # Sends a SIGTERM signal
                p.join(timeout=5)  # Wait for the process to exit
        
        print("All processes have been terminated.")
        print("--- BioCoder-Edge Application Shut Down ---")

if __name__ == "__main__":
    main()