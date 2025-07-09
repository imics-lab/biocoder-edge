# main.py

import yaml
import time
from multiprocessing import Process, Queue
import sys

# Import the main classes from the source directory
from src.motion_detector.detector import MotionDetector
from src.animal_analyzer.analyzer import AnimalAnalyzer
from src.data_uploader.uploader import DataUploader

def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file.
    
    Args:
        config_path (str): The path to the configuration file.
        
    Returns:
        dict: The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def main():
    """
    The main entry point for the BioCoder-Edge application.
    
    This function initializes and starts the three main modules:
    1. MotionDetector: Watches the camera for motion.
    2. AnimalAnalyzer: Analyzes motion events for animals.
    3. DataUploader: Uploads confirmed events to the cloud.
    
    Each module runs in its own independent process to ensure that a slow
    or blocked module does not affect the others.
    """
    print("--- BioCoder-Edge Application Starting ---")

    # 1. Load configuration from the YAML file
    config = load_config()

    # 2. Create the shared queue for communication between the
    #    MotionDetector and the AnimalAnalyzer.
    frame_queue = Queue()

    # 3. Instantiate the main module classes
    try:
        motion_detector = MotionDetector(config)
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