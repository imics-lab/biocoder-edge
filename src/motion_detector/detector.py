import cv2
import time
from multiprocessing import Queue
from typing import Dict

class MotionDetector:
    """
    Detects motion in a video stream and passes event frames to a queue.
    """
    def __init__(self, shared_queue: Queue, config: Dict):
        """
        Initializes the Motion Detector.
        :param shared_queue: The multiprocessing.Queue for outbound frames.
        :param config: A dictionary with operational parameters.
        """
        # ... implementation details ...

    def start(self) -> None:
        """
        Starts the main processing loop of the motion detector.
        This method will block until stop() is called or an error occurs.
        """
        # ... implementation details ...

    def stop(self) -> None:
        """
        Signals the main processing loop to terminate gracefully.
        """
        # ... implementation details ...

    def _processing_loop(self) -> None:
        """
        The private method containing the main while loop for frame processing.
        """
        # ... implementation details ...