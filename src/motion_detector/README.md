### **Specification for Module A: Motion Detection Pipeline**

#### **1. Overview**

This document outlines the functional requirements and technical specifications for the Motion Detection module (Module A). This module serves as the primary data acquisition and event trigger for the Vision-Based Animal Monitoring System. Its core responsibility is to continuously monitor a video feed with high computational efficiency, identify significant motion events, and stream high-resolution frames of these events to a downstream analysis module (Module B) via a shared data queue.

#### **2. Core Functional Requirements**

*   **Continuous Video Monitoring:** The module must continuously capture frames from the designated video source.
*   **Efficient Motion Detection:** A lightweight motion detection algorithm shall be applied to a down-scaled, grayscale version of the video stream to minimize CPU load and power consumption.
*   **Noise and False Positive Reduction:** The system must filter out insignificant motion caused by environmental factors such as wind, minor shadow changes, and sensor noise.
*   **Event Triggering:** A "motion event" is defined as the detection of sustained, significant motion. Upon triggering, the module shall begin streaming frames.
*   **Event Cooldown:** The module must implement a cooldown period after motion ceases to ensure the entire event is captured, including moments when an animal is temporarily still.
*   **Stateful Operation:** The module will operate in distinct states (e.g., `IDLE`, `DETECTING`) to manage event logic correctly.

#### **3. Interface Contract: Communication with Module B**

This section defines the mandatory interface for data exchange with the downstream analysis module. Adherence to this contract is critical for system integration.

*   **Communication Channel:** The module will receive a `multiprocessing.Queue` object during its initialization. This queue is the sole channel for outbound communication.
*   **Data Transmission Protocol:**
    1.  **Video Frames:** During an active motion event, individual, high-resolution, color video frames shall be placed into the queue.
        *   **Object Type:** `numpy.ndarray`
        *   **Data Type:** `uint8`
        *   **Dimensions:** `(Height, Width, 3)` corresponding to the original camera resolution.
        *   **Color Space:** BGR (the default for OpenCV).
    2.  **End-of-Event Signal:** Upon the conclusion of a motion event (including the cooldown period), a single sentinel value must be placed into the queue to signify the termination of the frame stream for that event.
        *   **Object Type:** The Python `None` object.

*   **Data Flow Example:** A typical event stream on the queue will appear as follows: `[frame_1, frame_2, ..., frame_n, None]`.

#### **4. Technical Implementation Specifications**

*   **Primary Libraries:** `opencv-python`, `numpy`, `multiprocessing`, `time`.
*   **Class Structure:** All logic should be encapsulated within a Python class named `MotionDetector`. This class will manage the video capture device, the background model, and the internal state machine.
*   **Configuration Management:** Parameters must be externalized into a configuration dictionary passed during initialization to allow for tuning without code modification.

    **Required Configuration Parameters:**
    *   `MOTION_FRAME_WIDTH` (int): The width to which frames are resized for motion analysis (e.g., `640`).
    *   `GAUSSIAN_BLUR_KERNEL_SIZE` (tuple): The `(width, height)` of the kernel for the Gaussian blur filter (e.g., `(21, 21)`).
    *   `MIN_MOTION_CONTOUR_AREA` (int): The minimum pixel area of a contour to be considered significant motion (e.g., `500`).
    *   `MOTION_COOLDOWN_SECONDS` (float): The duration in seconds to wait for new motion before declaring an event over (e.g., `10.0`).

#### **5. Processing Pipeline Logic**

The `MotionDetector` class shall implement a main processing loop with the following sequential logic:

1.  **Initialization Phase:**
    *   Instantiate `cv2.VideoCapture` for the camera.
    *   Instantiate the background subtractor model: `cv2.createBackgroundSubtractorMOG2()`. Set `detectShadows=True` to aid in filtering.

2.  **Per-Frame Processing Loop:**
    a. **Acquisition:** Read a frame from the video source (`original_frame`).
    b. **Pre-processing for Analysis:**
        i.  Create a copy of the frame and resize it to the width specified by `MOTION_FRAME_WIDTH`, maintaining aspect ratio.
        ii. Convert the resized frame to grayscale.
        iii. Apply a Gaussian blur using `GAUSSIAN_BLUR_KERNEL_SIZE`.
    c. **Motion Mask Generation:** Apply the background subtractor to the pre-processed frame to generate a `foreground_mask`.
    d. **Mask Cleaning:**
        i.  Apply `cv2.threshold` to the mask to create a binary image.
        ii. Apply morphological opening (`cv2.erode` followed by `cv2.dilate`) to remove noise.
    e. **Contour Analysis:** Use `cv2.findContours` to identify distinct regions of motion.
    f. **Significance Test:** Iterate through the contours. If any contour's area exceeds `MIN_MOTION_CONTOUR_AREA`, a boolean flag `motion_found_this_frame` is set to `True`.

3.  **State Machine Logic:**
    *   **In State `IDLE`:** If `motion_found_this_frame` is `True`:
        *   Transition state to `DETECTING`.
        *   Record the current time as `last_motion_time`.
        *   Place the full-resolution `original_frame` into the shared queue.
    *   **In State `DETECTING`:**
        *   If `motion_found_this_frame` is `True`:
            *   Update `last_motion_time` with the current time.
            *   Place the `original_frame` into the shared queue.
        *   If `motion_found_this_frame` is `False`:
            *   Evaluate if `(current_time - last_motion_time) > MOTION_COOLDOWN_SECONDS`.
            *   If `True`, the event is concluded:
                *   Place `None` into the shared queue.
                *   Transition state back to `IDLE`.

#### **6. Class Interface Definition (Skeleton)**

```python
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
```