import cv2
import time
from multiprocessing import Queue
from typing import Dict, Optional, List
import numpy as np

class MotionDetector:
    """
    Detects motion in a video stream and passes event frames to a queue.
    This module acts as a computationally cheap "gatekeeper" to trigger
    more intensive analysis.
    """
    def __init__(self, config: Dict, debug_mode: bool = False, video_source=0):
        """
        Initializes the Motion Detector.
        :param config: A dictionary with operational parameters.
        :param debug_mode: If True, displays visual feedback windows.
        :param video_source: The camera index (e.g., 0) or a path to a video file.
        """
        # Isolate the relevant configuration section
        self.config = config['motion_detector']
        
        # --- Extract configuration parameters for easy access ---
        self.motion_frame_width = self.config['motion_frame_width']
        self.min_area = self.config['min_motion_contour_area']
        self.cooldown = self.config['motion_cooldown_seconds']
        
        # Kernel size for Gaussian blur - must be odd
        self.blur_kernel = tuple(self.config.get('blur_kernel_size', [21,21]))
        
        # --- Initialize components ---
        self.video_source = video_source
        self.camera = None # Initialize camera as None

        # Initialize the background subtractor.
        # `detectShadows=True` is crucial for outdoor settings to help filter out shadows.
        self.bg_subtractor = None
        
        self.is_running = False
        self.queue = None
        self.debug_mode = debug_mode
        self.frame_delay = 0
        
    def start(self, shared_queue: Optional[Queue] = None) -> None:
        """
        Starts the main processing loop of the motion detector.
        This method will block until stop() is called or an error occurs.
        :param shared_queue: The multiprocessing.Queue to send frames to. Can be None in debug mode.
        """
        if self.is_running:
            print("Motion detector is already running.")
            return
        
         # 2. ADD camera initialization here. This now runs
        #    inside the new process, avoiding the error.
        print("Initializing video source...")
        self.camera = cv2.VideoCapture(self.video_source)
        if not self.camera.isOpened():
            # You can decide how to handle this error. Raising it might
            # not be visible, so printing and returning is safer.
            print(f"FATAL: Cannot open video source: {self.video_source} in child process.")
            return
        
        if isinstance(self.video_source, str):
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.frame_delay = 1 / fps
                print(f"Video file detected. Simulating FPS: {fps:.2f} (Delay: {self.frame_delay:.4f}s)")
            else:
                self.frame_delay = 1 / 30 # Default if FPS is not available
                print(f"Video file detected, but FPS not readable. Defaulting to 30 FPS.")
                
        print("Video source initialized successfully.")
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
            
        self.queue = shared_queue
        self.is_running = True
        print("Starting Motion Detector processing loop...")
        self._processing_loop()


    def stop(self) -> None:
        """
        Signals the main processing loop to terminate gracefully.
        """
        self.is_running = False
        print("Stopping Motion Detector...")


    def _processing_loop(self) -> None:
        """
        The private method containing the main while loop for frame processing.
        """
        state = "IDLE"
        last_motion_time = 0
        consecutive_failures = 0
        max_failures = 5

        while self.is_running:
            # 1. Read a frame from the camera
            ret, original_frame = self.camera.read()
            
            if not ret:
                if isinstance(self.video_source, str):
                    print("End of video file reached. Finalizing event.")
                    # If an event was in progress, send the final signal
                    if self.queue and state == "DETECTING":
                        self.queue.put(None)
                    break # Exit the loop cleanly
                
                consecutive_failures += 1
                print(f"Failed to grab frame (attempt {consecutive_failures})")
            
                if consecutive_failures >= max_failures:
                    print("Maximum consecutive failures reached. Exiting loop.")
                    break
            
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0

            # 2. Pre-process the frame for motion analysis
            processed_frame = self._preprocess_frame(original_frame)

            # 3. Apply background subtraction to get a motion mask
            fg_mask = self.bg_subtractor.apply(processed_frame)

            # 4. Clean the mask to remove noise
            cleaned_mask = self._clean_mask(fg_mask)

            # 5. Find contours and check for significant motion
            significant_contours = self._find_significant_contours(cleaned_mask)
            motion_found_this_frame = len(significant_contours) > 0

            # 6. Implement the state machine logic
            if state == "IDLE":
                if motion_found_this_frame:
                    print("Motion detected! Changing to DETECTING state.")
                    state = "DETECTING"
                    last_motion_time = time.time()
                    # Put the first high-resolution frame into the queue if it exists
                    if self.queue: self.queue.put(original_frame)
            
            elif state == "DETECTING":
                if motion_found_this_frame:
                    # If motion continues, update the timer and send the frame
                    last_motion_time = time.time()
                    if self.queue: self.queue.put(original_frame)
                else:
                    # If motion stops, check if the cooldown period has expired
                    if time.time() - last_motion_time > self.cooldown:
                        print(f"Cooldown of {self.cooldown}s expired. Event finished. Changing to IDLE state.")
                        # Send the end-of-event signal and reset state
                        if self.queue: self.queue.put(None)
                        state = "IDLE"

            if self.frame_delay > 0:
                time.sleep(self.frame_delay)
            
            # If in debug mode, display the visual feedback window
            if self.debug_mode:
                self._show_debug_window(original_frame, cleaned_mask, significant_contours, state, last_motion_time)
                # Allow 'q' to quit the loop when in debug mode
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    self.is_running = False
        
        # --- Cleanup ---
        print("Motion detector loop terminated. Releasing resources.")
        self.camera.release()
        if self.debug_mode:
            cv2.destroyAllWindows()


    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resizes, converts to grayscale, and blurs a frame."""
        # Resize to a consistent width to speed up processing
        h, w, _ = frame.shape
        ratio = self.motion_frame_width / float(w)
        dim = (self.motion_frame_width, int(h * ratio))
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to smooth the image and reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        return blurred


    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Applies thresholding and morphological operations to a mask."""
        # Threshold the mask to get a binary image (black and white)
        # A lower threshold (e.g., 25) is often better for MOG2's shadow detection
        _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
        
        # Create a kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Erode to remove small white noise specks
        eroded = cv2.erode(thresh, kernel, iterations=2)
        
        # Dilate to close gaps in remaining objects
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        
        return dilated


    def _find_significant_contours(self, mask: np.ndarray) -> List:
        """Finds contours and returns a list of those large enough to be significant."""
        # Find the outlines (contours) of all white objects in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_contours = []
        for contour in contours:
            # If the area of a contour is less than our minimum, ignore it
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            significant_contours.append(contour)
            
        return significant_contours


    def _show_debug_window(self, frame: np.ndarray, mask: np.ndarray, contours: List, state: str, last_motion_time: float) -> None:
        """Displays the visual feedback window for debugging."""
        # Draw bounding boxes on the original frame
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Scale coordinates back to the original frame size
            proc_h, proc_w = mask.shape
            orig_h, orig_w, _ = frame.shape
            scale_x, scale_y = orig_w / proc_w, orig_h / proc_h
            
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            orig_w = int(w * scale_x)
            orig_h = int(h * scale_y)
            
            cv2.rectangle(frame, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (0, 255, 0), 2)

        # Prepare text for display
        state_text = f"State: {state}"
        color = (0, 255, 0) if state == "DETECTING" else (0, 0, 255)
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if state == "DETECTING":
            cooldown_remaining = max(0, self.cooldown - (time.time() - last_motion_time))
            cooldown_text = f"Cooldown: {cooldown_remaining:.1f}s"
            cv2.putText(frame, cooldown_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Prepare mask for display (convert to 3-channel BGR)
        display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_mask, "Cleaned Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Resize mask to match original frame height for stacking
        h1, _, _ = frame.shape
        h2, w2, _ = display_mask.shape
        display_mask_resized = cv2.resize(display_mask, (int(w2 * h1/h2), h1))
        
        # Combine frames for a side-by-side view
        combined_view = np.hstack((frame, display_mask_resized))
        
        # Resize the combined view to fit better on screen
        # Scale down to 50% of original size - adjust this value if needed
        scale_factor = 0.5
        height, width = combined_view.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_view = cv2.resize(combined_view, (new_width, new_height))
        
        cv2.imshow("Motion Detector Test", resized_view)