import cv2
import time
from multiprocessing import Queue
from typing import Dict

class MotionDetector:
    """
    Detects motion in a video stream and passes event frames to a queue.
    This module acts as a computationally cheap "gatekeeper" to trigger
    more intensive analysis.
    """
    def __init__(self, config: Dict):
        """
        Initializes the Motion Detector.
        :param config: A dictionary with operational parameters.
        """
        # Isolate the relevant configuration section
        self.config = config['motion_detector']
        
        # --- Extract configuration parameters for easy access ---
        self.motion_frame_width = self.config['motion_frame_width']
        self.min_area = self.config['min_motion_contour_area']
        self.cooldown = self.config['motion_cooldown_seconds']
        
        # Kernel size for Gaussian blur - must be odd
        self.blur_kernel = tuple(self.config.get('blur_kernel_size', [21,21]))
        
        # 1️⃣ Detect if a CUDA-enabled GPU is present
        self.use_gpu = False
        try:
            self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except AttributeError:
            # OpenCV wasn’t built with CUDA
            self.use_gpu = False

        print(f"GPU accel {'ENABLED' if self.use_gpu else 'NOT available'} – "
              f"{'using GPU pipelines' if self.use_gpu else 'falling back to CPU'}")
        
        # --- Initialize components ---
        print("Initializing camera...")
        self.camera = cv2.VideoCapture("/Users/himsonchapagain/Downloads/Deer.mp4") # Use 0 for the default webcam
        if not self.camera.isOpened():
            raise IOError("Cannot open webcam. Please check camera connection.")
        print("Camera initialized successfully.")

        # Initialize the background subtractor.
        # `detectShadows=True` is crucial for outdoor settings to help filter out shadows.
        # Background subtractor: GPU version if available
        if self.use_gpu:
            self.bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )
        
        self.is_running = False
        self.queue = None


    def start(self, shared_queue: Queue) -> None:
        """
        Starts the main processing loop of the motion detector.
        This method will block until stop() is called or an error occurs.
        :param shared_queue: The multiprocessing.Queue to send frames to.
        """
        if self.is_running:
            print("Motion detector is already running.")
            return
            
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
        
        # If we’ve already set self.queue in start(), send a final sentinel
        if self.queue is not None:
            try:
                self.queue.put(None)
                print("Sent final None sentinel to queue for graceful shutdown.")
            except Exception as e:
                print(f"Failed to send shutdown sentinel: {e}")

    def _processing_loop(self) -> None:
        """
        The private method containing the main while loop for frame processing.
        """
        state = "IDLE"
        last_motion_time = None

        while self.is_running:
            # 1. Read a frame from the camera
            ret, original_frame = self.camera.read()
            if not ret:
                print("Failed to grab frame from camera. Exiting loop.")
                break

            if self.use_gpu:
               # Upload frame
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(original_frame)

                # Resize on GPU
                h, w = original_frame.shape[:2]
                new_w = self.motion_frame_width
                new_h = int(h * (new_w / float(w)))
                gpu_resized = cv2.cuda.resize(gpu_frame, (new_w, new_h))

                # Grayscale + blur on GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY)
                blur_filter = cv2.cuda.createGaussianFilter(
                    gpu_gray.type(), gpu_gray.type(),
                    ksize=self.blur_kernel, sigma1=0
                )
                gpu_blurred = blur_filter.apply(gpu_gray)

                # Background subtraction
                fg_gpu = self.bg_subtractor.apply(gpu_blurred)

                # Threshold + morph on GPU
                _, fg_thresh = cv2.cuda.threshold(fg_gpu, 250, 255, cv2.THRESH_BINARY)
                erode = cv2.cuda.createMorphologyFilter(
                    cv2.MORPH_ERODE, fg_thresh.type(), kernel=None, iterations=2
                )
                dilate = cv2.cuda.createMorphologyFilter(
                    cv2.MORPH_DILATE, fg_thresh.type(), kernel=None, iterations=2
                )
                gpu_eroded = erode.apply(fg_thresh)
                gpu_clean  = dilate.apply(gpu_eroded)

                # Download mask for contour detection
                mask = gpu_clean.download()
            else:
                # CPU fallback
                processed = self._preprocess_frame(original_frame)
                fg_mask   = self.bg_subtractor.apply(processed)
                mask      = self._clean_mask(fg_mask)

            # 5. Find contours and check for significant motion
            motion_found_this_frame = self._find_significant_motion(mask)

            # 6. Implement the state machine logic
            if state == "IDLE":
                if motion_found_this_frame:
                    print("Motion detected! Changing to DETECTING state.")
                    state = "DETECTING"
                    last_motion_time = time.time()
                    # Put the first high-resolution frame into the queue
                    self.queue.put(original_frame)
            
            elif state == "DETECTING":
                if motion_found_this_frame:
                    # If motion continues, update the timer and send the frame
                    last_motion_time = time.time()
                    self.queue.put(original_frame)
                else:
                    # If motion stops, check if the cooldown period has expired
                    if time.time() - last_motion_time > self.cooldown:
                        print(f"Cooldown of {self.cooldown}s expired. Event finished. Changing to IDLE state.")
                        # Send the end-of-event signal and reset state
                        self.queue.put(None)
                        state = "IDLE"
        
        # --- Cleanup ---
        print("Motion detector loop terminated. Releasing resources.")
        self.camera.release()

    def _preprocess_frame(self, frame):
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

    def _clean_mask(self, mask):
        """Applies thresholding and morphological operations to a mask."""
        # Threshold the mask to get a binary image (black and white)
        _, thresh = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        
        # Erode to remove small white noise specks
        eroded = cv2.erode(thresh, None, iterations=2)
        
        # Dilate to close gaps in remaining objects
        dilated = cv2.dilate(eroded, None, iterations=2)
        
        return dilated

    def _find_significant_motion(self, mask) -> bool:
        """Finds contours and checks if any are large enough to be significant."""
        # Find the outlines (contours) of all white objects in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # If the area of a contour is less than our minimum, ignore it
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            # If we find at least one contour that is large enough,
            # we consider this a motion frame and can stop searching.
            return True
            
        # If we loop through all contours and find none are large enough
        return False