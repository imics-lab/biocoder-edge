import cv2
import time
import os
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
        
        # --- Live View Configuration ---
        live_view_config = config.get('live_view', {})
        self.live_view_enabled = live_view_config.get('enabled', False)
        if self.live_view_enabled:
            self.live_frame_path = live_view_config.get('ram_disk_path')
            self.lock_file_path = live_view_config.get('lock_file_path')
            self.lock_stale_timeout_seconds = float(live_view_config.get('lock_stale_timeout_seconds', 2.0))
            self.jpeg_quality = int(live_view_config.get('jpeg_quality', 60))
            print(f"Live view enabled. Will write to {self.live_frame_path} when lock file is present.")
        
        # --- Initialize components ---
        self.video_source = video_source
        self.camera = None # Initialize camera as None

        # Initialize the background subtractor.
        # `detectShadows=True` is crucial for outdoor settings to help filter out shadows.
        self.bg_subtractor = None
        self.bg_knn = None
        
        self.is_running = False
        self.queue = None
        self.debug_mode = debug_mode
        self.frame_delay = 0

        # --- ROI (keep only bottom region) ---
        self.roi_bottom_ratio = float(self.config.get("roi_bottom_ratio", 0.65))
        self._roi_mask: Optional[np.ndarray] = None  # set after first preprocess

        # --- Energy gate (EMA background) + freeze window ---
        self.accum_bg: Optional[np.ndarray] = None
        self.accum_alpha = float(self.config.get("accum_alpha", 0.015))
        self.energy_threshold = int(self.config.get("energy_threshold", 10))
        self.hint_area_px = int(self.config.get("hint_area_px", 800))
        self.bg_freeze_frames = 0
        self.bg_freeze_after_hint = int(self.config.get("bg_freeze_after_hint_frames", 120))

        # --- Background models: MOG2 (required) + optional KNN ---
        self.use_knn = bool(self.config.get("use_knn", False))
        if self.use_knn:
            self.bg_knn = cv2.createBackgroundSubtractorKNN(
                history=300, dist2Threshold=400.0, detectShadows=True
            )

        # --- Temporal persistence (debounce + slow-motion rescue) ---
        self.persist_decay = float(self.config.get("persist_decay", 0.90))
        self.persist_thresh = float(self.config.get("persist_thresh", 1.8))
        self._persist_heat: Optional[np.ndarray] = None

        # --- Morphology kernels (prebuilt; avoid per-frame alloc) ---
        self.se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # --- Binary median (integral image) ---
        self.median_kernel = int(self.config.get("binary_median_kernel", 13))



        
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
            
            # --- Live View Frame Writing (On-Demand) ---
            # If the feature is enabled and the lock file exists, it means a viewer
            # is active. Write the latest frame to the RAM disk for consumption.
            should_write_live = False
            if self.live_view_enabled and os.path.exists(self.lock_file_path):
                try:
                    mtime = os.path.getmtime(self.lock_file_path)
                    if (time.time() - mtime) <= self.lock_stale_timeout_seconds:
                        should_write_live = True
                except Exception:
                    should_write_live = False

            if should_write_live:
                # Resize frame to half size horizontally and vertically
                height, width = original_frame.shape[:2]
                resized_frame = cv2.resize(
                    original_frame,
                    (width // 2, height // 2),
                    interpolation=cv2.INTER_AREA,
                )

                # Encode and atomically replace to avoid readers seeing partial writes
                ok, buffer = cv2.imencode(
                    ".jpg",
                    resized_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if ok:
                    tmp_path = f"{self.live_frame_path}.tmp"
                    try:
                        with open(tmp_path, "wb") as tmp_file:
                            tmp_file.write(buffer.tobytes())
                            tmp_file.flush()
                            os.fsync(tmp_file.fileno())
                        os.replace(tmp_path, self.live_frame_path)
                    except Exception:
                        # If atomic replace fails for any reason, best-effort fallback
                        try:
                            cv2.imwrite(
                                self.live_frame_path,
                                resized_frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                            )
                        except Exception:
                            pass
            
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

            # 2 Pre-process (resize → gray → blur)
            processed_frame = self._preprocess_frame(original_frame)

            # Initialize ROI mask if first frame
            if self._roi_mask is None:
                self._init_roi_mask(processed_frame.shape)

            # 3 Energy gate (EMA background) + optional freeze of learning
            energy_mask = self._compute_energy_mask(processed_frame, state)

            # 4 Background subtraction (MOG2 + optional KNN), gated by energy + ROI
            fg_mask0 = self._apply_background_models(processed_frame, state)
            fg_mask0 = (fg_mask0 & energy_mask).astype(np.uint8)

            # 5 Temporal persistence (debounce slow/partial motion)
            stable_u8 = self._apply_persistence(fg_mask0)

            # 6 Clean mask (binary 'median' via integral image + morphology)
            cleaned_mask = self._clean_motion_mask(stable_u8)

            # 7 Contours (area + min w/h + solidity gate)
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
                # When in the DETECTING state, always send the frame to the analyzer.
                if self.queue: self.queue.put(original_frame)

                # If motion is found in the current frame, reset the cooldown timer.
                if motion_found_this_frame:
                    last_motion_time = time.time()
                # If no motion is found, check if the cooldown period has expired.
                else:
                    if time.time() - last_motion_time > self.cooldown:
                        print(f"Cooldown of {self.cooldown}s expired. Event finished. Changing to IDLE state.")
                        # Send the end-of-event signal and reset state.
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

    def _init_roi_mask(self, proc_shape: tuple) -> None:
        """Create a binary ROI mask that keeps bottom roi_bottom_ratio of the frame."""
        h_d, w_d = proc_shape
        self._roi_mask = np.zeros((h_d, w_d), np.uint8)
        top = int((1.0 - self.roi_bottom_ratio) * h_d)
        self._roi_mask[top:, :] = 1

    def _compute_energy_mask(self, processed_frame: np.ndarray, state: str) -> np.ndarray:
        """
        Maintain a slow EMA background and produce a robust 'energy' mask of true scene change.
        Freeze learning for a short window when large changes occur.
        """
        if self.accum_bg is None:
            self.accum_bg = processed_frame.astype(np.float32)

        # Update EMA only when not frozen and IDLE
        if self.bg_freeze_frames == 0 and state == "IDLE":
            cv2.accumulateWeighted(processed_frame.astype(np.float32), self.accum_bg, self.accum_alpha)

        bg_img = self.accum_bg.astype(np.uint8)
        diff_u8 = cv2.absdiff(processed_frame, bg_img)

        # Cheap, robust energy mask in {0,1}
        _, energy01 = cv2.threshold(diff_u8, self.energy_threshold, 1, cv2.THRESH_BINARY)
        energy01 = cv2.medianBlur((energy01 * 255).astype(np.uint8), 3) // 255

        # Constrain to ROI
        energy01 = (energy01 & self._roi_mask).astype(np.uint8)

        # Freeze learning if large area is energized
        if int(np.sum(energy01)) >= self.hint_area_px:
            self.bg_freeze_frames = max(self.bg_freeze_frames, self.bg_freeze_after_hint)

        # Decay the freeze window
        if self.bg_freeze_frames > 0:
            self.bg_freeze_frames -= 1

        return energy01

    def _apply_background_models(self, processed_frame: np.ndarray, state: str) -> np.ndarray:
        """
        Apply MOG2 (+ optional KNN) with a learning rate that respects freeze/DETECTING states.
        Returns a binary {0,1} foreground mask.
        """
        # Learning rate: pause during freeze or while detecting to avoid absorbing objects
        lr_bg = 0.0 if (self.bg_freeze_frames > 0 or state == "DETECTING") else 0.01

        mog2_raw = self.bg_subtractor.apply(processed_frame, learningRate=lr_bg)
        mog2_fg = (mog2_raw >= 200).astype(np.uint8)

        if self.use_knn and self.bg_knn is not None:
            knn_raw = self.bg_knn.apply(processed_frame, learningRate=lr_bg)
            knn_fg = (knn_raw >= 200).astype(np.uint8)
            return np.bitwise_or(mog2_fg, knn_fg).astype(np.uint8)

        return mog2_fg

    def _apply_persistence(self, fg01: np.ndarray) -> np.ndarray:
        """
        Temporal persistence: decays a heat map and keeps motion alive across frames.
        Input/Output are {0,1} → returns {0,255} u8 mask.
        """
        if self._persist_heat is None:
            self._persist_heat = np.zeros_like(fg01, dtype=np.float32)

        self._persist_heat = self._persist_heat * self.persist_decay + fg01.astype(np.float32)
        stable01 = (self._persist_heat >= self.persist_thresh).astype(np.uint8)
        return (stable01 * 255).astype(np.uint8)

    def _clean_motion_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        """
        Clean a binary {0,255} mask:
        1) integral-image 'binary median' (box majority),
        2) close (fill gaps),
        3) open (remove specks).
        """
        k = max(11, self.median_kernel)
        cleaned = self._binary_median_integral(mask_u8, k=k)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.se_close, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.se_open, iterations=1)
        return cleaned

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
        """
        Find contours on a binary {0,255} mask and keep only solid-enough blobs.
        Adds min width/height and solidity checks to drop skinny grass blades.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant = []
        # Minimum dims at detection scale (tune for motion_frame_width)
        min_w = max(8, int(self.motion_frame_width * 0.01))   # ~1% of width
        min_h = max(8, int(self.motion_frame_width * 0.006))  # ~0.6% of width

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w < min_w or h < min_h:
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 1.0:
                continue

            solidity = float(area) / hull_area
            if solidity < 0.45:  # drop ultra-stringy shapes
                continue

            significant.append(c)

        return significant

    def _binary_median_integral(self, mask_u8: np.ndarray, k: int = 11) -> np.ndarray:
        """
        Fast binary 'median' via integral image (box majority).
        Input: {0,255}. Output: {0,255}. Window size k must be odd.
        """
        if k % 2 == 0:
            k += 1

        m = (mask_u8 > 0).astype(np.uint8)  # {0,1}
        h, w = m.shape
        I = cv2.integral(m, sdepth=cv2.CV_32S)  # shape (h+1, w+1)

        r = k // 2
        y0 = np.clip(np.arange(h) - r, 0, h)
        y1 = np.clip(np.arange(h) + r + 1, 0, h)
        x0 = np.clip(np.arange(w) - r, 0, w)
        x1 = np.clip(np.arange(w) + r + 1, 0, w)

        Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
        Y1, X1 = np.meshgrid(y1, x1, indexing="ij")

        S = I[Y1, X1] - I[Y0, X1] - I[Y1, X0] + I[Y0, X0]
        area = (Y1 - Y0) * (X1 - X0)
        out01 = (S * 2 >= area).astype(np.uint8)

        return (out01 * 255).astype(np.uint8)


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