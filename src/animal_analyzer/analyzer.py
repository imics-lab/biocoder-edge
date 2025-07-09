import cv2
import time
import json
import os
import subprocess
import uuid
from multiprocessing import Queue
from typing import Dict, List

# It's assumed a YOLO library like 'ultralytics' is installed.
# from ultralytics import YOLO

class AnimalAnalyzer:
    """
    Analyzes video frame events from a queue, detects animals using YOLO,
    and packages significant events as video and JSON metadata files.
    """

    def __init__(self, shared_queue: Queue, config: Dict):
        """
        Initializes the Animal Analyzer.
        :param shared_queue: The multiprocessing.Queue to receive frames from.
        :param config: A dictionary with operational parameters.
        """
        self.queue = shared_queue
        self.config = config
        self.is_running = False

        # --- Initialize YOLO Model ---
        # self.model = YOLO(self.config['YOLO_MODEL_PATH'])
        print("YOLO model loaded (placeholder).")

        # --- Ensure Directories Exist ---
        os.makedirs(self.config['OUTPUT_PENDING_DIR'], exist_ok=True)
        os.makedirs(self.config['OUTPUT_TEMP_DIR'], exist_ok=True)
        print(f"Output directories ensured at {self.config['OUTPUT_PENDING_DIR']} and {self.config['OUTPUT_TEMP_DIR']}")


    def start(self) -> None:
        """Starts the main event processing loop."""
        if self.is_running:
            print("Animal analyzer is already running.")
            return
        self.is_running = True
        print("Starting Animal Analyzer...")
        self._event_loop()


    def stop(self) -> None:
        """Signals the main processing loop to terminate."""
        self.is_running = False
        print("Stopping Animal Analyzer...")


    def _event_loop(self) -> None:
        """The main loop that waits for and processes whole events."""
        while self.is_running:
            print("Waiting for a new event...")
            # This line blocks until the first frame of an event arrives.
            first_frame = self.queue.get()

            if first_frame is None: # Could happen if module A stops gracefully
                continue
            
            # --- A NEW EVENT HAS STARTED ---
            event_id = f"event_{time.strftime('%Y%m%d-%H%M%S')}_{str(uuid.uuid4())[:8]}"
            print(f"New event received: {event_id}")

            ram_buffer = [first_frame]
            temp_file_parts = []
            tracked_objects = {}
            all_confirmed_detections = []
            significant_event_detected = False
            
            # --- Process all frames for this event ---
            while True:
                frame = self.queue.get()
                if frame is None:
                    # End-of-event signal received
                    break
                
                ram_buffer.append(frame)
                
                # 1. RUN YOLO INFERENCE
                # results = self.model(frame, verbose=False)
                # current_detections = self._parse_yolo_results(results)
                current_detections = [] # Placeholder

                # 2. UPDATE OBJECT TRACKING
                # tracked_objects = self._update_tracker(tracked_objects, current_detections)
                
                # 3. CHECK FOR EVENT SIGNIFICANCE & GATHER DETECTIONS
                # for obj_id, obj_data in tracked_objects.items():
                #     if obj_data['label'] in self.config['SPECIES_OF_INTEREST']:
                #         if obj_data['frames_seen'] > self.config['EVENT_CONFIRMATION_FRAMES']:
                #             significant_event_detected = True
                #             all_confirmed_detections.append(...) # Add detection details
                
                # 4. HANDLE MEMORY MANAGEMENT (SPILL TO DISK)
                if len(ram_buffer) >= self.config['RAM_FRAME_LIMIT']:
                    print(f"RAM limit reached. Spilling {len(ram_buffer)} frames to disk...")
                    temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                    temp_file_parts.append(temp_path)
                    ram_buffer.clear()
            
            # --- EVENT FINALIZATION ---
            print(f"Event {event_id} finished. Finalizing...")

            # Write any remaining frames in RAM to the last part
            if ram_buffer:
                temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                temp_file_parts.append(temp_path)
                ram_buffer.clear()

            if significant_event_detected and temp_file_parts:
                print(f"Significant event! Packaging video and metadata for {event_id}.")
                # 5. FINALIZE VIDEO (CONCATENATE PARTS)
                final_video_path = self._concatenate_parts(event_id, temp_file_parts)
                
                # 6. GENERATE AND SAVE JSON METADATA
                self._create_json_metadata(event_id, final_video_path, all_confirmed_detections)
            else:
                print(f"Insignificant event or no frames. Cleaning up temporary files for {event_id}.")
                self._cleanup_temp_files(temp_file_parts)


    def _write_video_part(self, event_id: str, frames: List, part_num: int) -> str:
        """Encodes a list of frames into a temporary MP4 video file."""
        if not frames:
            return ""
        
        height, width, _ = frames[0].shape
        temp_path = os.path.join(self.config['OUTPUT_TEMP_DIR'], f"{event_id}_part_{part_num}.mp4")
        
        # Use a reliable video codec like 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 15.0, (width, height)) # Assume 15 FPS
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"  > Wrote {temp_path}")
        return temp_path


    def _concatenate_parts(self, event_id: str, temp_parts: List[str]) -> str:
        """Uses FFmpeg to losslessly concatenate video parts into a final video."""
        final_video_path = os.path.join(self.config['OUTPUT_PENDING_DIR'], f"{event_id}.mp4")
        
        # Create a file list for FFmpeg
        file_list_path = os.path.join(self.config['OUTPUT_TEMP_DIR'], f"{event_id}_filelist.txt")
        with open(file_list_path, 'w') as f:
            for path in temp_parts:
                f.write(f"file '{path}'\n")

        # FFmpeg command for safe concatenation
        # -y: overwrite output file if it exists
        # -f concat: specify the format as concatenation
        # -safe 0: needed to allow absolute paths in the file list
        # -c copy: copies the stream without re-encoding (fast and lossless)
        command = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', file_list_path, 
            '-c', 'copy', final_video_path
        ]
        
        print(f"Running FFmpeg to create {final_video_path}...")
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Clean up the temp files and the list file
        self._cleanup_temp_files(temp_parts)
        os.remove(file_list_path)

        return final_video_path
    
    
    def _create_json_metadata(self, event_id: str, video_path: str, detections: List) -> None:
        """Creates and saves the JSON metadata file for an event."""
        metadata = {
            "eventId": event_id,
            "deviceId": "jetson_orin_001", # Placeholder
            "location": { "latitude": 0.0, "longitude": 0.0 }, # Placeholder
            "timestamp_start_utc": "...", # Placeholder
            "timestamp_end_utc": "...", # Placeholder
            "local_video_path": video_path,
            "video_duration_seconds": 0.0, # Placeholder
            "event_summary": {}, # To be filled from analysis
            "detections": detections
        }
        
        json_path = os.path.join(self.config['OUTPUT_PENDING_DIR'], f"{event_id}.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  > Wrote JSON metadata to {json_path}")

    
    def _cleanup_temp_files(self, file_paths: List[str]) -> None:
        """Deletes a list of temporary files."""
        for path in file_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error deleting temp file {path}: {e}")

    # --- TODO: Implement these core logic functions ---
    def _parse_yolo_results(self, results):
        """Parses the output from the YOLO model into a standardized format."""
        pass

    def _update_tracker(self, tracked_objects, current_detections):
        """Updates the state of tracked objects based on new detections."""
        pass