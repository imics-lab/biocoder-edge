import cv2
import time
import json
import os
import subprocess
import uuid
import numpy as np
from multiprocessing import Queue
from typing import Dict, List, Tuple
from scipy.spatial import distance as dist
from ultralytics import YOLO
from collections import Counter

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
        # We only need the 'animal_analyzer' part of the config
        self.config = config['animal_analyzer']
        self.device_id = self.config.get('device_id', 'unknown_device')
        self.location  = self.config.get('location', {'latitude':0.0,'longitude':0.0})
        self.output_fps = self.config.get('output_fps', 30.0) 
        self.is_running = False

        # --- Initialize YOLO Model ---
        print("Loading YOLO model...")
        self.model = YOLO(self.config['yolo_model_path'])
        print("YOLO model loaded successfully.")

        # --- Ensure Directories Exist ---
        os.makedirs(self.config['output_pending_dir'], exist_ok=True)
        os.makedirs(self.config['output_temp_dir'], exist_ok=True)
        print(f"Output directories ensured at {self.config['output_pending_dir']} and {self.config['output_temp_dir']}")


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
            first_frame = self.queue.get()

            if first_frame is None:
                continue
            
            # --- A NEW EVENT HAS STARTED ---
            event_id = f"event_{time.strftime('%Y%m%d-%H%M%S')}_{str(uuid.uuid4())[:8]}"
            event_start_time = time.time()
            print(f"New event received: {event_id}")

            ram_buffer = [first_frame]
            temp_file_parts = []
            tracked_objects = {}
            next_object_id = 0
            all_confirmed_detections = []
            significant_event_detected = False
            frame_index = 0
            
            # --- Process all frames for this event ---
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                
                ram_buffer.append(frame)
                frame_index += 1
                
                # 1. RUN YOLO INFERENCE AND PARSE RESULTS
                results = self.model(frame, verbose=False)
                current_detections = self._parse_yolo_results(results)

                # 2. UPDATE OBJECT TRACKING
                tracked_objects, next_object_id = self._update_tracker(
                    tracked_objects, current_detections, next_object_id
                )
                
                # 3. CHECK FOR EVENT SIGNIFICANCE & GATHER DETECTIONS
                for obj_id, obj_data in tracked_objects.items():
                    # Check if this object just became significant
                    if (obj_data['label'] in self.config['species_of_interest'] and
                        obj_data['frames_seen'] == self.config['event_confirmation_frames']):
                        print(f"  > Confirmed significant object: {obj_data['label']} (ID: {obj_id})")
                        significant_event_detected = True

                    # If the event is significant, log all new detections of tracked, interesting species
                    if significant_event_detected and obj_data['label'] in self.config['species_of_interest'] and obj_data['just_seen']:
                        cx, cy = obj_data['centroid']
                        w,  h  = obj_data['last_box_xywh'][2], obj_data['last_box_xywh'][3]
                        all_confirmed_detections.append({
                            "frame_index": frame_index,
                            "label": obj_data['label'],
                            "confidence": float(obj_data['last_confidence']),
                            "box_xywh": [cx, cy, int(w), int(h)]
                        })

                # 4. HANDLE MEMORY MANAGEMENT (SPILL TO DISK)
                if len(ram_buffer) >= self.config['ram_frame_limit']:
                    print(f"RAM limit reached. Spilling {len(ram_buffer)} frames to disk...")
                    temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                    if temp_path:
                        temp_file_parts.append(temp_path)
                    ram_buffer.clear()
            
            # --- EVENT FINALIZATION ---
            event_end_time = time.time()
            print(f"Event {event_id} finished. Finalizing...")

            if significant_event_detected and ram_buffer:
                temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                if temp_path:
                    temp_file_parts.append(temp_path)
            ram_buffer.clear()

            if significant_event_detected and temp_file_parts:
                print(f"Significant event! Packaging video and metadata for {event_id}.")
                final_video_path = self._concatenate_parts(event_id, temp_file_parts)
                if final_video_path:
                    actual_duration = self._get_video_duration(final_video_path)
                    self._create_json_metadata(event_id, final_video_path, all_confirmed_detections, event_start_time, event_end_time, actual_duration)
            else:
                print(f"Insignificant event or no frames. Cleaning up temporary files for {event_id}.")
                self._cleanup_temp_files(temp_file_parts)
    
    def _get_video_duration(self, video_path: str) -> float:
        """Calculates the precise duration of a video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps > 0:
                return frame_count / fps
            return 0.0
        except Exception as e:
            print(f"Could not calculate video duration for {video_path}: {e}")
            return 0.0

    def _parse_yolo_results(self, results) -> List[Dict]:
        """Parses the output from the YOLO model into a standardized list of dicts."""
        detections = []
        if results and results[0]:
            boxes = results[0].boxes
            for box in boxes:
                if box.conf[0] >= self.config['confidence_threshold']:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    cx, cy = int(x1 + w/2), int(y1 + h/2)
                    detections.append({
                        'label': self.model.names[int(box.cls[0])],
                        'confidence': box.conf[0].cpu().numpy(),
                        'box_xywh':  (cx, cy, int(w), int(h)),
                        'centroid': (cx, cy)
                    })
        return detections

    def _update_tracker(self, tracked_objects: Dict, current_detections: List, next_object_id: int) -> Tuple[Dict, int]:
        """Updates the state of tracked objects using a simple centroid-based algorithm."""
        
        # If there are no existing objects, register all new detections
        if len(tracked_objects) == 0:
            for det in current_detections:
                tracked_objects[next_object_id] = {
                    'id': next_object_id, 'label': det['label'], 'centroid': det['centroid'],
                    'frames_seen': 1, 'frames_since_seen': 0, 'just_seen': True,
                    'last_confidence': det['confidence'], 'last_box_xywh':  det['box_xywh'],
                }
                next_object_id += 1
            return tracked_objects, next_object_id

        # Prepare lists of existing and new centroids
        existing_obj_ids = list(tracked_objects.keys())
        existing_centroids = [tracked_objects[oid]['centroid'] for oid in existing_obj_ids]
        new_centroids = [det['centroid'] for det in current_detections]

        # Mark all objects as not seen in this frame initially
        for oid in tracked_objects:
            tracked_objects[oid]['just_seen'] = False

        # If no new detections, just increment disappearance counter for all existing objects
        if len(current_detections) == 0:
            for oid in tracked_objects:
                tracked_objects[oid]['frames_since_seen'] += 1
        # Otherwise, perform matching
        else:
            # Calculate the distance between all pairs of existing and new centroids
            D = dist.cdist(np.array(existing_centroids), np.array(new_centroids))

            # Find the best match for each existing object (smallest distance)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # If distance is too large, it's not the same object
                if D[row, col] > self.config.get('tracking_max_distance', 75): # Use a default if not in config
                    continue

                # It's a match! Update the tracked object.
                object_id = existing_obj_ids[row]
                tracked_objects[object_id]['centroid'] = new_centroids[col]
                tracked_objects[object_id]['frames_seen'] += 1
                tracked_objects[object_id]['frames_since_seen'] = 0
                tracked_objects[object_id]['just_seen'] = True
                tracked_objects[object_id]['last_confidence'] = current_detections[col]['confidence']
                tracked_objects[object_id]['last_box_xywh'] = current_detections[col]['box_xywh']
                
                used_rows.add(row)
                used_cols.add(col)

            # Handle objects that were not matched
            unmatched_rows = set(range(len(existing_obj_ids))) - used_rows
            unmatched_cols = set(range(len(new_centroids))) - used_cols

            for row in unmatched_rows:
                object_id = existing_obj_ids[row]
                tracked_objects[object_id]['frames_since_seen'] += 1

            for col in unmatched_cols:
                det = current_detections[col]
                tracked_objects[next_object_id] = {
                    'id': next_object_id, 'label': det['label'], 'centroid': det['centroid'],
                    'frames_seen': 1, 'frames_since_seen': 0, 'just_seen': True,
                    'last_confidence': det['confidence'], 'last_box_xywh': det['box_xywh']
                }
                next_object_id += 1

        # Prune stale tracks that have been gone for too long
        final_tracked_objects = {}
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['frames_since_seen'] <= self.config['tracking_inactivity_frames']:
                final_tracked_objects[obj_id] = obj_data

        return final_tracked_objects, next_object_id

    def _write_video_part(self, event_id: str, frames: List, part_num: int) -> str:
        if not frames: return ""
        height, width, _ = frames[0].shape
        temp_path = os.path.join(self.config['output_temp_dir'], f"{event_id}_part_{part_num}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, self.output_fps, (width, height))
        for frame in frames: out.write(frame)
        out.release()
        print(f"  > Wrote {temp_path}")
        return temp_path

    def _concatenate_parts(self, event_id: str, temp_parts: List[str]) -> str:
        if not temp_parts: return ""
        final_video_path = os.path.join(self.config['output_pending_dir'], f"{event_id}.mp4")
        file_list_path = os.path.join(self.config['output_temp_dir'], f"{event_id}_filelist.txt")
        with open(file_list_path, 'w') as f:
            for path in temp_parts: f.write(f"file '{os.path.abspath(path)}'\n")
        command = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0', '-i', file_list_path,
        '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
        '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-shortest',
        '-movflags', '+faststart',
        final_video_path
    ]
        try:
            print(f"Running FFmpeg to create {final_video_path}...")
            subprocess.run(command, check=True, capture_output=True, text=True)
            self._cleanup_temp_files(temp_parts)
            os.remove(file_list_path)
            return final_video_path
        except subprocess.CalledProcessError as e:
            print(f"Error during FFmpeg concatenation: {e.stderr}")
            self._cleanup_temp_files(temp_parts)
            os.remove(file_list_path)
            return ""

    def _create_json_metadata(self, event_id: str, video_path: str, detections: List, start_time: float, end_time: float, actual_duration: float) -> None:
        species_counts = Counter(d['label'] for d in detections)
        unique_species = list(species_counts.keys())
        if species_counts:
            primary = species_counts.most_common(1)[0][0]
        else:
            primary = "N/A"
        metadata = {
            "eventId": event_id,
            "deviceId": self.device_id,
            "location": {
                "latitude": self.location['latitude'],
                "longitude": self.location['longitude']
            },
            "timestamp_start_utc": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            "timestamp_end_utc": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
            "local_video_path": video_path,
            "video_duration_seconds": round(actual_duration, 2),
            "event_summary": {
                "species_list": unique_species,
                "primary_species": primary,
                "max_confidence": max((d['confidence'] for d in detections), default=0.0)
            },
            "detections": detections
        }
        json_path = os.path.join(self.config['output_pending_dir'], f"{event_id}.json")
        with open(json_path, 'w') as f: json.dump(metadata, f, indent=4)
        print(f"  > Wrote JSON metadata to {json_path}")

    def _cleanup_temp_files(self, file_paths: List[str]) -> None:
        for path in file_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error deleting temp file {path}: {e}")