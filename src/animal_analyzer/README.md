### **Specification for Module B: Animal Analysis and Data Packaging Pipeline**

#### **1. Overview**

This document outlines the functional requirements and technical specifications for the Animal Analysis module (Module B). This module serves as the primary intelligence and data processing unit of the Vision-Based Animal Monitoring System. Its core responsibility is to receive streams of video frames from the Motion Detection module (Module A), analyze them using a YOLO object detection model to identify species of interest, and package confirmed events (video and metadata) for local storage and subsequent upload.

#### **2. Core Functional Requirements**

*   **Event Frame Consumption:** The module must continuously monitor a shared data queue for incoming video frames that constitute a motion event.
*   **Object Detection:** For each frame received, a YOLO-based object detection model shall be executed to identify objects and their bounding boxes.
*   **Event Significance Analysis:** The module must analyze the sequence of detections across an entire event to robustly confirm the presence of a "species of interest," filtering out transient false positives and tolerating transient false negatives.
*   **Memory Management:** For long events, the module must manage memory usage efficiently, preventing crashes due to excessive RAM consumption by offloading frame data to temporary disk storage.
*   **Data Packaging:** For confirmed significant events, the module must produce two tightly-linked assets:
    1.  A final, consolidated MP4 video file of the event.
    2.  A detailed JSON file containing event metadata, including all relevant detections.
*   **Data Persistence:** The packaged video and JSON files must be saved to a designated local directory, ready for a separate upload process.

#### **3. Interface Contract: Communication with Module A**

This section defines the mandatory interface for data exchange with the upstream Motion Detection module.

*   **Communication Channel:** The module will receive a `multiprocessing.Queue` object during its initialization. This queue is the sole channel for inbound data from Module A.
*   **Data Reception Protocol:** The module must be prepared to receive two types of objects from the queue:
    1.  **Video Frames:**
        *   **Object Type:** `numpy.ndarray`
        *   **Expected Dimensions/Type:** `(Height, Width, 3)`, `uint8`, BGR color space.
    2.  **End-of-Event Signal:**
        *   **Object Type:** The Python `None` object. This signal indicates that the current motion event has concluded and no more frames for this event will be sent. The module must then finalize the processing of the received event.

#### **4. Technical Implementation Specifications**

*   **Primary Libraries:** `opencv-python`, `numpy`, `ultralytics` (or other YOLO library), `multiprocessing`, `time`, `json`, `os`, `subprocess` (for FFmpeg).
*   **Class Structure:** All logic should be encapsulated within a Python class named `AnimalAnalyzer`. This class will manage the YOLO model, event state, and data packaging logic.
*   **Configuration Management:** Parameters must be externalized into a configuration dictionary.

    **Required Configuration Parameters:**
    *   `SPECIES_OF_INTEREST` (list of str): A list of class names that are considered significant (e.g., `['deer', 'bear', 'fox']`).
    *   `YOLO_MODEL_PATH` (str): The file path to the trained YOLO model weights (e.g., `'yolov8n.pt'`).
    *   `CONFIDENCE_THRESHOLD` (float): The minimum confidence score for a YOLO detection to be considered valid (e.g., `0.4`).
    *   `EVENT_CONFIRMATION_FRAMES` (int): The number of consecutive frames an object must be tracked to confirm it as a significant entity (e.g., `5`).
    *   `TRACKING_INACTIVITY_FRAMES` (int): The number of frames an object can be missing before its track is dropped (e.g., `10`).
    *   `RAM_FRAME_LIMIT` (int): The maximum number of frames to hold in RAM before spilling to a temporary disk file (e.g., `300`).
    *   `OUTPUT_PENDING_DIR` (str): The directory path to save final video and JSON packages (e.g., `'/data/pending_upload/'`).
    *   `OUTPUT_TEMP_DIR` (str): A directory for storing temporary video parts (e.g., `'/tmp/'`).

#### **5. Processing Pipeline Logic**

The `AnimalAnalyzer` class shall implement a main processing loop that waits for and processes events.

1.  **Initialization Phase:**
    *   Load the YOLO model into memory.
    *   Ensure the output directories (`OUTPUT_PENDING_DIR`, `OUTPUT_TEMP_DIR`) exist.

2.  **Main Loop (Event-Driven):**
    a. **Wait for Event Start:** The loop blocks on `queue.get()`. The first frame received marks the beginning of a new event.
    b. **Initialize Event State:** Upon receiving the first frame, create a new event context, including: a list for frames in RAM (`ram_buffer`), a list for temp video file paths (`temp_file_parts`), a dictionary for object tracking (`tracked_objects`), and a `significant_event_detected` flag set to `False`.
    c. **Per-Frame Event Processing Loop:** Continue getting items from the queue until `None` is received.
        i.   Add the new frame to the `ram_buffer`.
        ii.  Run YOLO inference on the frame.
        iii. Implement object tracking logic to update `tracked_objects`. Match new detections to existing tracks based on proximity. Update `frames_seen` and `frames_since_seen` counters for each tracked object.
        iv.  Check for event significance: If any tracked object in `SPECIES_OF_INTEREST` has its `frames_seen` counter exceed `EVENT_CONFIRMATION_FRAMES`, set `significant_event_detected = True`.
        v.   **Memory Management:** If the length of `ram_buffer` exceeds `RAM_FRAME_LIMIT`, encode the frames in the buffer into a temporary MP4 video file in `OUTPUT_TEMP_DIR`. Add the file path to the `temp_file_parts` list and clear the `ram_buffer`.

3.  **Event Finalization (Triggered by `None` from queue):**
    a. **Decision Point:** Check the `significant_event_detected` flag for the completed event.
    b. **If Event is NOT Significant:** Discard all data. Delete any files in `temp_file_parts` and clear the `ram_buffer`. Return to waiting for the next event.
    c. **If Event IS Significant:** Proceed with packaging.
        i.   **Finalize Video:** If `ram_buffer` is not empty, write its contents to a final temporary video part. If any temporary parts exist, use an external tool like FFmpeg (via `subprocess`) to concatenate all files in `temp_file_parts` into a single, final video file in `OUTPUT_PENDING_DIR`. The filename should be unique (e.g., based on a timestamp). Delete the temporary parts.
        ii.  **Generate JSON Metadata:** Create a comprehensive JSON object containing all event metadata (see section below). The JSON should include the final video's filename.
        iii. **Save JSON:** Save the JSON object to a file in `OUTPUT_PENDING_DIR` with a name matching the video file.

#### **6. JSON Metadata Structure**

The generated JSON file must conform to the following structure:

```json
{
  "eventId": "string",            // Unique identifier for the event (e.g., timestamp-based)
  "deviceId": "string",           // Unique ID of the Jetson device
  "location": {
    "latitude": "float",
    "longitude": "float"
  },
  "timestamp_start_utc": "string", // ISO 8601 timestamp for the event start
  "timestamp_end_utc": "string",   // ISO 8601 timestamp for the event end
  "local_video_path": "string",   // Full local path to the final video file
  "video_duration_seconds": "float",
  "event_summary": {
    "species_list": ["string"],     // List of unique confirmed species labels
    "primary_species": "string",    // The most confident or persistent species
    "max_confidence": "float"       // Highest confidence score observed for a species of interest
  },
  "detections": [                   // A detailed list of every confirmed detection
    {
      "frame_index": "int",         // The frame number within the event video
      "label": "string",            // The detected class label
      "confidence": "float",        // The YOLO confidence score
      "box_xywh": ["int"]           // Bounding box [x_center, y_center, width, height]
    }
  ]
}
```
