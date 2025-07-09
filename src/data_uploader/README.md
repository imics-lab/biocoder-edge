### **Specification for Module C: Data Uploader**

#### **1. Overview**

This document outlines the functional requirements and technical specifications for the Data Uploader module (Module C). This module operates as a resilient, background service responsible for the reliable transmission of processed data from the edge device to a remote cloud server. Its primary function is to periodically scan for completed event packages, upload them to their designated storage and database endpoints, and handle network failures gracefully to ensure no data is lost.

#### **2. Core Functional Requirements**

*   **Asynchronous Operation:** The module must run independently of the data capture and analysis modules, operating on a periodic schedule.
*   **Filesystem-Based Queueing:** The module will use a designated local directory as its input queue, processing data packages as they appear.
*   **Transactional Uploads:** Each event package (composed of a video and a JSON file) shall be treated as a single, atomic unit of work. The entire upload process for a single event must succeed before local files are cleaned up.
*   **Secure Data Transfer:** All file transfers to the remote server must be conducted over a secure protocol (SFTP).
*   **Database Integration:** The module must record event metadata in a remote PostgreSQL database, updating the record's status upon successful file transfer.
*   **Resilience and Auto-Retry:** In the event of a network or server error at any stage of the upload process, the local files for the current job must remain untouched, allowing the system to automatically retry the upload on a subsequent scan.

#### **3. Interface Contract: Communication with Module B**

The interface for this module is based on the local filesystem, providing a decoupled "hand-off" point from the Animal Analyzer (Module B).

*   **Input Interface:**
    *   **Location:** A designated "pending upload" directory (e.g., `data/pending_upload/`).
    *   **Trigger:** The presence of a `.json` file in this directory signals a new job.
    *   **Contract:** For every `[event_id].json` file, a corresponding `[event_id].mp4` file is expected to exist in the same directory.
*   **Output Interface:**
    *   **Location:** A designated "uploaded" directory (e.g., `data/uploaded/`).
    *   **Action:** Upon the successful completion of an entire upload transaction, the corresponding local `.json` and `.mp4` files are moved from the input directory to this output directory for archival purposes.

#### **4. Technical Implementation Specifications**

*   **Primary Libraries:** `psycopg2-binary` (for PostgreSQL), `paramiko` (for SFTP), `os`, `time`, `json`, `shutil`.
*   **Class Structure:** All logic should be encapsulated within a Python class named `DataUploader`.
*   **Configuration Management:** All external parameters will be provided via a configuration dictionary.

    **Required Configuration Parameters (`uploader` section):**
    *   `scan_interval_seconds` (int): The number of seconds the module sleeps between scanning for new files.
    *   `sftp` (dict):
        *   `host` (str): IP address or hostname of the SFTP server.
        *   `port` (int): SFTP port (typically `22`).
        *   `username` (str): SFTP username.
        *   `ssh_key_path` (str): Path to the private SSH key for authentication.
        *   `remote_video_dir` (str): The absolute path on the remote server to store video files.
        *   `remote_json_dir` (str): The absolute path on the remote server to store JSON files.
    *   `database` (dict):
        *   `host`, `port`, `dbname`, `user`, `password` for the PostgreSQL connection.

#### **5. Processing Pipeline Logic**

The `DataUploader` class will implement a main loop that orchestrates the upload process. The logic for handling a single job must be strictly sequential and fail-safe.

1.  **Main Loop:**
    a. The loop runs indefinitely as long as the service is active.
    b. **Scan for Jobs:** Get a list of all `.json` files in the configured `pending_upload` directory.
    c. **Process Jobs:** Iterate through the list of found JSON files. For each file, execute the `_process_job` logic described below.
    d. **Sleep:** After iterating through all found jobs, sleep for `scan_interval_seconds`.

2.  **Single Job Processing Logic (`_process_job`):**
    This logic must be wrapped in a `try...except` block to catch any exceptions (network errors, file not found, etc.) and ensure the main loop continues to run.

    a. **Read Local Data:** Load the JSON file. Check for the existence of the corresponding video file. If the video is missing, log an error and skip this job.
    b. **Phase 1: Database INSERT:**
        i.   Establish a connection to the PostgreSQL database.
        ii.  Execute an `INSERT` statement into the `events` table with the metadata from the JSON file. Set the upload status field to `'pending'` and leave remote path fields as `NULL`.
        iii. If the `INSERT` fails, log the error, close the DB connection, and abort the job. The local files remain, to be retried later.
    c. **Phase 2: Secure File Transfer (SFTP):**
        i.   Establish an SFTP connection to the remote server.
        ii.  Upload the video file to the `remote_video_dir`.
        iii. Upload the JSON file to the `remote_json_dir`.
        iv.  If any file transfer fails, log the error, close the SFTP connection, and abort the job. The database record will remain in the 'pending' state, and local files will be retried later.
    d. **Phase 3: Database UPDATE:**
        i.   If file transfers succeed, execute an `UPDATE` statement on the `events` table for the current `eventId`.
        ii.  Update the record to set the `status` to `'completed'` and fill in the `remote_video_path` and `remote_json_path` columns.
        iii. If the `UPDATE` fails, log a critical error (as this indicates an inconsistent state) and abort the job. The retry mechanism will handle it, though it might re-upload the files.
    e. **Phase 4: Local Cleanup:**
        i.   **If and only if all previous phases (INSERT, SFTP, UPDATE) have completed successfully**, move the local `.json` and `.mp4` files from the `pending_upload` directory to the `uploaded` directory using `shutil.move`.
