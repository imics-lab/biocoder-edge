import os
import time
import json
import shutil
import psycopg2
import paramiko
from typing import Dict

class DataUploader:
    """
    Manages the resilient upload of event packages (video and JSON)
    to a remote server and database.
    """
    def __init__(self, config: Dict):
        """
        Initializes the Data Uploader.
        :param config: A dictionary with 'uploader' configuration.
        """
        self.config = config['uploader']
        self.pending_dir = config['animal_analyzer']['output_pending_dir']
        self.uploaded_dir = os.path.join(os.path.dirname(self.pending_dir.rstrip('/')), 'uploaded')
        self.is_running = False

    def start(self) -> None:
        """Starts the main uploader loop."""
        if self.is_running:
            print("Data uploader is already running.")
            return
        self.is_running = True
        print("Starting Data Uploader...")
        self._processing_loop()

    def stop(self) -> None:
        """Signals the main loop to terminate gracefully."""
        self.is_running = False
        print("Stopping Data Uploader...")

    def _processing_loop(self) -> None:
        """The main loop for scanning and processing jobs."""
        while self.is_running:
            print(f"Scanning {self.pending_dir} for new jobs...")
            try:
                job_files = [f for f in os.listdir(self.pending_dir) if f.endswith('.json')]
                for job_file in job_files:
                    json_path = os.path.join(self.pending_dir, job_file)
                    self._process_job(json_path)
            except Exception as e:
                print(f"An error occurred during the scan loop: {e}")

            # Sleep for the configured interval
            time.sleep(self.config['scan_interval_seconds'])
    
    def _process_job(self, json_path: str) -> None:
        """
        Handles the complete upload transaction for a single event package.
        """
        print(f"Processing job: {os.path.basename(json_path)}")
        
        db_conn = None
        sftp_client = None
        upload_successful = False

        try:
            # Step 1: Read local data
            # ... Load JSON, find video path ...

            # Step 2: DB INSERT
            # db_conn = self._connect_db()
            # ... Execute INSERT SQL ...

            # Step 3: SFTP UPLOAD
            # sftp_client = self._connect_sftp()
            # ... Upload video and json files ...
            
            # Step 4: DB UPDATE
            # ... Execute UPDATE SQL ...
            
            # If we reach here, all remote operations were successful
            upload_successful = True

        except Exception as e:
            print(f"Failed to process job {os.path.basename(json_path)}. Error: {e}. Will retry later.")
        
        finally:
            # Close connections if they were opened
            # if db_conn: db_conn.close()
            # if sftp_client: sftp_client.close()
            
            # Step 5: LOCAL CLEANUP (only on full success)
            if upload_successful:
                print(f"Successfully uploaded {os.path.basename(json_path)}. Moving local files.")
                # self._move_local_files(json_path, video_path)