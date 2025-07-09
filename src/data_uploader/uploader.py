import os
import time
import json
import shutil
import psycopg2
import paramiko
from typing import Dict, Tuple

class DataUploader:
    """
    Manages the resilient upload of event packages (video and JSON)
    to a remote server and database.
    """
    def __init__(self, config: Dict):
        """
        Initializes the Data Uploader.
        :param config: The full application configuration dictionary.
        """
        self.config = config['uploader']
        self.pending_dir = config['animal_analyzer']['output_pending_dir']
        # Construct the 'uploaded' path relative to the 'pending' path
        self.uploaded_dir = os.path.join(os.path.dirname(os.path.dirname(self.pending_dir.rstrip('/'))), 'uploaded')
        os.makedirs(self.uploaded_dir, exist_ok=True)
        
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
            print(f"Uploader: Scanning {self.pending_dir} for new jobs...")
            try:
                # Find all .json files, which are the triggers for a job
                job_files = [f for f in os.listdir(self.pending_dir) if f.endswith('.json')]
                if not job_files:
                    print("Uploader: No new jobs found.")
                
                for job_file in job_files:
                    if not self.is_running: break # Allow for a faster exit
                    json_path = os.path.join(self.pending_dir, job_file)
                    self._process_job(json_path)

            except Exception as e:
                print(f"Uploader: An unexpected error occurred during the scan loop: {e}")

            # Sleep for the configured interval before the next scan
            for _ in range(self.config['scan_interval_seconds']):
                if not self.is_running: break
                time.sleep(1)
    
    def _process_job(self, json_path: str) -> None:
        """
        Handles the complete upload transaction for a single event package.
        This function is designed to be atomic: if any step fails, the entire
        operation for this job is aborted, leaving local files untouched for retry.
        """
        print(f"Uploader: Processing job -> {os.path.basename(json_path)}")
        
        db_conn = None
        ssh_client = None
        upload_successful = False

        try:
            # --- Step 1: Read and Validate Local Data ---
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            video_path = metadata.get('local_video_path')
            if not video_path or not os.path.exists(video_path):
                print(f"Uploader: Error - Video file not found for {json_path}. Skipping.")
                return

            # --- Step 2: DB INSERT (Phase 1) ---
            print("  > Connecting to database...")
            db_conn = self._connect_db()
            cursor = db_conn.cursor()
            
            # This INSERT query assumes a table named 'events' exists.
            # It sets the status to 'pending'.
            sql_insert = """
                INSERT INTO events (eventId, deviceId, timestamp_start_utc, timestamp_end_utc, 
                                    video_duration_seconds, primary_species, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                ON CONFLICT (eventId) DO NOTHING;
            """
            # Using ON CONFLICT prevents errors if we retry a job where INSERT succeeded but a later step failed.
            cursor.execute(sql_insert, (
                metadata['eventId'], metadata['deviceId'], metadata['timestamp_start_utc'],
                metadata['timestamp_end_utc'], metadata['video_duration_seconds'],
                metadata['event_summary']['primary_species']
            ))
            db_conn.commit()
            print("  > DB record inserted/ensured in 'pending' state.")

            # --- Step 3: SFTP UPLOAD ---
            print("  > Connecting to SFTP server...")
            ssh_client, sftp_client = self._connect_sftp()
            
            remote_video_name = os.path.basename(video_path)
            remote_json_name = os.path.basename(json_path)
            
            remote_video_path = os.path.join(self.config['sftp']['remote_video_dir'], remote_video_name)
            remote_json_path = os.path.join(self.config['sftp']['remote_json_dir'], remote_json_name)

            print(f"  > Uploading video to {remote_video_path}...")
            sftp_client.put(video_path, remote_video_path)
            print(f"  > Uploading json to {remote_json_path}...")
            sftp_client.put(json_path, remote_json_path)
            
            # --- Step 4: DB UPDATE (Phase 2) ---
            print("  > Updating database record to 'completed'...")
            sql_update = """
                UPDATE events 
                SET status = 'completed', remote_video_path = %s, remote_json_path = %s
                WHERE eventId = %s;
            """
            cursor.execute(sql_update, (remote_video_path, remote_json_path, metadata['eventId']))
            db_conn.commit()
            
            # If we reach here, all remote operations were successful
            upload_successful = True

        except (psycopg2.Error, paramiko.SSHException, IOError) as e:
            print(f"Uploader: A recoverable error occurred while processing {os.path.basename(json_path)}. Error: {e}. Will retry later.")
            # Rollback any partial DB transaction
            if db_conn:
                db_conn.rollback()
        
        finally:
            # --- Ensure all connections are closed ---
            if db_conn:
                db_conn.close()
            if ssh_client:
                ssh_client.close()
            
            # --- Step 5: LOCAL CLEANUP (only on full success) ---
            if upload_successful:
                print(f"Uploader: Successfully uploaded {os.path.basename(json_path)}. Moving local files.")
                self._move_local_files(json_path, video_path)

    def _connect_db(self):
        """Establishes and returns a PostgreSQL database connection."""
        return psycopg2.connect(**self.config['database'])

    def _connect_sftp(self) -> Tuple[paramiko.SSHClient, paramiko.SFTPClient]:
        """Establishes and returns an SFTP client and its parent SSH client."""
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh_client.connect(
            hostname=self.config['sftp']['host'],
            port=self.config['sftp']['port'],
            username=self.config['sftp']['username'],
            key_filename=os.path.expanduser(self.config['sftp']['ssh_key_path'])
        )
        sftp_client = ssh_client.open_sftp()
        return ssh_client, sftp_client

    def _move_local_files(self, json_path: str, video_path: str) -> None:
        """Moves successfully uploaded files to the 'uploaded' directory."""
        try:
            shutil.move(json_path, os.path.join(self.uploaded_dir, os.path.basename(json_path)))
            shutil.move(video_path, os.path.join(self.uploaded_dir, os.path.basename(video_path)))
        except (IOError, OSError) as e:
            print(f"Uploader: Critical error - Failed to move local files after successful upload: {e}")