import os
import posixpath
import time
import json
import shutil
import logging
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
        self.uploaded_dir = os.path.join(os.path.dirname(self.pending_dir.rstrip('/')), 'uploaded')
        os.makedirs(self.uploaded_dir, exist_ok=True)

        self.failed_dir = os.path.join(os.path.dirname(self.pending_dir.rstrip('/')), 'failed')
        os.makedirs(self.failed_dir, exist_ok=True)

        self._upload_failures: Dict[str, int] = {}
        self._max_upload_retries = 3

        self.is_running = False

    def start(self) -> None:
        """Starts the main uploader loop."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_uploader.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger('DataUploader')
        
        if not self.config.get('enabled', True):
            logger.info("Data uploader is disabled in configuration. Files will remain in pending_upload folder.")
            return
        
        if self.is_running:
            logger.warning("Data uploader is already running.")
            return
        self.is_running = True
        logger.info("Starting Data Uploader...")
        self._processing_loop()

    def stop(self) -> None:
        """Signals the main loop to terminate gracefully."""
        self.is_running = False
        print("Stopping Data Uploader...")

    def _processing_loop(self) -> None:
        """The main loop for scanning and processing jobs."""
        logger = logging.getLogger('DataUploader')
        while self.is_running:
            logger.info("Scanning %s for new jobs...", self.pending_dir)
            try:
                job_files = [f for f in os.listdir(self.pending_dir) if f.endswith('.json')]
                if not job_files:
                    logger.info("No new jobs found.")
                
                for job_file in job_files:
                    if not self.is_running: break
                    json_path = os.path.join(self.pending_dir, job_file)
                    self._process_job(json_path)

            except Exception as e:
                logger.exception("An unexpected error occurred during the scan loop: %s", e)

            for _ in range(self.config['scan_interval_seconds']):
                if not self.is_running: break
                time.sleep(1)
    
    def _record_failure(self, job_key: str, json_path: str, video_path: str, event_id: str, error_msg: str) -> bool:
        """
        Records a failure for a job and moves it to failed/ if max retries exceeded.
        Returns True if the job was permanently failed, False if it will be retried.
        """
        logger = logging.getLogger('DataUploader')
        count = self._upload_failures.get(job_key, 0) + 1
        self._upload_failures[job_key] = count
        if count >= self._max_upload_retries:
            logger.error(
                "Giving up on %s after %d failed attempts. Last error: %s. Moving to failed/.",
                job_key, count, error_msg
            )
            self._fail_job(json_path, video_path, event_id)
            return True
        logger.error("  > %s (attempt %d/%d). Will retry later.", error_msg, count, self._max_upload_retries)
        return False

    def _process_job(self, json_path: str) -> None:
        """
        Handles the complete upload transaction for a single event package.
        Uses two separate DB connections to avoid keeping a connection idle
        during long SFTP uploads.
        """
        logger = logging.getLogger('DataUploader')
        job_key = os.path.basename(json_path)
        logger.info("Processing job -> %s", job_key)

        event_id = None
        video_path = None

        try:
            # --- Step 1: Read and Validate Local Data ---
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            event_id = metadata['eventId']
            video_path = metadata.get('local_video_path')
            if not video_path or not os.path.exists(video_path):
                self._record_failure(
                    job_key, json_path, None, event_id,
                    "Video file not found for %s" % json_path
                )
                return

            # --- Step 2: DB INSERT (Phase 1) ---
            # Mark the event as 'pending' in the database.
            # We connect and disconnect immediately.
            db_conn = None
            try:
                db_conn = self._connect_db()
                with db_conn:
                    with db_conn.cursor() as cursor:
                        sql_insert = """
                            INSERT INTO events (event_id, device_id, timestamp_start_utc, timestamp_end_utc,
                                                video_duration_seconds, primary_species, status, timezone, latitude, longitude)
                            VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, %s, %s)
                            ON CONFLICT (event_id) DO NOTHING;
                        """
                        cursor.execute(sql_insert, (
                            metadata['eventId'], metadata['deviceId'], metadata['timestamp_start_utc'],
                            metadata['timestamp_end_utc'], metadata['video_duration_seconds'],
                            metadata['event_summary']['primary_species'], metadata['location']['timezone'],
                            metadata['location']['latitude'], metadata['location']['longitude']
                        ))
                logger.info("  > DB record ensured in 'pending' state.")
            except psycopg2.Error as e:
                self._record_failure(
                    job_key, json_path, video_path, event_id,
                    "Failed to ensure DB record: %s" % e
                )
                return
            finally:
                if db_conn:
                    db_conn.close()

            # --- Step 3: SFTP UPLOAD (No DB connection open) ---
            remote_video_name = os.path.basename(video_path)
            remote_json_name = os.path.basename(json_path)
            # Use posixpath for SFTP paths (always POSIX-style, regardless of client OS)
            remote_video_path = posixpath.join(self.config['sftp']['remote_video_dir'], remote_video_name)
            remote_json_path = posixpath.join(self.config['sftp']['remote_json_dir'], remote_json_name)

            ssh_client = None
            try:
                ssh_client, sftp_client = self._connect_sftp()

                # Smart upload for Video
                local_video_size = os.path.getsize(video_path)
                try:
                    remote_stat = sftp_client.stat(remote_video_path)
                    if remote_stat.st_size == local_video_size:
                        logger.info("  > Video already exists on remote server with correct size. Skipping upload.")
                    else:
                        logger.warning("  > Remote video size mismatch (%d vs %d). Re-uploading.", remote_stat.st_size, local_video_size)
                        sftp_client.put(video_path, remote_video_path)
                except IOError:
                    logger.info("  > Uploading video to %s...", remote_video_path)
                    sftp_client.put(video_path, remote_video_path)

                # Smart upload for JSON
                local_json_size = os.path.getsize(json_path)
                try:
                    remote_stat = sftp_client.stat(remote_json_path)
                    if remote_stat.st_size == local_json_size:
                        logger.info("  > JSON metadata already exists on remote server. Skipping upload.")
                    else:
                        logger.warning("  > Remote JSON size mismatch (%d vs %d). Re-uploading.", remote_stat.st_size, local_json_size)
                        sftp_client.put(json_path, remote_json_path)
                except IOError:
                    logger.info("  > Uploading json to %s...", remote_json_path)
                    sftp_client.put(json_path, remote_json_path)

            except (paramiko.SSHException, IOError) as e:
                self._record_failure(
                    job_key, json_path, video_path, event_id,
                    "SFTP error: %s" % e
                )
                return
            finally:
                if ssh_client:
                    ssh_client.close()

            # --- Step 4: DB UPDATE (Phase 2) ---
            # Mark the event as 'completed' and insert detections.
            db_conn = None
            try:
                db_conn = self._connect_db()
                with db_conn:
                    with db_conn.cursor() as cursor:
                        # 4.1 Update main event record
                        logger.info("  > Updating DB record to 'completed'...")
                        sql_update = """
                            UPDATE events
                            SET status = 'completed', remote_video_path = %s, remote_json_path = %s
                            WHERE event_id = %s;
                        """
                        cursor.execute(sql_update, (remote_video_path, remote_json_path, metadata['eventId']))

                        # 4.2 Insert into detections table
                        logger.info("  > Inserting detections...")
                        classes_detected = metadata['event_summary']['species_list']
                        counts = {}
                        for det in metadata.get('detections', []):
                            lbl = det['label']
                            counts[lbl] = counts.get(lbl, 0) + 1

                        sql_insert_det = """
                            INSERT INTO detections (event_id, detection_json, classes_detected, max_count_per_frame)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (event_id) DO NOTHING;
                        """
                        cursor.execute(sql_insert_det, (
                            metadata['eventId'], json.dumps(metadata),
                            classes_detected, json.dumps(counts)
                        ))

                # If we get here, everything is done!
                logger.info("Successfully processed %s. Moving local files.", job_key)
                self._move_local_files(json_path, video_path)
                self._upload_failures.pop(job_key, None)

            except psycopg2.Error as e:
                self._record_failure(
                    job_key, json_path, video_path, event_id,
                    "Failed to finalize DB: %s" % e
                )
            finally:
                if db_conn:
                    db_conn.close()

        except Exception as e:
            self._record_failure(
                job_key, json_path, video_path, event_id,
                "Unexpected error: %s" % e
            )

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
        logger = logging.getLogger('DataUploader')
        try:
            shutil.move(json_path, os.path.join(self.uploaded_dir, os.path.basename(json_path)))
            shutil.move(video_path, os.path.join(self.uploaded_dir, os.path.basename(video_path)))
        except (IOError, OSError) as e:
            logger.critical("Failed to move local files after successful upload: %s", e)

    def _fail_job(self, json_path: str, video_path: str, event_id: str) -> None:
        """Moves a permanently failed job to the 'failed' directory and updates DB status."""
        logger = logging.getLogger('DataUploader')
        job_key = os.path.basename(json_path)

        if event_id:
            db_conn = None
            try:
                db_conn = self._connect_db()
                with db_conn:
                    with db_conn.cursor() as cursor:
                        cursor.execute(
                            "UPDATE events SET status = 'upload_failed' WHERE event_id = %s;",
                            (event_id,)
                        )
                logger.info("  > DB record for %s updated to 'upload_failed'.", event_id)
            except psycopg2.Error as e:
                logger.error("Failed to update DB status to 'upload_failed' for %s: %s", event_id, e)
            finally:
                if db_conn:
                    db_conn.close()

        try:
            shutil.move(json_path, os.path.join(self.failed_dir, os.path.basename(json_path)))
            if video_path and os.path.exists(video_path):
                shutil.move(video_path, os.path.join(self.failed_dir, os.path.basename(video_path)))
            logger.info("  > Files for %s moved to %s.", job_key, self.failed_dir)
        except (IOError, OSError) as e:
            logger.critical("Failed to move files to failed/ for %s: %s", job_key, e)

        self._upload_failures.pop(job_key, None)
