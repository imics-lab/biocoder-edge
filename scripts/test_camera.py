#!/usr/bin/env python3
"""
Camera Test Script

This script tests the camera/video source before running the main BioCoder-Edge application.
It emulates how the MotionDetector initializes and uses the camera.

Usage:
    python scripts/test_camera.py                    # Test default camera (index 0)
    python scripts/test_camera.py --source 1         # Test camera at index 1
    python scripts/test_camera.py --source video.mp4 # Test with a video file
"""

import cv2
import time
import argparse
import sys


def gstreamer_pipeline(
    device_id=0,
    capture_width=640,
    capture_height=480,
    framerate=30
):
    """
    Constructs a GStreamer pipeline for Jetson.
    This simplified pipeline allows for more direct format negotiation.
    """
    return (
        f"v4l2src device=/dev/video{device_id} ! "
        "nvvidconv ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )


def test_camera(video_source, use_gstreamer=True):
    """
    Tests the camera/video source by displaying frames and basic information.
    
    Args:
        video_source: Camera index (int) or path to video file (str)
        use_gstreamer: If True, use a GStreamer pipeline for camera capture.
    """
    print("=" * 60)
    print("BioCoder-Edge Camera Test")
    print("=" * 60)
    print(f"Video source: {video_source}")
    print()
    
    # Initialize the camera (same as detector.py does)
    print("Initializing video source...")
    if isinstance(video_source, int) and use_gstreamer:
        pipeline = gstreamer_pipeline(device_id=video_source)
        print(f"Using GStreamer pipeline: {pipeline}")
        camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        print("Using default OpenCV backend.")
        camera = cv2.VideoCapture(video_source)
    
    if not camera.isOpened():
        print(f"FATAL: Cannot open video source: {video_source}")
        print("\nTroubleshooting tips:")
        print("  - If using a camera index, try different values (0, 1, 2, etc.)")
        print("  - Check if another application is using the camera")
        print("  - Verify camera permissions")
        print("  - If using a file, check the path is correct")
        return False
    
    print("✓ Video source opened successfully!")
    print()
    
    # Get and display camera properties
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    
    print("Camera Properties:")
    print(f"  Resolution: {width} x {height}")
    print(f"  FPS: {fps:.2f}" if fps > 0 else "  FPS: Unknown")
    
    # Detect if it's a video file
    is_video_file = isinstance(video_source, str)
    if is_video_file:
        frame_count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        print(f"  Total frames: {frame_count}")
        print(f"  Duration: {duration:.2f} seconds")
    
    print()
    print("Starting live preview...")
    print("Press 'q' to quit, 's' to save a snapshot")
    print("-" * 60)
    
    # Frame processing loop
    frame_count = 0
    start_time = time.time()
    consecutive_failures = 0
    max_failures = 5
    snapshot_count = 0
    
    try:
        while True:
            # Read frame (same as detector.py)
            ret, frame = camera.read()
            
            if not ret:
                if is_video_file:
                    print("\nEnd of video file reached.")
                    break
                
                consecutive_failures += 1
                print(f"Failed to grab frame (attempt {consecutive_failures})")
                
                if consecutive_failures >= max_failures:
                    print("Maximum consecutive failures reached. Exiting.")
                    break
                
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            frame_count += 1
            
            # Calculate actual FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed
            else:
                actual_fps = 0
            
            # Add information overlay on the frame
            display_frame = frame.copy()
            
            # Background rectangle for text readability
            cv2.rectangle(display_frame, (5, 5), (350, 100), (0, 0, 0), -1)
            
            # Text overlay
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {actual_fps:.2f}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Size: {width}x{height}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Camera Test - Press 'q' to quit, 's' to save", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuit requested by user.")
                break
            elif key == ord('s'):
                snapshot_count += 1
                snapshot_name = f"camera_snapshot_{snapshot_count}.jpg"
                cv2.imwrite(snapshot_name, frame)
                print(f"Saved snapshot: {snapshot_name}")
            
            # Simulate FPS for video files (same as detector.py)
            if is_video_file and fps > 0:
                time.sleep(1 / fps)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Cleanup (same as detector.py)
        print("\nCleaning up...")
        camera.release()
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print("Test Summary:")
        print(f"  Total frames captured: {frame_count}")
        print(f"  Test duration: {elapsed:.2f} seconds")
        if elapsed > 0:
            print(f"  Average FPS: {frame_count / elapsed:.2f}")
        if snapshot_count > 0:
            print(f"  Snapshots saved: {snapshot_count}")
        print("=" * 60)
        print("✓ Camera test completed successfully!")
        print()
    
    return True


def main():
    """Main entry point for the camera test script."""
    parser = argparse.ArgumentParser(
        description="Test camera/video source for BioCoder-Edge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_camera.py                      # Test default camera
  python scripts/test_camera.py --source 1           # Test camera at index 1
  python scripts/test_camera.py --source video.mp4   # Test video file
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help="Camera index (0, 1, 2, ...) or path to video file (default: 0)"
    )
    
    parser.add_argument(
        '--no-gstreamer',
        action='store_true',
        help="Do not use GStreamer pipeline (use default OpenCV backend)"
    )
    
    args = parser.parse_args()
    
    # Convert to int if it's a numeric camera index
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
    
    # Run the test
    success = test_camera(video_source, use_gstreamer=not args.no_gstreamer)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

