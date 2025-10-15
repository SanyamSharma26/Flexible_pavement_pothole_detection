#!/usr/bin/env python3
"""
Live Pothole Detection with IP Webcam (DroidCam)
Run this script to start live pothole detection using your phone's IP camera (DroidCam)
"""

import sys
import logging
from live_detector import LivePotholeDetector
import cv2

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

IPCAM_URL = "http://172.20.10.2:8080/video"  #  IP stream

def main():
    """Main function to run live detection with IP camera"""
    print("\U0001F697 Starting Live Pothole Detection System (IP Webcam)")
    print("=" * 50)
    print("Controls:")
    print("  Q - Quit")
    print("  C - Toggle confidence display")
    print("  F - Toggle FPS display")
    print("  G - Toggle GPS display")
    print("  D - Toggle depth display")
    print("  S - Save current frame")
    print("=" * 50)
    
    # Create live detector
    live_detector = LivePotholeDetector()
    # Set lower resolution for smoother performance
    if live_detector.cap:
        live_detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        live_detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        live_detector.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    try:
        # Start live detection with IP camera URL
        live_detector.run_interactive(camera_index=IPCAM_URL)
    except KeyboardInterrupt:
        print("\n\U0001F6D1 Stopping live detection...")
    except Exception as e:
        logger.error(f"Error running live detection: {e}")
        print(f"\u274C Error: {e}")
    finally:
        print("\u2705 Live detection stopped")

if __name__ == "__main__":
    main() 