#!/usr/bin/env python3
"""
Live Pothole Detection Launcher
Run this script to start live pothole detection with your webcam or phone camera
"""

import sys
import logging
from live_detector import LivePotholeDetector

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run live detection"""
    print("🚗 Starting Live Pothole Detection System")
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
    
    try:
        # Start live detection
        live_detector.run_interactive()
    except KeyboardInterrupt:
        print("\n🛑 Stopping live detection...")
    except Exception as e:
        logger.error(f"Error running live detection: {e}")
        print(f"❌ Error: {e}")
    finally:
        print("✅ Live detection stopped")


if __name__ == "__main__":
    main() 