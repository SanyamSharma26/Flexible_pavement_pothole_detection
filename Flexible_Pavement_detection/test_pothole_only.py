#!/usr/bin/env python3
"""
Test script for pothole-only detection
"""

import cv2
import logging
from pothole_only_detector import PotholeOnlyDetector

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def test_pothole_only_detection():
    """Test the pothole-only detection system"""
    print("🧪 Testing Pothole-Only Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = PotholeOnlyDetector()
    print("✅ Pothole-only detector initialized")
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("📹 Camera opened successfully")
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 3rd frame for better performance
        if frame_count % 3 != 0:
            continue
        
        # Detect potholes only
        potholes, annotated_frame = detector.detect_potholes(frame)
        
        # Add info overlay
        info_text = f"Frame: {frame_count} | Potholes: {len(potholes)} | Total Detections: {detection_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", (10, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Pothole-Only Detection Test', annotated_frame)
        
        # Count detections
        if potholes:
            detection_count += len(potholes)
            print(f"Frame {frame_count}: Found {len(potholes)} potholes")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            cv2.imwrite(f"pothole_test_frame_{frame_count}.jpg", annotated_frame)
            print(f"Frame saved as pothole_test_frame_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Test Results:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Total potholes detected: {detection_count}")
    print(f"   Average detections per frame: {detection_count/frame_count:.2f}")
    print("✅ Test completed!")

if __name__ == "__main__":
    test_pothole_only_detection() 