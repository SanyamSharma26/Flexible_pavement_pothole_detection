#!/usr/bin/env python3
"""
Test script for Enhanced Pothole Detector
Tests the enhanced detector with various scenarios to ensure it only detects potholes
"""

import cv2
import time
import logging
from datetime import datetime
import os

from enhanced_pothole_detector import EnhancedPotholeDetector
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_detector():
    """Test the enhanced pothole detector"""
    print("🧪 Testing Enhanced Pothole Detector")
    print("=" * 50)
    
    # Initialize detector
    try:
        detector = EnhancedPotholeDetector()
        print("✅ Enhanced detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False
    
    # Test with camera
    print("\n📹 Testing with camera input...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera at index {config.CAMERA_INDEX}")
        return False
    
    print("✅ Camera opened successfully")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    print("\n🎯 Starting detection test (Press 'q' to quit, 's' to save frame)")
    print("Controls:")
    print("  Q - Quit test")
    print("  S - Save current frame")
    print("  Any other key - Continue")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Process frame
        start_process = time.time()
        potholes, annotated_frame = detector.detect_potholes(frame)
        process_time = time.time() - start_process
        
        # Update detection count
        if potholes:
            detection_count += len(potholes)
            print(f"🔍 Frame {frame_count}: Found {len(potholes)} pothole(s) in {process_time*1000:.1f}ms")
            
            for i, pothole in enumerate(potholes):
                print(f"   Pothole {i+1}: Area={pothole.area:.0f}, "
                      f"Confidence={pothole.confidence:.2f}, "
                      f"Severity={pothole.severity.value}, "
                      f"Depth={pothole.depth:.1f}cm")
        
        # Add test info overlay
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Add info to frame
        info_text = f"Frame: {frame_count} | FPS: {fps:.1f} | Process: {process_time*1000:.1f}ms"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        det_text = f"Total Detections: {detection_count} | Current: {len(potholes)}"
        cv2.putText(annotated_frame, det_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add test status
        status_text = "ENHANCED DETECTOR TEST - Press 'q' to quit"
        cv2.putText(annotated_frame, status_text, (10, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Enhanced Pothole Detector Test', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"test_frame_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"💾 Saved frame as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print test results
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {detection_count}")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
    print(f"Test duration: {total_time:.1f} seconds")
    
    if detection_count > 0:
        print("\n✅ Enhanced detector is working and detecting potholes!")
    else:
        print("\n⚠️  No potholes detected during test (this might be normal if no potholes are present)")
    
    return True

def test_with_video_file():
    """Test with video file if available"""
    video_path = "p.mp4"
    if not os.path.exists(video_path):
        print(f"⚠️  Video file {video_path} not found, skipping video test")
        return
    
    print(f"\n🎬 Testing with video file: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video file: {video_path}")
        return
    
    detector = EnhancedPotholeDetector()
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 10th frame for speed
        if frame_count % 10 == 0:
            potholes, annotated_frame = detector.detect_potholes(frame)
            if potholes:
                detection_count += len(potholes)
                print(f"Frame {frame_count}: Found {len(potholes)} pothole(s)")
        
        # Show progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    print(f"Video test complete: {detection_count} detections in {frame_count} frames")

if __name__ == "__main__":
    print("🚗 Enhanced Pothole Detector Test Suite")
    print("=" * 50)
    
    # Test with camera
    success = test_enhanced_detector()
    
    # Test with video file
    test_with_video_file()
    
    print("\n🎉 Test suite completed!") 