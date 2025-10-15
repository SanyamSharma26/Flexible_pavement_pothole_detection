#!/usr/bin/env python3
"""
Live Detection Test Script
This script helps you test if your live pothole detection system is working correctly
"""

import cv2
import time
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def test_camera_access():
    """Test if camera can be accessed"""
    print("🔍 Testing camera access...")
    
    cap = cv2.VideoCapture(0)  # Try camera index 0
    
    if not cap.isOpened():
        print("❌ Camera access failed!")
        print("   Possible issues:")
        print("   - No camera connected")
        print("   - Camera is being used by another application")
        print("   - Wrong camera index")
        return False
    
    # Test reading a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frames from camera!")
        cap.release()
        return False
    
    print(f"✅ Camera working! Frame size: {frame.shape}")
    cap.release()
    return True


def test_model_loading():
    """Test if YOLO model can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        from ultralytics import YOLO
        from config import config
        
        if not os.path.exists(config.YOLO_MODEL_PATH):
            print(f"❌ Model file not found: {config.YOLO_MODEL_PATH}")
            return False
        
        model = YOLO(config.YOLO_MODEL_PATH)
        print("✅ YOLO model loaded successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_detector_initialization():
    """Test if detector can be initialized"""
    print("\n🔧 Testing detector initialization...")
    
    try:
        from detector import PotholeDetector
        
        detector = PotholeDetector()
        print("✅ Detector initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Detector initialization failed: {e}")
        return False


def test_database_connection():
    """Test if database can be accessed"""
    print("\n💾 Testing database connection...")
    
    try:
        from database import PotholeDatabase
        
        db = PotholeDatabase()
        print("✅ Database connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_live_detection_basic():
    """Test basic live detection functionality"""
    print("\n📹 Testing basic live detection...")
    
    try:
        from live_detector import LivePotholeDetector
        
        detector = LivePotholeDetector()
        print("✅ Live detector created successfully!")
        
        # Test camera start
        if detector.start_camera(0):
            print("✅ Camera started successfully!")
            detector.stop()
            return True
        else:
            print("❌ Failed to start camera!")
            return False
            
    except Exception as e:
        print(f"❌ Live detection test failed: {e}")
        return False


def test_single_frame_detection():
    """Test detection on a single frame"""
    print("\n🖼️ Testing single frame detection...")
    
    try:
        from detector import PotholeDetector
        
        # Capture a test frame
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera for frame test")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Cannot read test frame")
            return False
        
        # Test detection
        detector = PotholeDetector()
        potholes, annotated_frame = detector.detect_potholes(frame)
        
        print(f"✅ Detection test completed! Found {len(potholes)} potholes")
        
        # Save test result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_filename = f"test_detection_{timestamp}.jpg"
        cv2.imwrite(test_filename, annotated_frame)
        print(f"✅ Test result saved as: {test_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single frame detection test failed: {e}")
        return False


def run_quick_live_test():
    """Run a quick live test for 10 seconds"""
    print("\n🚀 Running quick live test (10 seconds)...")
    print("   Press 'q' to quit early")
    
    try:
        from live_detector import LivePotholeDetector
        
        detector = LivePotholeDetector()
        
        if not detector.start_camera(0):
            print("❌ Failed to start camera for live test")
            return False
        
        detector.running = True
        
        # Start processing thread
        import threading
        processing_thread = threading.Thread(target=detector.process_frames)
        processing_thread.start()
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:  # Run for 10 seconds
            if detector.frame_queue.empty():
                time.sleep(0.01)
                continue
            
            frame = detector.frame_queue.get()
            frame_count += 1
            
            # Process frame
            potholes, annotated_frame = detector.detector.detect_potholes(frame)
            
            # Display frame
            cv2.imshow('Live Test', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        detector.running = False
        processing_thread.join(timeout=2)
        detector.stop()
        
        fps = frame_count / 10
        print(f"✅ Live test completed! Average FPS: {fps:.1f}")
        return True
        
    except Exception as e:
        print(f"❌ Live test failed: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'cv2',
        'torch',
        'ultralytics',
        'numpy',
        'sqlite3'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'torch':
                import torch
            elif package == 'ultralytics':
                import ultralytics
            elif package == 'numpy':
                import numpy
            elif package == 'sqlite3':
                import sqlite3
            
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install opencv-python torch ultralytics numpy")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def main():
    """Main test function"""
    print("🧪 Live Detection System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Camera Access", test_camera_access),
        ("Model Loading", test_model_loading),
        ("Detector Initialization", test_detector_initialization),
        ("Database Connection", test_database_connection),
        ("Live Detector Creation", test_live_detection_basic),
        ("Single Frame Detection", test_single_frame_detection),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"⚠️ Test '{test_name}' failed")
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your live detection system is ready!")
        print("\nTo start live detection, run:")
        print("   python run_live.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues above before running live detection.")
    
    # Ask if user wants to run quick live test
    if passed_tests >= 5:  # If most tests passed
        print("\n" + "=" * 50)
        response = input("Would you like to run a quick 10-second live test? (y/n): ")
        if response.lower() in ['y', 'yes']:
            run_quick_live_test()


if __name__ == "__main__":
    main() 