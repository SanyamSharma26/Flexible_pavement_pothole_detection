#!/usr/bin/env python3
"""
Camera Calibration Script for Pothole Detection
Helps calibrate the camera for real-world measurements
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime

from enhanced_pothole_detector import EnhancedPotholeDetector
from config import config

class CameraCalibrator:
    """
    Camera calibration system for real-world measurements
    """
    
    def __init__(self):
        self.detector = EnhancedPotholeDetector()
        self.calibration_data = {}
        self.reference_objects = []
        
    def add_reference_object(self, name: str, real_width_cm: float, real_height_cm: float = None):
        """Add a reference object for calibration"""
        self.reference_objects.append({
            'name': name,
            'real_width_cm': real_width_cm,
            'real_height_cm': real_height_cm or real_width_cm
        })
        print(f"✅ Added reference object: {name} ({real_width_cm}cm wide)")
    
    def interactive_calibration(self, camera_index: int = 0):
        """Interactive calibration using camera feed"""
        print("🎯 Interactive Camera Calibration")
        print("=" * 50)
        print("Instructions:")
        print("1. Place a known reference object in the camera view")
        print("2. Press 'c' to capture the current frame")
        print("3. Click and drag to measure the reference object")
        print("4. Enter the real width in centimeters")
        print("5. Press 'q' to quit")
        print("=" * 50)
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Failed to open camera at index {camera_index}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
        
        measuring = False
        start_point = None
        end_point = None
        captured_frame = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal measuring, start_point, end_point
            
            if event == cv2.EVENT_LBUTTONDOWN:
                measuring = True
                start_point = (x, y)
                end_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and measuring:
                end_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                measuring = False
                end_point = (x, y)
        
        cv2.namedWindow('Camera Calibration')
        cv2.setMouseCallback('Camera Calibration', mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            display_frame = frame.copy()
            
            # Show instructions
            cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if captured_frame is not None:
                cv2.putText(display_frame, "Frame captured! Click and drag to measure", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                display_frame = captured_frame.copy()
            
            # Draw measurement line
            if start_point and end_point:
                cv2.line(display_frame, start_point, end_point, (0, 255, 255), 2)
                cv2.circle(display_frame, start_point, 5, (0, 255, 0), -1)
                cv2.circle(display_frame, end_point, 5, (0, 0, 255), -1)
                
                # Calculate pixel distance
                pixel_distance = np.sqrt((end_point[0] - start_point[0])**2 + 
                                       (end_point[1] - start_point[1])**2)
                cv2.putText(display_frame, f"Pixels: {pixel_distance:.1f}", 
                           (end_point[0] + 10, end_point[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                captured_frame = frame.copy()
                print("📸 Frame captured! Now measure the reference object.")
            elif key == ord('m') and start_point and end_point:
                # Manual measurement
                pixel_distance = np.sqrt((end_point[0] - start_point[0])**2 + 
                                       (end_point[1] - start_point[1])**2)
                
                try:
                    real_width = float(input(f"Enter real width of measured object (cm): "))
                    self.add_reference_object("Manual", real_width)
                    
                    # Calibrate detector
                    success = self.detector.calibrate_camera(captured_frame, pixel_distance)
                    if success:
                        print(f"✅ Camera calibrated! {self.detector.pixels_per_cm:.2f} pixels/cm")
                        self.save_calibration()
                    else:
                        print("❌ Calibration failed")
                        
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\n❌ Calibration cancelled")
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def auto_calibration_with_road_markings(self, camera_index: int = 0):
        """Automatic calibration using road markings"""
        print("🛣️  Automatic Calibration with Road Markings")
        print("=" * 50)
        print("This method assumes standard road markings:")
        print("- Lane width: ~350cm")
        print("- Road marking width: ~15cm")
        print("- Center line width: ~10cm")
        print("=" * 50)
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Failed to open camera at index {camera_index}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
        
        print("📹 Looking for road markings...")
        print("Press 'q' to quit, 'c' to capture and calibrate")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=100, maxLineGap=10)
            
            display_frame = frame.copy()
            
            if lines is not None:
                # Draw detected lines
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Show line count
                cv2.putText(display_frame, f"Lines detected: {len(lines)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "Press 'c' to calibrate, 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Auto Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Use average line width as reference
                if lines is not None and len(lines) > 0:
                    line_widths = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        width = abs(x2 - x1)
                        if width > 0:
                            line_widths.append(width)
                    
                    if line_widths:
                        avg_width = np.mean(line_widths)
                        print(f"📏 Average line width: {avg_width:.1f} pixels")
                        
                        # Assume this is a road marking (15cm wide)
                        success = self.detector.calibrate_camera(frame, avg_width)
                        if success:
                            print(f"✅ Auto-calibrated! {self.detector.pixels_per_cm:.2f} pixels/cm")
                            self.save_calibration()
                        else:
                            print("❌ Auto-calibration failed")
                    else:
                        print("❌ No valid lines detected for calibration")
                else:
                    print("❌ No lines detected. Try again with better road markings.")
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def save_calibration(self):
        """Save calibration data to file"""
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'pixels_per_cm': self.detector.pixels_per_cm,
            'focal_length': self.detector.focal_length,
            'calibration_distance': self.detector.calibration_distance,
            'camera_height': self.detector.camera_height,
            'reference_objects': self.reference_objects
        }
        
        with open('camera_calibration.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"💾 Calibration saved to camera_calibration.json")
    
    def load_calibration(self):
        """Load calibration data from file"""
        if os.path.exists('camera_calibration.json'):
            with open('camera_calibration.json', 'r') as f:
                data = json.load(f)
            
            self.detector.pixels_per_cm = data.get('pixels_per_cm', 10.0)
            self.detector.focal_length = data.get('focal_length', 1000.0)
            self.detector.calibration_distance = data.get('calibration_distance', 100.0)
            self.detector.camera_height = data.get('camera_height', 150.0)
            self.reference_objects = data.get('reference_objects', [])
            
            print(f"📂 Calibration loaded: {self.detector.pixels_per_cm:.2f} pixels/cm")
            return True
        return False

def main():
    """Main calibration function"""
    print("🎯 Camera Calibration for Pothole Detection")
    print("=" * 50)
    
    calibrator = CameraCalibrator()
    
    # Try to load existing calibration
    if calibrator.load_calibration():
        print("✅ Using existing calibration")
    else:
        print("⚠️  No existing calibration found")
    
    print("\nCalibration Options:")
    print("1. Interactive calibration (manual measurement)")
    print("2. Auto calibration (road markings)")
    print("3. Use default calibration")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        calibrator.interactive_calibration(config.CAMERA_INDEX)
    elif choice == '2':
        calibrator.auto_calibration_with_road_markings(config.CAMERA_INDEX)
    elif choice == '3':
        print("✅ Using default calibration (10 pixels/cm)")
        calibrator.save_calibration()
    elif choice == '4':
        print("👋 Exiting calibration")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 