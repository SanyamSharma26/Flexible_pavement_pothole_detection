import cv2
import numpy as np
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from queue import Queue
import torch

from config import config
from enhanced_pothole_detector import EnhancedPotholeDetector
from models import Pothole, Severity
from database import PotholeDatabase
from utils import save_detection_image

logger = logging.getLogger(__name__)


class LivePotholeDetector:
    """
    Enhanced live pothole detection system with real-time processing
    """
    
    def __init__(self):
        self.detector = EnhancedPotholeDetector()
        self.db = PotholeDatabase()
        self.running = False
        self.frame_queue = Queue(maxsize=3)  # Limit queue size for real-time performance
        self.result_queue = Queue(maxsize=10)
        self.cap = None
        self.processing_thread = None
        self.display_thread = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection statistics
        self.total_detections = 0
        self.detection_history = []
        
        # UI elements
        self.show_confidence = True
        self.show_fps = True
        self.show_gps = True
        self.show_depth = True
        
    def start_camera(self, camera_index: int = 0) -> bool:
        """Initialize and start the camera"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {camera_index}")
                return False
                
            logger.info(f"Camera started successfully at {config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def capture_frames(self):
        """Capture frames from camera in a separate thread"""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.01)
                continue
                
            # Clear old frames and add new one
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
                    
            self.frame_queue.put(frame)
    
    def process_frames(self):
        """Process frames for pothole detection in a separate thread"""
        while self.running:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame = self.frame_queue.get(timeout=0.1)
                
                # Process frame
                start_time = time.time()
                potholes, annotated_frame = self.detector.detect_potholes(frame)
                processing_time = time.time() - start_time
                
                # Update FPS
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Add processing info to frame
                annotated_frame = self.add_ui_overlay(annotated_frame, potholes, processing_time)
                
                # Store result
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        break
                self.result_queue.put((annotated_frame, potholes))
                
                # Process detections
                if potholes:
                    self.process_detections(potholes, frame)
                    
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")
                time.sleep(0.01)
    
    def display_frames(self):
        """Display processed frames in a separate thread"""
        while self.running:
            try:
                if self.result_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                annotated_frame, potholes = self.result_queue.get(timeout=0.1)
                
                # Display frame
                cv2.imshow('Live Pothole Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                elif key == ord('g'):
                    self.show_gps = not self.show_gps
                elif key == ord('d'):
                    self.show_depth = not self.show_depth
                elif key == ord('s'):
                    self.save_current_frame(annotated_frame)
                    
            except Exception as e:
                logger.error(f"Error in frame display: {e}")
                time.sleep(0.01)
    
    def add_ui_overlay(self, frame: np.ndarray, potholes: List[Pothole], processing_time: float) -> np.ndarray:
        """Add UI overlay with information to the frame"""
        overlay = frame.copy()
        
        # Add FPS counter
        if self.show_fps:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(overlay, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add processing time
        proc_text = f"Process: {processing_time*1000:.1f}ms"
        cv2.putText(overlay, proc_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add detection count
        det_text = f"Detections: {len(potholes)}"
        cv2.putText(overlay, det_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add total detections
        total_text = f"Total: {self.total_detections}"
        cv2.putText(overlay, total_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add controls info
        controls_text = "Controls: Q=Quit, C=Confidence, F=FPS, G=GPS, D=Depth, S=Save"
        cv2.putText(overlay, controls_text, (10, overlay.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def process_detections(self, potholes: List[Pothole], original_frame: np.ndarray):
        """Process detected potholes"""
        for pothole in potholes:
            self.total_detections += 1
            
            # Check for duplicates
            if not self.db.is_duplicate(pothole.latitude, pothole.longitude):
                try:
                    pothole_id = self.db.add_pothole(pothole)
                    if pothole_id:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        image_path = save_detection_image(original_frame, pothole_id, timestamp)
                        
                        # Add to history
                        self.detection_history.append({
                            'id': pothole_id,
                            'severity': pothole.severity.value,
                            'depth': pothole.depth,
                            'timestamp': datetime.now(),
                            'location': (pothole.latitude, pothole.longitude)
                        })
                        
                        # Keep only last 100 detections in history
                        if len(self.detection_history) > 100:
                            self.detection_history.pop(0)
                        
                        logger.info(f"New pothole detected: ID={pothole_id}, "
                                  f"Severity={pothole.severity.value}, "
                                  f"Depth={pothole.depth:.3f}m")
                        
                except Exception as e:
                    logger.error(f"Database error: {e}")
                    self.db.save_offline_log([pothole])
    
    def save_current_frame(self, frame: np.ndarray):
        """Save current frame as image"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"live_capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Frame saved as {filename}")
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        if not self.detection_history:
            return {}
        
        severities = [d['severity'] for d in self.detection_history]
        depths = [d['depth'] for d in self.detection_history]
        
        return {
            'total_detections': self.total_detections,
            'recent_detections': len(self.detection_history),
            'severity_distribution': {
                'low': severities.count('low'),
                'medium': severities.count('medium'),
                'high': severities.count('high'),
                'critical': severities.count('critical')
            },
            'avg_depth': np.mean(depths) if depths else 0,
            'max_depth': np.max(depths) if depths else 0,
            'current_fps': self.current_fps
        }
    
    def start(self, camera_index: int = 0):
        """Start live detection"""
        if not self.start_camera(camera_index):
            return False
        
        self.running = True
        
        # Start threads
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.display_thread = threading.Thread(target=self.display_frames)
        
        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.start()
        
        # Start processing and display threads
        self.processing_thread.start()
        self.display_thread.start()
        
        logger.info("Live detection started")
        return True
    
    def stop(self):
        """Stop live detection"""
        self.running = False
        
        # Wait for threads to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        if self.display_thread:
            self.display_thread.join(timeout=2)
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Create detection summary
        summary_file = self.detector.create_detection_summary()
        if summary_file:
            print(f"📊 Detection summary saved: {summary_file}")
        
        logger.info("Live detection stopped")
        print("✅ Live detection stopped")
    
    def run_interactive(self, camera_index: int = 0):
        """Run live detection with interactive controls"""
        if not self.start(camera_index):
            return
        
        try:
            while self.running:
                time.sleep(0.1)
                
                # Print stats every 10 seconds
                if int(time.time()) % 10 == 0:
                    stats = self.get_detection_stats()
                    if stats:
                        logger.info(f"Stats: {stats}")
                        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()


def main():
    """Main function for testing live detection"""
    live_detector = LivePotholeDetector()
    live_detector.run_interactive()


if __name__ == "__main__":
    main() 