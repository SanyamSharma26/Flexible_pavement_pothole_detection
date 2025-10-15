import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import logging
from datetime import datetime
from typing import Tuple, List, Dict
import os
import sys

from config import config
from models import Pothole, Severity

logger = logging.getLogger(__name__)

class PotholeOnlyDetector:
    """
    Specialized detector that only detects potholes with additional validation
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load YOLO model
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
        
        # Pothole-specific detection parameters
        self.confidence_threshold = 0.7  # Higher confidence for potholes
        self.min_area = 200  # Minimum pothole area in pixels
        self.max_area = 50000  # Maximum pothole area in pixels
        self.aspect_ratio_range = (0.3, 3.0)  # Valid aspect ratios for potholes
        
    def is_valid_pothole_shape(self, contour, area):
        """Validate if the detected object has pothole-like characteristics"""
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                return False
            
            # Check area constraints
            if not (self.min_area <= area <= self.max_area):
                return False
            
            # Check if shape is roughly circular/elliptical (potholes are usually round)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.3:  # Too irregular
                    return False
            
            # Check if it's not too elongated
            if w > 0 and h > 0:
                elongation = max(w, h) / min(w, h)
                if elongation > 4:  # Too elongated
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in shape validation: {e}")
            return False
    
    def validate_pothole_location(self, x1, y1, x2, y2, image_shape):
        """Validate if the detection is in a reasonable road area"""
        h, w = image_shape[:2]
        
        # Potholes should be in the lower part of the image (road surface)
        road_zone_y = h * 0.4  # Road starts at 40% from top
        if y1 < road_zone_y:
            return False
        
        # Potholes shouldn't be at the very edges
        margin = 50
        if x1 < margin or x2 > w - margin or y2 > h - margin:
            return False
        
        return True
    
    def analyze_texture(self, image, mask):
        """Analyze texture to confirm it's a road surface defect"""
        try:
            # Apply mask to get the detected region
            masked_region = cv2.bitwise_and(image, image, mask=mask)
            
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture features
            # Standard deviation (roughness)
            std_dev = np.std(gray[gray > 0])
            
            # Mean intensity
            mean_intensity = np.mean(gray[gray > 0])
            
            # Potholes typically have:
            # - Lower mean intensity (darker)
            # - Higher standard deviation (more texture variation)
            
            # Simple validation based on typical pothole characteristics
            if mean_intensity > 150:  # Too bright (might be reflection)
                return False
            
            if std_dev < 20:  # Too uniform (might be smooth surface)
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in texture analysis: {e}")
            return True  # Default to True if analysis fails
    
    def detect_potholes(self, image: np.ndarray, gps_data: Dict = None) -> Tuple[List[Pothole], np.ndarray]:
        """
        Detect ONLY potholes with strict validation
        """
        h, w = image.shape[:2]
        
        # Use higher confidence threshold for pothole detection
        results = self.yolo_model.predict(image, conf=self.confidence_threshold)
        potholes = []
        annotated_image = image.copy()
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for idx, (mask, box) in enumerate(zip(masks, boxes)):
                    # Resize mask to original image size
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Get the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Apply strict validation filters
                        if not self.is_valid_pothole_shape(largest_contour, area):
                            logger.debug(f"Rejected: Invalid shape (area={area:.0f})")
                            continue
                        
                        if not self.validate_pothole_location(x1, y1, x2, y2, image.shape):
                            logger.debug(f"Rejected: Invalid location")
                            continue
                        
                        if not self.analyze_texture(image, mask_binary):
                            logger.debug(f"Rejected: Invalid texture")
                            continue
                        
                        # Additional validation: Check if it's not a shadow or reflection
                        if self.is_likely_shadow_or_reflection(image, x1, y1, x2, y2):
                            logger.debug(f"Rejected: Likely shadow/reflection")
                            continue
                        
                        # If we reach here, it's likely a pothole
                        logger.info(f"Valid pothole detected: area={area:.0f}, conf={float(box.conf[0]):.2f}")
                        
                        # Estimate depth (simplified for now)
                        avg_depth = self.estimate_pothole_depth(area)
                        
                        # Calculate severity
                        severity = self.calculate_severity(area, avg_depth)
                        
                        # Create Pothole object
                        pothole = Pothole(
                            latitude=gps_data.get('latitude', 0.0) if gps_data else 0.0,
                            longitude=gps_data.get('longitude', 0.0) if gps_data else 0.0,
                            city=gps_data.get('city', 'Unknown') if gps_data else 'Unknown',
                            region=gps_data.get('region', 'Unknown') if gps_data else 'Unknown',
                            severity=severity,
                            area=area,
                            depth=avg_depth,
                            confidence=float(box.conf[0]),
                            timestamp=datetime.now()
                        )
                        potholes.append(pothole)
                        
                        # Annotate image
                        color = self._get_severity_color(severity)
                        
                        # Draw contour
                        cv2.drawContours(annotated_image, [largest_contour], -1, color, 2)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Add text annotation
                        label = f"POTHOLE {severity.value.upper()}"
                        depth_label = f"Depth: {avg_depth*100:.1f}cm"
                        conf_label = f"Conf: {float(box.conf[0]):.2f}"
                        
                        cv2.putText(annotated_image, label, (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(annotated_image, depth_label, (x1, y1 - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.putText(annotated_image, conf_label, (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return potholes, annotated_image
    
    def is_likely_shadow_or_reflection(self, image, x1, y1, x2, y2):
        """Check if detection might be a shadow or reflection"""
        try:
            # Extract the region
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                return False
            
            # Convert to HSV for better shadow detection
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Calculate saturation and value
            saturation = np.mean(hsv[:, :, 1])
            value = np.mean(hsv[:, :, 2])
            
            # Shadows typically have low saturation and low value
            if saturation < 30 and value < 80:
                return True
            
            # Reflections typically have high value and low saturation
            if value > 200 and saturation < 50:
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error in shadow/reflection detection: {e}")
            return False
    
    def estimate_pothole_depth(self, area):
        """Estimate pothole depth based on area (simplified)"""
        # Simple depth estimation based on area
        # Larger potholes tend to be deeper
        area_factor = min(area / 1000, 2.0)  # Normalize area
        base_depth = 0.03  # 3cm base depth
        depth = base_depth + (area_factor * 0.05)  # Add up to 10cm based on area
        return min(depth, 0.15)  # Cap at 15cm
    
    def calculate_severity(self, area: float, depth: float) -> Severity:
        """Calculate severity based on area and depth"""
        depth_cm = depth * 100
        
        # Simplified scoring
        area_score = min(area / 1000, 50)  # Area score up to 50
        depth_score = min(depth_cm * 5, 50)  # Depth score up to 50
        
        total_score = area_score + depth_score
        
        if total_score < 25:
            return Severity.LOW
        elif total_score < 50:
            return Severity.MEDIUM
        elif total_score < 75:
            return Severity.HIGH
        else:
            return Severity.CRITICAL
    
    def _get_severity_color(self, severity: Severity) -> Tuple[int, int, int]:
        """Get color based on severity"""
        colors = {
            Severity.LOW: (0, 255, 0),      # Green
            Severity.MEDIUM: (0, 255, 255), # Yellow
            Severity.HIGH: (0, 165, 255),   # Orange
            Severity.CRITICAL: (0, 0, 255)  # Red
        }
        return colors.get(severity, (255, 255, 255)) 