import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import os
import sys
from scipy import ndimage
from sklearn.cluster import KMeans

from config import config
from models import Pothole, Severity

logger = logging.getLogger(__name__)

class EnhancedPotholeDetector:
    """
    Advanced pothole-only detector with multiple validation layers and real-world measurements
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load YOLO model
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
        
        # Initialize text file logging
        self.log_file = f"pothole_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.setup_text_logging()
        
        # Enhanced detection parameters
        self.confidence_threshold = 0.6  # Balanced confidence threshold
        self.min_area = 200  # Minimum pothole area in pixels
        self.max_area = 50000  # Maximum pothole area in pixels
        self.aspect_ratio_range = (0.3, 3.0)  # Aspect ratio range
        
        # Texture analysis parameters
        self.texture_std_min = 20  # Minimum texture variation
        self.texture_mean_max = 150  # Maximum mean intensity
        
        # Edge density parameters
        self.min_edge_density = 0.05  # Minimum edge density
        self.max_edge_density = 0.9  # Maximum edge density
        
        # Color analysis parameters
        self.dark_threshold = 120  # Potholes should be darker
        
        # Calibration parameters for real-world measurements
        self.pixels_per_cm = 10.0  # Default calibration (10 pixels = 1 cm)
        self.calibration_distance = 100.0  # Distance from camera in cm
        self.focal_length = 1000.0  # Camera focal length in pixels
        self.camera_height = 150.0  # Camera height from ground in cm
        
    def setup_text_logging(self):
        """Setup text file logging for detections"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("POTHOLE DETECTION LOG\n")
                f.write("=" * 50 + "\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Format: Timestamp | Area(cm²) | Width×Height(cm) | Depth(cm) | Severity | Confidence | GPS\n")
                f.write("-" * 50 + "\n")
            logger.info(f"Detection log file created: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to create log file: {e}")
    
    def log_detection_to_file(self, pothole: Pothole, area_cm2: float, width_cm: float, height_cm: float, depth_cm: float, confidence: float):
        """Log detection details to text file"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            gps_info = f"{pothole.latitude:.6f},{pothole.longitude:.6f}" if pothole.latitude != 0.0 else "N/A"
            
            log_line = f"{timestamp} | {area_cm2:.1f}cm² | {width_cm:.1f}×{height_cm:.1f}cm | {depth_cm:.1f}cm | {pothole.severity.value} | {confidence:.2f} | {gps_info}\n"
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
                
        except Exception as e:
            logger.error(f"Failed to log detection: {e}")
    
    def calibrate_camera(self, image: np.ndarray, reference_width_pixels: float) -> bool:
        """
        Calibrate camera using a known reference object
        Args:
            image: Input image
            reference_width_pixels: Width of reference object in pixels
        """
        try:
            # Calculate pixels per cm based on reference object
            self.pixels_per_cm = reference_width_pixels / self.calibration_reference_cm
            
            # Calculate focal length for depth estimation
            self.focal_length = reference_width_pixels * self.calibration_distance / self.calibration_reference_cm
            
            logger.info(f"Camera calibrated: {self.pixels_per_cm:.2f} pixels/cm, "
                       f"focal length: {self.focal_length:.1f} pixels")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def pixels_to_cm(self, pixels: float) -> float:
        """Convert pixels to centimeters"""
        return pixels / self.pixels_per_cm
    
    def cm_to_pixels(self, cm: float) -> float:
        """Convert centimeters to pixels"""
        return cm * self.pixels_per_cm
    
    def estimate_distance_from_camera(self, object_width_pixels: float, real_width_cm: float) -> float:
        """
        Estimate distance from camera using object size
        Args:
            object_width_pixels: Width of object in pixels
            real_width_cm: Real width of object in cm
        Returns:
            Distance in cm
        """
        if object_width_pixels > 0:
            return (real_width_cm * self.focal_length) / object_width_pixels
        return self.calibration_distance  # Default distance
    
    def calculate_real_area_cm2(self, pixel_area: float, distance_cm: float = None) -> float:
        """
        Calculate real area in square centimeters
        Args:
            pixel_area: Area in pixels
            distance_cm: Distance from camera (optional, uses default if not provided)
        Returns:
            Area in square centimeters
        """
        if distance_cm is None:
            distance_cm = self.calibration_distance
        
        # Convert pixel area to cm²
        # Area scales with distance squared
        scale_factor = (distance_cm / self.calibration_distance) ** 2
        area_cm2 = (pixel_area / (self.pixels_per_cm ** 2)) * scale_factor
        
        return area_cm2
    
    def calculate_real_dimensions_cm(self, width_pixels: float, height_pixels: float, 
                                   distance_cm: float = None) -> Tuple[float, float]:
        """
        Calculate real dimensions in centimeters
        Args:
            width_pixels: Width in pixels
            height_pixels: Height in pixels
            distance_cm: Distance from camera (optional)
        Returns:
            Tuple of (width_cm, height_cm)
        """
        if distance_cm is None:
            distance_cm = self.calibration_distance
        
        # Convert to cm
        width_cm = self.pixels_to_cm(width_pixels)
        height_cm = self.pixels_to_cm(height_pixels)
        
        # Adjust for distance
        scale_factor = distance_cm / self.calibration_distance
        width_cm *= scale_factor
        height_cm *= scale_factor
        
        return width_cm, height_cm
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better detection"""
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def analyze_edge_density(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Analyze edge density in the detected region"""
        try:
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask=mask)
            
            # Convert to grayscale
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges using Canny
            edges = cv2.Canny(blurred, 50, 150)
            
            # Calculate edge density
            total_pixels = np.sum(mask > 0)
            edge_pixels = np.sum(edges > 0)
            
            if total_pixels > 0:
                edge_density = edge_pixels / total_pixels
                return edge_density
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error in edge density analysis: {e}")
            return 0.0
    
    def analyze_color_distribution(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Analyze color distribution in the detected region"""
        try:
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask=mask)
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
            
            # Get non-zero pixels
            non_zero = mask > 0
            
            if not np.any(non_zero):
                return {"is_dark": False, "color_variance": 0}
            
            # Analyze brightness (L channel in LAB)
            l_channel = lab[:,:,0][non_zero]
            mean_brightness = np.mean(l_channel)
            brightness_std = np.std(l_channel)
            
            # Analyze saturation (S channel in HSV)
            s_channel = hsv[:,:,1][non_zero]
            mean_saturation = np.mean(s_channel)
            
            # Analyze hue (H channel in HSV)
            h_channel = hsv[:,:,0][non_zero]
            hue_variance = np.var(h_channel)
            
            return {
                "is_dark": mean_brightness < self.dark_threshold,
                "mean_brightness": mean_brightness,
                "brightness_std": brightness_std,
                "mean_saturation": mean_saturation,
                "hue_variance": hue_variance,
                "color_variance": brightness_std + hue_variance
            }
            
        except Exception as e:
            logger.debug(f"Error in color analysis: {e}")
            return {"is_dark": False, "color_variance": 0}
    
    def analyze_texture_patterns(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Advanced texture pattern analysis"""
        try:
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            
            # Get non-zero pixels
            non_zero = gray[mask > 0]
            
            if len(non_zero) == 0:
                return {"texture_score": 0, "is_irregular": False}
            
            # Calculate texture features
            mean_intensity = np.mean(non_zero)
            std_intensity = np.std(non_zero)
            
            # Calculate local binary pattern (simplified)
            # Apply different filters for texture analysis
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_magnitude_masked = gradient_magnitude[mask > 0]
            
            mean_gradient = np.mean(gradient_magnitude_masked)
            
            # Calculate texture irregularity
            # Potholes should have irregular texture patterns
            texture_score = (std_intensity * mean_gradient) / (mean_intensity + 1)
            
            # Check if texture is irregular enough
            is_irregular = (std_intensity > self.texture_std_min and 
                          mean_intensity < self.texture_mean_max and
                          texture_score > 50)
            
            return {
                "texture_score": texture_score,
                "is_irregular": is_irregular,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "mean_gradient": mean_gradient
            }
            
        except Exception as e:
            logger.debug(f"Error in texture analysis: {e}")
            return {"texture_score": 0, "is_irregular": False}
    
    def validate_pothole_geometry(self, contour: np.ndarray, area: float) -> bool:
        """Enhanced geometric validation"""
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
            
            # Calculate more geometric features
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Circularity
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Convexity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Potholes should be somewhat circular but not perfect circles
                if circularity < 0.2 or circularity > 0.9:
                    return False
                
                # Potholes should have good solidity (not too irregular)
                if solidity < 0.6:
                    return False
            
            # Check elongation
            if w > 0 and h > 0:
                elongation = max(w, h) / min(w, h)
                if elongation > 3:  # Too elongated
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in geometry validation: {e}")
            return False
    
    def validate_road_location(self, x1: int, y1: int, x2: int, y2: int, 
                             image_shape: Tuple[int, int, int]) -> bool:
        """Enhanced road location validation"""
        h, w = image_shape[:2]
        
        # Potholes should be in the road area (lower 60% of image)
        road_zone_y = h * 0.4
        if y1 < road_zone_y:
            return False
        
        # Potholes shouldn't be at the very edges
        margin = 30
        if x1 < margin or x2 > w - margin or y2 > h - margin:
            return False
        
        # Potholes should be roughly in the center area (not extreme left/right)
        center_margin = w * 0.1
        if x1 < center_margin or x2 > w - center_margin:
            return False
        
        return True
    
    def detect_shadows_and_reflections(self, image: np.ndarray, x1: int, y1: int, 
                                     x2: int, y2: int) -> bool:
        """Enhanced shadow and reflection detection"""
        try:
            # Extract the region
            region = image[y1:y2, x1:x2]
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            
            # Check for reflection characteristics
            # Reflections are usually bright and have low saturation
            mean_saturation = np.mean(hsv[:,:,1])
            mean_brightness = np.mean(lab[:,:,0])
            
            # Check for shadow characteristics
            # Shadows are usually dark but have similar color to surroundings
            b, g, r = cv2.split(region)
            color_variance = np.var(b) + np.var(g) + np.var(r)
            
            # Reflection detection
            is_reflection = (mean_brightness > 180 and mean_saturation < 50)
            
            # Shadow detection
            is_shadow = (mean_brightness < 80 and color_variance < 100)
            
            return is_reflection or is_shadow
            
        except Exception as e:
            logger.debug(f"Error in shadow/reflection detection: {e}")
            return False
    
    def detect_potholes(self, image: np.ndarray, gps_data: Dict = None) -> Tuple[List[Pothole], np.ndarray]:
        """
        Enhanced pothole detection with multiple validation layers
        """
        h, w = image.shape[:2]
        
        # Preprocess image
        enhanced_image = self.preprocess_image(image)
        
        # Run YOLO detection with higher confidence
        results = self.yolo_model.predict(enhanced_image, conf=self.confidence_threshold)
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
                        
                        # Apply comprehensive validation
                        if not self.validate_pothole_geometry(largest_contour, area):
                            logger.debug(f"Rejected: Invalid geometry (area={area:.0f})")
                            continue
                        
                        if not self.validate_road_location(x1, y1, x2, y2, image.shape):
                            logger.debug(f"Rejected: Invalid location")
                            continue
                        
                        # Texture analysis (optional validation)
                        texture_info = self.analyze_texture_patterns(image, mask_binary)
                        texture_score = texture_info.get("texture_score", 0)
                        
                        # Color analysis (optional validation)
                        color_info = self.analyze_color_distribution(image, mask_binary)
                        is_dark = color_info.get("is_dark", True)
                        
                        # Edge density analysis (optional validation)
                        edge_density = self.analyze_edge_density(image, mask_binary)
                        
                        # Shadow/reflection detection
                        is_shadow_reflection = self.detect_shadows_and_reflections(image, x1, y1, x2, y2)
                        
                        # Combined validation score
                        validation_score = 0
                        if texture_info.get("is_irregular", False):
                            validation_score += 1
                        if is_dark:
                            validation_score += 1
                        if self.min_edge_density <= edge_density <= self.max_edge_density:
                            validation_score += 1
                        if not is_shadow_reflection:
                            validation_score += 1
                        
                        # Require at least 2 out of 4 validation criteria to pass
                        if validation_score < 2:
                            logger.debug(f"Rejected: Low validation score ({validation_score}/4)")
                            continue
                        
                        # If we reach here, it's a valid pothole
                        confidence = float(box.conf[0])
                        
                        # Calculate real-world measurements
                        width_pixels = x2 - x1
                        height_pixels = y2 - y1
                        
                        # Calculate real dimensions in cm
                        width_cm, height_cm = self.calculate_real_dimensions_cm(width_pixels, height_pixels)
                        
                        # Calculate real area in cm²
                        area_cm2 = self.calculate_real_area_cm2(area)
                        
                        # Estimate distance from camera
                        distance_cm = self.estimate_distance_from_camera(width_pixels, width_cm)
                        
                        # Estimate depth in cm
                        avg_depth_cm = self.estimate_pothole_depth_cm(area_cm2, texture_info)
                        
                        logger.info(f"✅ Valid pothole detected: area={area_cm2:.1f}cm², "
                                  f"dimensions={width_cm:.1f}x{height_cm:.1f}cm, "
                                  f"conf={confidence:.2f}, depth={avg_depth_cm:.1f}cm")
                        
                        # Calculate severity based on real measurements
                        severity = self.calculate_severity_cm(area_cm2, avg_depth_cm, confidence, width_cm)
                        
                        # Create Pothole object with real measurements
                        pothole = Pothole(
                            latitude=gps_data.get('latitude', 0.0) if gps_data else 0.0,
                            longitude=gps_data.get('longitude', 0.0) if gps_data else 0.0,
                            city=gps_data.get('city', 'Unknown') if gps_data else 'Unknown',
                            region=gps_data.get('region', 'Unknown') if gps_data else 'Unknown',
                            severity=severity,
                            area=area_cm2,  # Now in cm²
                            depth=avg_depth_cm,  # Now in cm
                            confidence=confidence,
                            timestamp=datetime.now()
                        )
                        potholes.append(pothole)
                        
                        # Annotate image with enhanced information
                        color = self._get_severity_color(severity)
                        
                        # Draw contour
                        cv2.drawContours(annotated_image, [largest_contour], -1, color, 3)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Add detailed annotation with real measurements
                        label = f"Pothole: {severity.value} (Conf: {confidence:.2f})"
                        cv2.putText(annotated_image, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Add real-world measurements
                        info_text = f"Area: {area_cm2:.1f}cm², Depth: {avg_depth_cm:.1f}cm"
                        cv2.putText(annotated_image, info_text, (x1, y2+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Add dimensions
                        dim_text = f"Size: {width_cm:.1f}x{height_cm:.1f}cm"
                        cv2.putText(annotated_image, dim_text, (x1, y2+40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Log detection to text file
                        self.log_detection_to_file(pothole, area_cm2, width_cm, height_cm, avg_depth_cm, confidence)
        
        return potholes, annotated_image
    
    def estimate_pothole_depth_cm(self, area_cm2: float, texture_info: Dict) -> float:
        """Enhanced depth estimation in centimeters based on area and texture"""
        # Base depth calculation in cm - more realistic scaling
        base_depth_cm = np.sqrt(area_cm2) * 0.03  # Reduced scale factor for more realistic depths
        
        # Adjust based on texture irregularity
        texture_factor = min(texture_info.get("texture_score", 0) / 100, 1.5)
        
        # Adjust based on area (larger potholes tend to be deeper)
        area_factor = min(area_cm2 / 500, 1.2)  # Normalize for cm²
        
        estimated_depth_cm = base_depth_cm * texture_factor * area_factor
        
        # Clamp to reasonable range (0.5-15 cm) - more realistic for road potholes
        return max(0.5, min(15.0, estimated_depth_cm))
    
    def calculate_severity_cm(self, area_cm2: float, depth_cm: float, confidence: float, width_cm: float = None) -> Severity:
        """Enhanced severity calculation based on new real-world measurements"""
        # Convert mm to cm for comparison
        # Small: depth <= 2.5cm and width <= 20cm
        # Medium: 2.5cm < depth <= 5cm and width <= 50cm
        # Large: depth > 5cm or width > 50cm
        
        # If width_cm is not provided, estimate from area
        if width_cm is None:
            width_cm = area_cm2 ** 0.5
        
        if depth_cm <= 2.5 and width_cm <= 20:
            return Severity.LOW  # Small
        elif 2.5 < depth_cm <= 5 and width_cm <= 50:
            return Severity.MEDIUM  # Medium
        else:
            return Severity.HIGH  # Large
    
    def estimate_pothole_depth(self, area: float, texture_info: Dict) -> float:
        """Legacy method - kept for compatibility"""
        area_cm2 = self.calculate_real_area_cm2(area)
        return self.estimate_pothole_depth_cm(area_cm2, texture_info)
    
    def calculate_severity(self, area: float, depth: float, confidence: float) -> Severity:
        """Legacy method - kept for compatibility"""
        area_cm2 = self.calculate_real_area_cm2(area)
        return self.calculate_severity_cm(area_cm2, depth, confidence)
    
    def _get_severity_color(self, severity: Severity) -> Tuple[int, int, int]:
        """Get color for severity level"""
        if severity == Severity.HIGH:
            return (0, 0, 255)  # Red
        elif severity == Severity.MEDIUM:
            return (0, 165, 255)  # Orange
        else:
            return (0, 255, 0)  # Green
    
    def create_detection_summary(self):
        """Create a summary report of all detections"""
        try:
            summary_file = f"pothole_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            # Read the log file and create summary
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Parse detections
                detections = []
                for line in lines[5:]:  # Skip header lines
                    if line.strip() and '|' in line:
                        parts = line.strip().split('|')
                        if len(parts) >= 7:
                            detections.append({
                                'timestamp': parts[0].strip(),
                                'area': float(parts[1].strip().replace('cm²', '')),
                                'dimensions': parts[2].strip(),
                                'depth': float(parts[3].strip().replace('cm', '')),
                                'severity': parts[4].strip(),
                                'confidence': float(parts[5].strip()),
                                'gps': parts[6].strip()
                            })
                
                # Create summary
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write("POTHOLE DETECTION SUMMARY REPORT\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Detections: {len(detections)}\n\n")
                    
                    # Severity breakdown
                    severity_counts = {}
                    total_area = 0
                    total_depth = 0
                    avg_confidence = 0
                    
                    for det in detections:
                        severity = det['severity']
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                        total_area += det['area']
                        total_depth += det['depth']
                        avg_confidence += det['confidence']
                    
                    f.write("SEVERITY BREAKDOWN:\n")
                    f.write("-" * 20 + "\n")
                    for severity, count in severity_counts.items():
                        percentage = (count / len(detections)) * 100
                        f.write(f"{severity}: {count} ({percentage:.1f}%)\n")
                    
                    f.write(f"\nSTATISTICS:\n")
                    f.write("-" * 20 + "\n")
                    if len(detections) > 0:
                        f.write(f"Average Area: {total_area/len(detections):.1f} cm²\n")
                        f.write(f"Average Depth: {total_depth/len(detections):.1f} cm\n")
                        f.write(f"Average Confidence: {avg_confidence/len(detections):.2f}\n")
                    else:
                        f.write("No detections recorded\n")
                    
                    f.write(f"\nDETAILED DETECTIONS:\n")
                    f.write("-" * 20 + "\n")
                    for i, det in enumerate(detections, 1):
                        f.write(f"{i}. {det['timestamp']} - {det['area']:.1f}cm² - {det['severity']} - Conf: {det['confidence']:.2f}\n")
                
                logger.info(f"Summary report created: {summary_file}")
                return summary_file
                
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            return None 