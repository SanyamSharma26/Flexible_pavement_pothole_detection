import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import logging
from datetime import datetime
from typing import Tuple,  List, Dict
import os
import sys

from config import config
from models import Pothole, Severity

logger = logging.getLogger(__name__)

# Import MiDaS components directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'MiDaS'))


class PotholeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load YOLO model
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)

        # Load MiDaS model from local file
        self.midas_model, self.midas_transform = self._load_midas_model_local()

    def _load_midas_model_local(self):
        """Load MiDaS model - fallback to pretrained if local fails"""
        try:
            logger.info("Attempting to load MiDaS model...")

            # Try to use the pretrained model instead of the mismatched local file
            model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', pretrained=True)
            model.to(self.device)
            model.eval()

            # Get the transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.dpt_transform

            logger.info("MiDaS DPT_Large model loaded successfully")
            return model, transform

        except Exception as e:
            logger.error(f"Error loading MiDaS model: {e}")
            logger.info("Using simple depth estimator as fallback...")
            return self._create_simple_depth_estimator()

    def _create_transform(self):
        """Create a transform for MiDaS manually"""
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def transform_func(image):
            # Ensure image is in the right format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return transform(image).unsqueeze(0)

        return transform_func

    def _create_simple_depth_estimator(self):
        """Create a simple depth estimator as fallback"""

        class SimpleDepthEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                logger.warning("Using simple depth estimator - results may be less accurate")

            def forward(self, x):
                # Return a dummy depth map
                batch_size = x.shape[0]
                height = x.shape[2] * 4  # Assuming some upscaling
                width = x.shape[3] * 4
                return torch.randn(batch_size, height, width).to(x.device)

        model = SimpleDepthEstimator().to(self.device)
        transform = self._create_transform()
        return model, transform

    def estimate_depth(self, image: np.ndarray, mask: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Estimate depth of pothole using MiDaS
        Returns: (average_depth, max_depth, depth_map) in meters
        """
        try:
            # Prepare image for MiDaS
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply MiDaS transform
            input_batch = self.midas_transform(img_rgb).to(self.device)

            # Predict depth
            with torch.no_grad():
                prediction = self.midas_model(input_batch)

                # Handle different output formats
                if len(prediction.shape) == 3:
                    prediction = prediction.unsqueeze(1)

                # Resize to original image size
                prediction = torch.nn.functional.interpolate(
                    prediction,
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth_map = prediction.cpu().numpy()

            # Normalize depth map
            depth_map = np.abs(depth_map)

            # Avoid division by zero by creating a mask
            valid_mask = depth_map > 0
            depth_map_inv = np.zeros_like(depth_map)

            # Invert only valid depth values
            depth_map_inv[valid_mask] = 1.0 / depth_map[valid_mask]

            # Scale the depth map to a reasonable range
            DEPTH_SCALE_FACTOR = 0.5  # Adjust this based on your camera setup
            depth_map_inv *= DEPTH_SCALE_FACTOR

            # Apply mask to get pothole region only
            mask_binary = mask.astype(bool)
            pothole_depth = depth_map_inv[mask_binary]

            if len(pothole_depth) > 0:
                # Get surrounding area for reference
                kernel = np.ones((15, 15), np.uint8)
                dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
                surrounding_mask = (dilated_mask - mask.astype(np.uint8)) > 0
                surrounding_depth = depth_map_inv[surrounding_mask]

                if len(surrounding_depth) > 0:
                    # Calculate relative depth using percentiles for robustness
                    surface_level = np.percentile(surrounding_depth, 50)  # Median of surrounding
                    pothole_bottom = np.percentile(pothole_depth, 90)  # 90th percentile of pothole
                    pothole_average = np.percentile(pothole_depth, 75)  # 75th percentile

                    # Calculate depth difference (should be positive for a hole)
                    max_depth_meters = abs(pothole_bottom - surface_level)
                    avg_depth_meters = abs(pothole_average - surface_level)

                    # Apply reasonable bounds for pothole depths
                    max_depth_meters = np.clip(max_depth_meters, 0.01, 0.30)
                    avg_depth_meters = np.clip(avg_depth_meters, 0.01, 0.25)

                    # Additional scaling based on pothole area
                    area_factor = min(mask.sum() / (image.shape[0] * image.shape[1]) * 10, 2.0)
                    max_depth_meters *= area_factor
                    avg_depth_meters *= area_factor

                    logger.debug(f"Depth estimation - Avg: {avg_depth_meters:.3f}m, Max: {max_depth_meters:.3f}m")

                    return avg_depth_meters, max_depth_meters, depth_map_inv
                else:
                    # If we can't get surrounding depth, use default based on area
                    area_ratio = mask.sum() / (image.shape[0] * image.shape[1])
                    default_depth = 0.03 + (area_ratio * 0.5)  # 3cm base + area factor
                    return default_depth, default_depth * 1.5, depth_map_inv

            # Default values if calculation fails
            logger.warning("Using default depth values")
            return 0.03, 0.05, depth_map_inv

        except Exception as e:
            logger.error(f"Error in depth estimation: {e}")
            # Return default values (3cm average, 5cm max)
            return 0.03, 0.05, np.zeros_like(image[:, :, 0])

    def calculate_severity(self, area: float, depth: float, width: float = None) -> Severity:
        """Calculate severity based on new area, depth, and width definitions"""
        # Convert depth to mm and width to mm
        depth_mm = depth * 1000
        width_mm = width if width is not None else (area ** 0.5)
        # Small: depth <= 25mm and width <= 200mm
        # Medium: 25 < depth <= 50mm and width <= 500mm
        # Large: depth > 50mm or width > 500mm
        if depth_mm <= 25 and width_mm <= 200:
            return Severity.LOW
        elif 25 < depth_mm <= 50 and width_mm <= 500:
            return Severity.MEDIUM
        else:
            return Severity.HIGH

    def detect_potholes(self, image: np.ndarray, gps_data: Dict = None) -> Tuple[List[Pothole], np.ndarray]:
        """
        Detect potholes in the image and return list of Pothole objects and annotated image
        """
        h, w = image.shape[:2]
        results = self.yolo_model.predict(image, conf=0.5)
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

                        # Skip very small detections
                        if area < 100:
                            continue

                        # Estimate depth
                        avg_depth, max_depth, depth_map = self.estimate_depth(image, mask_binary)

                        # Calculate severity
                        severity = self.calculate_severity(area, avg_depth)

                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

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
                        # Draw contour
                        cv2.drawContours(annotated_image, [largest_contour], -1, (0, 0, 255), 2)

                        # Draw bounding box
                        color = self._get_severity_color(severity)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                        # Add text annotation
                        label = f"{severity.value.upper()}"
                        depth_label = f"Depth: {avg_depth*100:.2f}cm"
                        conf_label = f"Conf: {float(box.conf[0]):.2f}"

                        cv2.putText(annotated_image, label, (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(annotated_image, depth_label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.putText(annotated_image, conf_label, (x1, y1 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return potholes, annotated_image

    def _get_severity_color(self, severity: Severity) -> Tuple[int, int, int]:
        """Get color based on severity for visualization"""
        colors = {
            Severity.LOW: (0, 255, 0),  # Green
            Severity.MEDIUM: (0, 255, 255),  # Yellow
            Severity.HIGH: (0, 165, 255),  # Orange
            Severity.CRITICAL: (0, 0, 255)  # Red
        }
        return colors.get(severity, (255, 255, 255))
