import os
import math
import cv2
import numpy as np
from config import config


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters"""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def save_detection_image(image: np.ndarray, pothole_id: int, timestamp: str) -> str:
    """Save detection image and return the path"""
    filename = f"pothole_{pothole_id}_{timestamp}.jpg"
    filepath = os.path.join(config.DATA_DIR, 'images', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, image)
    return filepath


def create_depth_report(potholes: list) -> dict:
    """Create a depth analysis report"""
    if not potholes:
        return {}

    depths = [p.depth for p in potholes]
    return {
        'average_depth': np.mean(depths),
        'max_depth': np.max(depths),
        'min_depth': np.min(depths),
        'std_depth': np.std(depths),
        'depth_distribution': {
            'shallow': len([d for d in depths if d < 0.02]),
            'moderate': len([d for d in depths if 0.02 <= d < 0.05]),
            'deep': len([d for d in depths if 0.05 <= d < 0.10]),
            'very_deep': len([d for d in depths if d >= 0.10])
        }
    }
