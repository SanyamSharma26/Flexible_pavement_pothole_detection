from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"



@dataclass
class Pothole:
    latitude: float
    longitude: float
    city: str
    region: str
    severity: Severity
    area: float  # in pixels
    depth: float  # in meters (from MiDaS)
    confidence: float
    timestamp: datetime
    id: Optional[int] = None
    image_path: Optional[str] = None

    def to_dict(self):
        return {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'city': self.city,
            'region': self.region,
            'severity': self.severity.value,
            'area': self.area,
            'depth': self.depth,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'image_path': self.image_path
        }
