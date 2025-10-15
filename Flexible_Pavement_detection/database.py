import sqlite3
from contextlib import contextmanager
import json
import os
from datetime import datetime
from typing import List, Optional, Dict
import logging

from config import config
from models import Pothole, Severity
from utils import calculate_distance

logger = logging.getLogger(__name__)


class PotholeDatabase:



    def __init__(self):
        self.db_path = config.DB_PATH  # e.g., 'potholes.db'
        self.init_database()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS potholes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    city TEXT,
                    region TEXT,
                    severity TEXT,
                    area REAL,
                    depth REAL,
                    confidence REAL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_location ON potholes(latitude, longitude)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_region ON potholes(region)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_severity ON potholes(severity)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON potholes(timestamp)")

    def is_duplicate(self, latitude: float, longitude: float) -> bool:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT latitude, longitude FROM potholes")
            for row in cur.fetchall():
                distance = calculate_distance(
                    latitude, longitude,
                    row['latitude'], row['longitude']
                )
                if distance <= config.DUPLICATE_RADIUS_METERS:
                    return True
        return False

    def add_pothole(self, pothole: Pothole) -> Optional[int]:
        if self.is_duplicate(pothole.latitude, pothole.longitude):
            logger.info(f"Duplicate pothole at ({pothole.latitude}, {pothole.longitude})")
            return None

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO potholes
                (latitude, longitude, city, region, severity, area, depth,
                 confidence, timestamp, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pothole.latitude, pothole.longitude, pothole.city,
                pothole.region, pothole.severity.value, pothole.area,
                pothole.depth, pothole.confidence,
                pothole.timestamp.isoformat(), pothole.image_path
            ))
            return cur.lastrowid

    def get_potholes(self, filters: Dict = None, sort_by: str = 'timestamp',
                     sort_order: str = 'DESC', limit: int = None) -> List[Pothole]:
        query = "SELECT * FROM potholes WHERE 1=1"
        params = []

        if filters:
            if 'region' in filters:
                query += " AND region = ?"
                params.append(filters['region'])
            if 'severity' in filters:
                query += " AND severity = ?"
                params.append(filters['severity'])
            if 'start_date' in filters:
                query += " AND timestamp >= ?"
                params.append(filters['start_date'])
            if 'end_date' in filters:
                query += " AND timestamp <= ?"
                params.append(filters['end_date'])

        valid_sort_columns = ['timestamp', 'severity', 'depth', 'area', 'confidence']
        if sort_by not in valid_sort_columns:
            sort_by = 'timestamp'

        query += f" ORDER BY {sort_by} {sort_order}"

        if limit:
            query += f" LIMIT {limit}"

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()

            return [
                Pothole(
                    id=row['id'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    city=row['city'],
                    region=row['region'],
                    severity=Severity(row['severity']),
                    area=row['area'],
                    depth=row['depth'],
                    confidence=row['confidence'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    image_path=row['image_path']
                )
                for row in rows
            ]

    def get_statistics(self) -> Dict:
        with self.get_connection() as conn:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) AS total FROM potholes")
            total = cur.fetchone()['total']

            cur.execute("""
                SELECT severity, COUNT(*) AS count
                FROM potholes GROUP BY severity
            """)
            severity_stats = {row['severity']: row['count'] for row in cur.fetchall()}

            cur.execute("""
                SELECT region, COUNT(*) AS count
                FROM potholes
                GROUP BY region
                ORDER BY count DESC
                LIMIT 10
            """)
            region_stats = [(row['region'], row['count']) for row in cur.fetchall()]

            return {
                'total': total,
                'by_severity': severity_stats,
                'top_regions': region_stats
            }

    def save_offline_log(self, potholes: List[Pothole]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(config.OFFLINE_LOG_DIR, f'potholes_{timestamp}.json')

        # Convert potholes to serializable format
        data = []
        for p in potholes:
            pothole_dict = {
                'latitude': float(p.latitude),  # Convert to Python float
                'longitude': float(p.longitude),
                'city': str(p.city),
                'region': str(p.region),
                'severity': p.severity.value,
                'area': float(p.area),
                'depth': float(p.depth),
                'confidence': float(p.confidence),
                'timestamp': p.timestamp.isoformat() if isinstance(p.timestamp, datetime) else str(p.timestamp),
                'image_path': p.image_path
            }
            data.append(pothole_dict)

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(potholes)} potholes to offline log: {filename}")
        except Exception as e:
            logger.error(f"Error saving offline log: {e}")

    def sync_offline_logs(self):
        """Sync offline logs to database when connection is restored"""
        if not os.path.exists(config.OFFLINE_LOG_DIR):
            return

        for filename in os.listdir(config.OFFLINE_LOG_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(config.OFFLINE_LOG_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    for item in data:
                        pothole = Pothole(
                            latitude=float(item['latitude']),
                            longitude=float(item['longitude']),
                            city=item['city'],
                            region=item['region'],
                            severity=Severity(item['severity']),
                            area=float(item['area']),
                            depth=float(item['depth']),
                            confidence=float(item['confidence']),
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            image_path=item.get('image_path')
                        )
                        self.add_pothole(pothole)

                    # Remove synced file
                    os.remove(filepath)
                    logger.info(f"Synced and removed offline log: {filename}")

                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted JSON file {filename}: {e}")
                    # Optionally move corrupted file to a backup directory
                    backup_dir = os.path.join(config.OFFLINE_LOG_DIR, 'corrupted')
                    os.makedirs(backup_dir, exist_ok=True)
                    os.rename(filepath, os.path.join(backup_dir, filename))
                    logger.info(f"Moved corrupted file to {backup_dir}")

                except Exception as e:
                    logger.error(f"Error syncing offline log {filename}: {e}")

