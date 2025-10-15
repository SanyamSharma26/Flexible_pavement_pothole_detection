import cv2
import serial
import logging
import threading
import time
from datetime import datetime
from queue import Queue
from geopy.geocoders import Nominatim

from config import config
from database import PotholeDatabase
from detector import PotholeDetector
from bot import PotholeBot
from gps_provider import SimulatedGPS, RealGPS
from utils import save_detection_image

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class PotholeDetectionSystem:
    def __init__(self):
        self.db = PotholeDatabase()
        self.detector = PotholeDetector()
        self.bot = PotholeBot(self.db)
        self.geolocator = Nominatim(user_agent="pothole_detector")
        self.detection_queue = Queue()
        self.running = False
        if config.USE_SIMULATION:
            self.gps = SimulatedGPS()
        else:
            self.gps = RealGPS(config.GPS_PORT, config.GPS_BAUDRATE)

    def process_video(self):
        """Main video processing loop"""
        self.running = True
        ser = None
        cap = None
        video_writer = None

        try:
            # Initialize serial port (for fallback legacy method)
            try:
                ser = serial.Serial(config.GPS_PORT, config.GPS_BAUDRATE, timeout=1)
                logger.info("GPS serial port opened")
            except Exception as e:
                logger.warning(f"Could not open GPS port: {e}. Continuing without legacy GPS.")


            # Open video or webcam
            if config.USE_LIVE_CAMERA:
                cap = cv2.VideoCapture(config.CAMERA_INDEX)
                logger.info("Using live webcam feed.")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
            else:
                cap = cv2.VideoCapture(config.VIDEO_FILE)
                logger.info(f"Using video file: {config.VIDEO_FILE}")

            if not cap.isOpened():
                raise ValueError("Could not open video source")

            # Prepare video writer if enabled
            if config.SAVE_VIDEO:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or 'mp4v' for .mp4 output
                video_writer = cv2.VideoWriter(
                    config.VIDEO_OUTPUT_PATH,
                    fourcc,
                    config.VIDEO_FPS,
                    (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
                )
                logger.info(f"Video recording enabled: {config.VIDEO_OUTPUT_PATH}")

            frame_count = 0
            last_gps_data = None

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Skip frames based on config
                if frame_count % config.FRAME_SKIP != 0:
                    continue

                # Resize frame
                frame = cv2.resize(frame, (config.VIDEO_WIDTH, config.VIDEO_HEIGHT))

                # Get GPS data
                gps_data = self.gps.get_gps_data()
                if gps_data:
                    last_gps_data = gps_data
                else:
                    gps_data = last_gps_data

                # Overlay GPS info on frame
                if gps_data:
                    gps_text = f"{gps_data['city']}, {gps_data['region']} ({gps_data['latitude']:.5f}, {gps_data['longitude']:.5f})"
                    cv2.putText(frame, gps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Detect potholes
                potholes, annotated_frame = self.detector.detect_potholes(frame, gps_data)

                # Process detected potholes
                for pothole in potholes:
                    if gps_data and not self.db.is_duplicate(pothole.latitude, pothole.longitude):
                        try:
                            pothole_id = self.db.add_pothole(pothole)
                            if pothole_id:
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                image_path = save_detection_image(annotated_frame, pothole_id, timestamp)
                                logger.info(f"New pothole detected: ID={pothole_id}, "
                                            f"Severity={pothole.severity.value}, "
                                            f"Depth={pothole.depth:.3f}m, "
                                            f"Location=({pothole.latitude:.6f}, {pothole.longitude:.6f})")
                        except Exception as e:
                            logger.error(f"Database error: {e}")
                            self.db.save_offline_log([pothole])

                # Save frame to video if enabled
                if config.SAVE_VIDEO and video_writer:
                    video_writer.write(annotated_frame)

                # Show live output
                cv2.imshow('Pothole Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Processing error: {e}")

        finally:
            if cap:
                cap.release()
            if ser and ser.is_open:
                ser.close()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            self.running = False
            logger.info("Video processing stopped")
            if isinstance(self.gps, RealGPS):
                self.gps.close()

    def sync_offline_data(self):
        """Periodically sync offline data"""
        while self.running:
            try:
                self.db.sync_offline_logs()
            except Exception as e:
                logger.error(f"Sync error: {e}")
            time.sleep(60)  # Sync every minute



    def run(self):
        """Run the complete system"""
        # Start video processing in a separate thread
        video_thread = threading.Thread(target=self.process_video)
        video_thread.start()

        # Start offline sync in a separate thread
        sync_thread = threading.Thread(target=self.sync_offline_data)
        sync_thread.start()

        try:
            # Run bot in main thread
            self.bot.run()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False
            video_thread.join()
            sync_thread.join()


def main():
    system = PotholeDetectionSystem()
    system.run()


if __name__ == '__main__':
    main()
