import serial
import logging
from abc import ABC, abstractmethod
from geopy.geocoders import Nominatim

logger = logging.getLogger(__name__)


class BaseGPS(ABC):
    @abstractmethod
    def get_gps_data(self):
        pass


class RealGPS(BaseGPS):
    def __init__(self, port, baudrate):
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.geolocator = Nominatim(user_agent="pothole_detector")
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            logger.info("GPS serial port opened")
        except Exception as e:
            logger.warning(f"Could not open GPS port: {e}")

    def get_gps_data(self):
        try:
            if not self.ser or not self.ser.is_open:
                return None

            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                return None

            lat, lon = self._parse_nmea_sentence(line)
            if lat is not None and lon is not None:
                city, region = self._get_location_info(lat, lon)
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'city': city,
                    'region': region
                }
        except Exception as e:
            logger.error(f"GPS reading error: {e}")
        return None

    def _parse_nmea_sentence(self, sentence):
        try:
            parts = sentence.split(',')
            if parts[0] == '$GPGGA' and len(parts) >= 6:
                lat = self._nmea_to_decimal(parts[2], parts[3])
                lon = self._nmea_to_decimal(parts[4], parts[5])
                return lat, lon
            elif parts[0] == '$GPRMC' and len(parts) >= 7 and parts[2] == 'A':
                lat = self._nmea_to_decimal(parts[3], parts[4])
                lon = self._nmea_to_decimal(parts[5], parts[6])
                return lat, lon
        except Exception as e:
            logger.debug(f"NMEA parse error: {e}")
        return None, None

    def _nmea_to_decimal(self, coord_str, direction):
        try:
            if not coord_str or '.' not in coord_str:
                return 0.0

            if direction in ['N', 'S']:
                degrees = int(coord_str[:2])
                minutes = float(coord_str[2:])
            else:
                degrees = int(coord_str[:3])
                minutes = float(coord_str[3:])

            decimal = degrees + minutes / 60.0
            if direction in ['S', 'W']:
                decimal *= -1
            return decimal
        except:
            return 0.0

    def _get_location_info(self, lat, lon):
        try:
            location = self.geolocator.reverse((lat, lon), timeout=5)
            if location and location.raw.get('address'):
                address = location.raw['address']
                return address.get('city', address.get('town', 'Unknown')), address.get('state', 'Unknown')
        except Exception as e:
            logger.debug(f"Geocoding error: {e}")
        return 'Unknown', 'Unknown'

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


class SimulatedGPS(BaseGPS):
    def __init__(self):
        self.geolocator = Nominatim(user_agent="pothole_detector_sim")

    def get_gps_data(self):
        try:
            lat, lon = 44.7866, 20.4489  # Belgrade
            city, region = self._get_location_info(lat, lon)
            return {
                'latitude': lat,
                'longitude': lon,
                'city': city,
                'region': region
            }
        except Exception as e:
            logger.debug(f"Simulated GPS er qcror: {e}")
            return None

    def _get_location_info(self, lat, lon):
        try:
            location = self.geolocator.reverse((lat, lon), timeout=5)
            if location and location.raw.get('address'):
                address = location.raw['address']
                return address.get('city', address.get('town', 'Unknown')), address.get('state', 'Unknown')
        except Exception as e:
            logger.debug(f"Simulated geocoding error: {e}")
        return 'Unknown', 'Unknown'
