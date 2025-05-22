import serial
import threading
import logging
from flask_socketio import SocketIO
from app.extensions import socketio

logger = logging.getLogger(__name__)

# Global GPS data and lock
gps_data = {
    "latitude": 0.0,
    "longitude": 0.0,
    "connected": False
}
gps_lock = threading.Lock()

def init_gps():
    """Initialize GPS thread"""
    try:
        gps_thread = threading.Thread(target=read_gps_data_from_serial, 
                                    args=('COM5', 115200))
        gps_thread.daemon = True
        gps_thread.start()
        logger.info("GPS thread started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GPS: {e}")

def read_gps_data_from_serial(port, baudrate):
    """Read GPS data from serial port"""
    global gps_data
    logger.info(f"Connecting to GPS on {port} at {baudrate}...")

    try:
        serial_port = serial.Serial(port, baudrate, timeout=1)
        logger.info("Serial port opened.")
        with gps_lock:
            gps_data["connected"] = True

        while True:
            if serial_port.in_waiting > 0:
                line = serial_port.readline().decode('utf-8', errors='ignore').strip()
                logger.debug(f"Raw GPS line: {line}")

                # Parse latitude and longitude using regex
                import re
                lat_match = re.search(r'Latitude:(-?\d+\.\d+)', line)
                lon_match = re.search(r'Longitude:(-?\d+\.\d+)', line)

                if lat_match:
                    with gps_lock:
                        gps_data['latitude'] = float(lat_match.group(1))
                        gps_data['connected'] = True

                if lon_match:
                    with gps_lock:
                        gps_data['longitude'] = float(lon_match.group(1))
                        gps_data['connected'] = True

                    # Emit once both are likely to be updated
                    socketio.emit('gps_update', gps_data)

    except Exception as e:
        logger.error(f"GPS Serial Error: {e}")
        with gps_lock:
            gps_data.update({
                "latitude": 0.0,
                "longitude": 0.0,
                "connected": False
            })
        socketio.emit('gps_update', gps_data)

def get_gps_data():
    """Get current GPS data"""
    with gps_lock:
        return {
            "latitude": gps_data["latitude"],
            "longitude": gps_data["longitude"],
            "speed": 40.0 if gps_data["connected"] else 0.0,  # Placeholder
            "connected": gps_data["connected"]
        }
