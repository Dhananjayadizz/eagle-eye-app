# gps_test.py
import serial
from serial.tools import list_ports
import time
import sys

def find_esp32_port():
    """Identify ESP32 based on common USB vendor IDs"""
    esp32_vids = {0x1A86, 0x10C4, 0x303A}  # Common CH340/CP210X/ESP32 vendor IDs
    ports = list_ports.comports()
    for port in ports:
        if port.vid in esp32_vids:
            return port.device
    return None

def parse_gps_data(line):
    """Parse GPS data from serial output with validation"""
    try:
        if line.startswith("Latitude:"):
            return ('latitude', float(line.split(':')[1].strip()))
        elif line.startswith("Longitude:"):
            return ('longitude', float(line.split(':')[1].strip()))
        elif "Waiting for GPS signal" in line:
            return ('status', 'no_fix')
    except (ValueError, IndexError) as e:
        print(f"Parsing error: {e}")
        return None

def main():
    port = find_esp32_port()
    if not port:
        print("ESP32 not found. Available ports:")
        for p in list_ports.comports():
            print(f"- {p.device}: {p.description}")
        sys.exit(1)

    print(f"Found ESP32 at {port}, connecting...")
    
    try:
        with serial.Serial(
            port=port,
            baudrate=115200,
            timeout=2,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        ) as ser:
            ser.reset_input_buffer()
            gps_data = {'latitude': None, 'longitude': None}
            
            while True:
                try:
                    # Read until newline to get complete messages
                    line = ser.read_until(b'\n').decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    print(f"Raw: {line}")  # Debug raw input
                    
                    result = parse_gps_data(line)
                    if result:
                        key, value = result
                        if key == 'status':
                            print("\033[33m" + value + "\033[0m")  # Yellow warning
                            gps_data = {'latitude': None, 'longitude': None}  # Reset
                        else:
                            gps_data[key] = value
                            if all(gps_data.values()):
                                print(f"\033[32mValid GPS Fix: {gps_data}\033[0m")
                            else:
                                print(f"Partial Data: {gps_data}")

                except UnicodeDecodeError:
                    print("Invalid data received")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()