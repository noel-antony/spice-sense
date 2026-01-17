"""
ESP32 Serial Bridge
Reads sensor data from ESP32 via COM port and sends to Flask server
"""

import serial
import serial.tools.list_ports
import requests
import re
import time
import threading
import sys

# Configuration
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
SERVER_URL = "http://localhost:5000/api/predict"

# Global flag to stop the reader
running = True

def list_ports():
    """List available COM ports"""
    ports = serial.tools.list_ports.comports()
    print("\n📡 Available COM Ports:")
    for port in ports:
        print(f"   {port.device}: {port.description}")
    return [p.device for p in ports]

# Global storage for multi-line parsing
sensor_buffer = {}

def parse_sensor_line(line):
    """
    Parse sensor data from ESP32 serial output.
    Supports multiple formats including the verbose analysis output.
    """
    global sensor_buffer
    
    # Pattern 1: Simple format "MQ135 Ratio: X.XX | MQ3: X.XXV | Gas: XX.XX kΩ"
    pattern = r'MQ135.*?Ratio:\s*([\d.]+).*?MQ3:\s*([\d.]+)\s*V.*?Gas:\s*([\d.]+)'
    match = re.search(pattern, line, re.IGNORECASE)
    if match:
        return {
            'mq135_ratio': float(match.group(1)),
            'mq3_voltage': float(match.group(2)),
            'gas_resistance_kOhm': float(match.group(3))
        }
    
    # Pattern 2: Parse individual lines from verbose output
    # "Gas Resistance:   67.39 kΩ"
    gas_match = re.search(r'Gas\s*Resistance:\s*([\d.]+)', line, re.IGNORECASE)
    if gas_match:
        sensor_buffer['gas_resistance_kOhm'] = float(gas_match.group(1))
    
    # "MQ135 Ratio:      3.8887"
    mq135_match = re.search(r'MQ135\s*Ratio:\s*([\d.]+)', line, re.IGNORECASE)
    if mq135_match:
        sensor_buffer['mq135_ratio'] = float(mq135_match.group(1))
    
    # "MQ3 Voltage:      1.515 V"
    mq3_match = re.search(r'MQ3\s*Voltage:\s*([\d.]+)', line, re.IGNORECASE)
    if mq3_match:
        sensor_buffer['mq3_voltage'] = float(mq3_match.group(1))
    
    # Check if we have all 3 values
    if all(k in sensor_buffer for k in ['mq135_ratio', 'mq3_voltage', 'gas_resistance_kOhm']):
        data = sensor_buffer.copy()
        sensor_buffer.clear()  # Reset for next reading
        return data
    
    # Pattern 3: JSON format
    if '{' in line and '}' in line:
        try:
            import json
            start = line.index('{')
            end = line.rindex('}') + 1
            data = json.loads(line[start:end])
            if 'mq135_ratio' in data or 'gas_resistance_kOhm' in data:
                return data
        except:
            pass
    
    return None

def send_to_server(data):
    """Send sensor data to Flask server"""
    try:
        response = requests.post(SERVER_URL, json=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            purity = result['predictions']['regression']['purity_percent']
            quality = result['predictions']['multiclass']['label']
            return purity, quality
        else:
            print(f"⚠️ Server error: {response.status_code}")
            return None, None
    except requests.exceptions.ConnectionError:
        print("⚠️ Cannot connect to server. Is it running?")
        return None, None
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None, None

def serial_reader(port, baud):
    """Main serial reading loop"""
    global running
    
    print(f"\n📡 Connecting to {port} at {baud} baud...")
    
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"✓ Connected to {port}")
        print("\n" + "="*60)
        print("Reading sensor data... (Press Ctrl+C to stop)")
        print("="*60 + "\n")
        
        last_sent = None
        
        while running:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        # Skip printing every line - only show important ones
                        if any(x in line for x in ['Gas Resistance:', 'MQ135 Ratio:', 'MQ3 Voltage:']):
                            # Try to parse sensor data
                            data = parse_sensor_line(line)
                            
                            if data:
                                # Don't send duplicates
                                data_key = f"{data['gas_resistance_kOhm']:.2f}-{data['mq135_ratio']:.2f}"
                                if data_key != last_sent:
                                    last_sent = data_key
                                    
                                    # Print what we're sending
                                    print(f"📤 Sending: Gas={data['gas_resistance_kOhm']:.2f}kΩ, "
                                          f"MQ135={data['mq135_ratio']:.2f}, MQ3={data['mq3_voltage']:.3f}V")
                                    
                                    # Send to server
                                    purity, quality = send_to_server(data)
                                    
                                    if purity is not None:
                                        print(f"   └─→ 🎯 {purity:.1f}% PURITY - {quality}")
                                        print()
                
                time.sleep(0.05)
                
            except serial.SerialException as e:
                print(f"⚠️ Serial error: {e}")
                break
                
    except serial.SerialException as e:
        print(f"❌ Could not open {port}: {e}")
        available = list_ports()
        if available:
            print(f"\nTry one of the available ports above.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("\n✓ Serial port closed")

def check_server():
    """Check if server is running"""
    try:
        r = requests.get("http://localhost:5000/api/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def main():
    global running
    
    print("\n" + "="*60)
    print("🌶️  ESP32 SERIAL BRIDGE - Spice Purity Tester")
    print("="*60)
    
    # Check server
    print("\n🔍 Checking server connection...")
    if check_server():
        print("✓ Server is running at http://localhost:5000")
    else:
        print("⚠️ Server not running!")
        print("   Start it with: python app.py")
        print("   Continuing anyway (data will buffer)...")
    
    # List available ports
    available = list_ports()
    
    if SERIAL_PORT not in available:
        print(f"\n⚠️ {SERIAL_PORT} not found!")
        if available:
            print(f"   Available ports: {', '.join(available)}")
        else:
            print("   No COM ports detected. Is ESP32 connected?")
        return
    
    # Start reading
    try:
        serial_reader(SERIAL_PORT, BAUD_RATE)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
        running = False

if __name__ == "__main__":
    main()
