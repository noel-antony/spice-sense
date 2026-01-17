"""
Test script to send sample data to the server
Run this after starting the server with: python app.py
"""

import requests
import time
import random

SERVER_URL = "http://localhost:5000/api/predict"

# Sample data for different purity levels
SAMPLES = {
    "100% Pure": {"mq135_ratio": 1.05, "mq3_voltage": 1.32, "gas_resistance_kOhm": 27.0},
    "75% Pure": {"mq135_ratio": 1.58, "mq3_voltage": 1.30, "gas_resistance_kOhm": 32.8},
    "50% Pure": {"mq135_ratio": 1.77, "mq3_voltage": 1.27, "gas_resistance_kOhm": 35.4},
    "25% Pure": {"mq135_ratio": 1.97, "mq3_voltage": 1.26, "gas_resistance_kOhm": 39.0},
}

def test_single():
    """Test with a single sample"""
    print("\n🧪 Testing Single Prediction")
    print("="*50)
    
    data = SAMPLES["100% Pure"]
    try:
        response = requests.post(SERVER_URL, json=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Sent: {data}")
            print(f"✓ Purity: {result['predictions']['regression']['purity_percent']}%")
            print(f"✓ Quality: {result['predictions']['multiclass']['label']}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server. Is it running?")
        print("  Start the server with: python app.py")
        return False

def test_all_samples():
    """Test with all sample types"""
    print("\n🧪 Testing All Purity Levels")
    print("="*50)
    
    for name, data in SAMPLES.items():
        try:
            response = requests.post(SERVER_URL, json=data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                pred_purity = result['predictions']['regression']['purity_percent']
                quality = result['predictions']['multiclass']['label']
                print(f"✓ {name:12} → Predicted: {pred_purity:5.1f}% ({quality})")
            else:
                print(f"✗ {name}: Error {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("✗ Connection failed")
            return

def simulate_esp32():
    """Simulate ESP32 sending data every 2 seconds"""
    print("\n🔄 Simulating ESP32 Data Stream")
    print("="*50)
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Pick random purity level with some variation
            base = random.choice(list(SAMPLES.values()))
            data = {
                "mq135_ratio": base["mq135_ratio"] + random.uniform(-0.1, 0.1),
                "mq3_voltage": base["mq3_voltage"] + random.uniform(-0.02, 0.02),
                "gas_resistance_kOhm": base["gas_resistance_kOhm"] + random.uniform(-1, 1)
            }
            
            try:
                response = requests.post(SERVER_URL, json=data, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    purity = result['predictions']['regression']['purity_percent']
                    quality = result['predictions']['multiclass']['label']
                    print(f"📤 Gas: {data['gas_resistance_kOhm']:.1f}kΩ → {purity:.0f}% {quality}")
            except:
                print("✗ Connection error")
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == "__main__":
    print("\n🌶️ Spice Purity Server - Test Client")
    print("="*50)
    
    # Check server health first
    try:
        r = requests.get("http://localhost:5000/api/health", timeout=2)
        print("✓ Server is running!")
    except:
        print("✗ Server not running. Start it first:")
        print("  cd server")
        print("  python app.py")
        exit(1)
    
    # Run tests
    test_single()
    test_all_samples()
    
    # Ask to simulate
    print("\n" + "="*50)
    response = input("Simulate ESP32 data stream? (y/n): ")
    if response.lower() == 'y':
        simulate_esp32()
