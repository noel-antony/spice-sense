# 🌶️ Spice Purity Server & Dashboard

Real-time spice purity detection system with ESP32 sensor integration and web dashboard.

## Architecture

```
┌─────────────┐      WiFi/HTTP      ┌──────────────┐      WebSocket      ┌─────────────┐
│   ESP32     │ ─────────────────▶ │  Flask Server │ ◀────────────────▶ │  Dashboard  │
│  (Sensors)  │   POST /api/predict │  (ML Models)  │   Real-time push   │   (Web UI)  │
└─────────────┘                     └──────────────┘                     └─────────────┘
```

## Quick Start

### 1. Install Python Dependencies

```bash
cd server
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Open Dashboard

Open your browser and go to: **http://localhost:5000**

### 4. Configure ESP32

Edit `esp32_wifi_client.ino` and update these lines:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_SSID";          // Your WiFi name
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";  // Your WiFi password
const char* SERVER_URL = "http://YOUR_PC_IP:5000/api/predict";  // Your PC's IP
```

To find your PC's IP address:
- Windows: Run `ipconfig` in Command Prompt
- Look for "IPv4 Address" under your WiFi adapter (e.g., `192.168.1.100`)

### 5. Upload to ESP32

1. Open `esp32_wifi_client.ino` in Arduino IDE
2. Install required libraries:
   - Adafruit BME680 Library
   - ArduinoJson
3. Select ESP32 board and correct COM port
4. Upload the sketch

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard web page |
| `/api/predict` | POST | Submit sensor data, get predictions |
| `/api/health` | GET | Server health check |
| `/api/history` | GET | Get prediction history |
| `/api/model-info` | GET | Get model metadata |

### POST /api/predict

**Request Body:**
```json
{
    "mq135_ratio": 1.65,
    "mq3_voltage": 1.28,
    "gas_resistance_kOhm": 32.5
}
```

**Response:**
```json
{
    "timestamp": "2026-01-16T10:30:45.123456",
    "raw_features": {
        "mq135_ratio": 1.65,
        "mq135_aqi": 60.61,
        "mq3_voltage": 1.28,
        "gas_resistance_kOhm": 32.5
    },
    "predictions": {
        "binary": {
            "label": "high_purity",
            "confidence": 98.5
        },
        "multiclass": {
            "label": "GOOD",
            "confidence": 95.2,
            "probabilities": {
                "best": 3.5,
                "good": 95.2,
                "poor": 1.2,
                "worst": 0.1
            }
        },
        "regression": {
            "purity_percent": 78.5
        }
    }
}
```

## Testing Without ESP32

You can test the server using curl or Python:

```bash
# Using curl
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"mq135_ratio": 1.65, "mq3_voltage": 1.28, "gas_resistance_kOhm": 32.5}'
```

```python
# Using Python
import requests

response = requests.post(
    'http://localhost:5000/api/predict',
    json={
        'mq135_ratio': 1.65,
        'mq3_voltage': 1.28,
        'gas_resistance_kOhm': 32.5
    }
)
print(response.json())
```

## Files Structure

```
raw_sensor_model/
├── server/
│   ├── app.py              # Flask server with ML inference
│   ├── requirements.txt    # Python dependencies
│   ├── templates/
│   │   └── dashboard.html  # Real-time web dashboard
│   └── README.md           # This file
├── esp32_wifi_client/
│   └── esp32_wifi_client.ino  # ESP32 WiFi sketch
├── binary_classifier_raw.pkl   # Binary classification model
├── multiclass_classifier_raw.pkl  # Multi-class model
├── regression_model_raw.pkl    # Regression model
└── model_metadata_raw.json     # Model metadata
```

## Troubleshooting

### ESP32 can't connect to server
1. Make sure ESP32 and PC are on the same WiFi network
2. Check Windows Firewall - allow Python/Flask through
3. Verify the SERVER_URL IP address is correct

### Dashboard not updating
1. Check browser console for WebSocket errors
2. Ensure the server is running
3. Try refreshing the page

### Model not loading
1. Ensure you're running from the correct directory
2. Check that .pkl files exist in the parent folder
