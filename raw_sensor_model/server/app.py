"""
Spice Purity Server - Flask Backend
Receives sensor data from ESP32, runs ML inference, serves dashboard
"""

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pickle
import json
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spice-purity-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================================
# LOAD ML MODELS
# ============================================================

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading ML models...")

with open(os.path.join(MODEL_DIR, 'binary_classifier_raw.pkl'), 'rb') as f:
    binary_data = pickle.load(f)
    binary_model = binary_data['model']
    binary_scaler = binary_data['scaler']
    binary_encoder = binary_data['label_encoder']

with open(os.path.join(MODEL_DIR, 'multiclass_classifier_raw.pkl'), 'rb') as f:
    multi_data = pickle.load(f)
    multi_model = multi_data['model']
    multi_encoder = multi_data['label_encoder']

with open(os.path.join(MODEL_DIR, 'regression_model_raw.pkl'), 'rb') as f:
    reg_data = pickle.load(f)
    reg_model = reg_data['model']
    reg_scaler = reg_data['scaler']

with open(os.path.join(MODEL_DIR, 'model_metadata_raw.json'), 'r') as f:
    metadata = json.load(f)

print("✓ Models loaded successfully!")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_aqi(mq135_ratio):
    """Calculate AQI from MQ135 ratio"""
    if mq135_ratio <= 0:
        return 500.0
    aqi = 100.0 / mq135_ratio
    return max(0, min(500, aqi))

def run_inference(mq135_ratio, mq3_voltage, gas_resistance_kOhm):
    """Run all ML models on sensor data"""
    
    # Calculate AQI from ratio
    mq135_aqi = calculate_aqi(mq135_ratio)
    
    # Prepare features: [mq135_aqi, mq3_voltage, gas_resistance_kOhm]
    features = np.array([[mq135_aqi, mq3_voltage, gas_resistance_kOhm]])
    
    # Scale features for classification
    features_scaled_clf = binary_scaler.transform(features)
    features_scaled_reg = reg_scaler.transform(features)
    
    # Binary classification
    binary_pred = binary_model.predict(features_scaled_clf)[0]
    binary_proba = binary_model.predict_proba(features_scaled_clf)[0]
    binary_label = binary_encoder.inverse_transform([binary_pred])[0]
    
    # Multi-class classification
    multi_pred = multi_model.predict(features_scaled_clf)[0]
    multi_proba = multi_model.predict_proba(features_scaled_clf)[0]
    multi_label = multi_encoder.inverse_transform([multi_pred])[0]
    
    # Regression
    purity_pred = reg_model.predict(features_scaled_reg)[0]
    purity_pred = max(0, min(100, purity_pred))  # Clamp 0-100
    
    return {
        'timestamp': datetime.now().isoformat(),
        'raw_features': {
            'mq135_ratio': round(mq135_ratio, 4),
            'mq135_aqi': round(mq135_aqi, 2),
            'mq3_voltage': round(mq3_voltage, 4),
            'gas_resistance_kOhm': round(gas_resistance_kOhm, 2)
        },
        'predictions': {
            'binary': {
                'label': binary_label,
                'confidence': round(float(max(binary_proba)) * 100, 1)
            },
            'multiclass': {
                'label': multi_label.upper(),
                'confidence': round(float(max(multi_proba)) * 100, 1),
                'probabilities': {
                    label: round(float(prob) * 100, 1) 
                    for label, prob in zip(multi_encoder.classes_, multi_proba)
                }
            },
            'regression': {
                'purity_percent': round(float(purity_pred), 1)
            }
        }
    }

# Store last N readings for history
reading_history = []
MAX_HISTORY = 100

# ============================================================
# API ROUTES
# ============================================================

@app.route('/')
def dashboard():
    """Serve the dashboard HTML"""
    return render_template('dashboard.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'features': metadata['feature_names']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Receive sensor data and return predictions"""
    try:
        data = request.get_json()
        
        # Extract sensor values
        mq135_ratio = float(data.get('mq135_ratio', 0))
        mq3_voltage = float(data.get('mq3_voltage', 0))
        gas_resistance = float(data.get('gas_resistance_kOhm', 0))
        
        # Run inference
        result = run_inference(mq135_ratio, mq3_voltage, gas_resistance)
        
        # Add to history
        reading_history.append(result)
        if len(reading_history) > MAX_HISTORY:
            reading_history.pop(0)
        
        # Broadcast to connected clients via WebSocket
        socketio.emit('new_reading', result)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get reading history"""
    return jsonify(reading_history)

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model metadata"""
    return jsonify(metadata)

# ============================================================
# WEBSOCKET EVENTS
# ============================================================

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'connected', 'history_count': len(reading_history)})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_history')
def handle_history_request():
    emit('history', reading_history)

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌶️  SPICE PURITY SERVER")
    print("="*50)
    print(f"Dashboard: http://localhost:5000")
    print(f"API Endpoint: http://localhost:5000/api/predict")
    print("="*50 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
