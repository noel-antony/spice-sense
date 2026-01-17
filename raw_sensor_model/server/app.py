"""
Spice Purity Server - Flask Backend
Receives sensor data from ESP32, runs ML inference, serves dashboard
Includes traceability and Firebase authentication
"""

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pickle
import json
import numpy as np
from datetime import datetime
import os
import qrcode
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spice-purity-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ============================================================
# FIREBASE AUTH (optional - won't crash if not configured)
# ============================================================
try:
    from firebase_auth import (
        require_auth, require_role, create_user, 
        set_user_role, list_all_users
    )
    FIREBASE_ENABLED = True
    print("✓ Firebase Auth enabled")
except Exception as e:
    FIREBASE_ENABLED = False
    print(f"⚠ Firebase Auth disabled: {e}")
    # Dummy decorators when Firebase is not available
    def require_auth(f):
        return f
    def require_role(roles):
        def decorator(f):
            return f
        return decorator

# ============================================================
# TRACEABILITY DATABASE
# ============================================================
try:
    from traceability_models import (
        init_db, Batch, SupplyChainEvent, PurityTest, Handler,
        generate_batch_id, generate_handler_id
    )
    init_db()
    TRACEABILITY_ENABLED = True
    print("✓ Traceability database initialized")
except Exception as e:
    TRACEABILITY_ENABLED = False
    print(f"⚠ Traceability disabled: {e}")

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
# QR CODE HELPER
# ============================================================

def generate_qr_code(data: str) -> str:
    """Generate QR code as base64 string"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

# ============================================================
# TRACEABILITY API ROUTES
# ============================================================

@app.route('/api/batches', methods=['GET'])
def get_batches():
    """Get all batches"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    batches = Batch.get_all()
    return jsonify({'success': True, 'count': len(batches), 'batches': batches})

@app.route('/api/batches', methods=['POST'])
@require_role(['admin', 'farmer'])
def create_batch():
    """Create a new batch"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    data = request.get_json()
    required = ['origin_farm', 'origin_location', 'harvest_date', 'quantity_kg']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    batch = Batch.create(
        origin_farm=data['origin_farm'],
        origin_location=data['origin_location'],
        harvest_date=data['harvest_date'],
        quantity_kg=float(data['quantity_kg']),
        farmer_name=data.get('farmer_name'),
        farmer_contact=data.get('farmer_contact'),
        initial_purity=data.get('initial_purity'),
        spice_type=data.get('spice_type', 'turmeric')
    )
    qr_data = f"https://spice-sense.vercel.app/track/{batch['id']}"
    batch['qr_code_base64'] = generate_qr_code(qr_data)
    return jsonify({'success': True, 'message': 'Batch created', 'batch': batch}), 201

@app.route('/api/batches/<batch_id>', methods=['GET'])
def get_batch(batch_id):
    """Get batch by ID"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    batch = Batch.get(batch_id)
    if not batch:
        return jsonify({'error': 'Batch not found'}), 404
    return jsonify({'success': True, 'batch': batch})

@app.route('/api/batches/<batch_id>/journey', methods=['GET'])
def get_batch_journey(batch_id):
    """Get complete journey of a batch"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    journey = Batch.get_full_journey(batch_id)
    if not journey:
        return jsonify({'error': 'Batch not found'}), 404
    return jsonify({'success': True, **journey})

@app.route('/api/batches/<batch_id>/verify', methods=['GET'])
def verify_batch_chain(batch_id):
    """Verify hash chain integrity for a batch"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    batch = Batch.get(batch_id)
    if not batch:
        return jsonify({'error': 'Batch not found'}), 404
    is_valid = SupplyChainEvent.verify_chain(batch_id)
    return jsonify({
        'success': True,
        'batch_id': batch_id,
        'chain_valid': is_valid,
        'integrity': 'VERIFIED ✓' if is_valid else 'COMPROMISED ✗'
    })

@app.route('/api/events', methods=['POST'])
@require_auth
def record_event():
    """Record a supply chain event"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    data = request.get_json()
    required = ['batch_id', 'event_type', 'stage']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    batch = Batch.get(data['batch_id'])
    if not batch:
        return jsonify({'error': 'Batch not found'}), 404
    event = SupplyChainEvent.record(
        batch_id=data['batch_id'],
        event_type=data['event_type'],
        stage=data['stage'],
        location=data.get('location'),
        handler_name=data.get('handler_name'),
        handler_id=data.get('handler_id'),
        handler_type=data.get('handler_type'),
        details=data.get('details'),
        purity_score=data.get('purity_score'),
        purity_grade=data.get('purity_grade'),
        quantity_kg=data.get('quantity_kg'),
        temperature_c=data.get('temperature_c'),
        humidity_percent=data.get('humidity_percent')
    )
    return jsonify({'success': True, 'message': 'Event recorded', 'event': event}), 201

@app.route('/api/purity-tests', methods=['POST'])
@require_role(['admin', 'tester'])
def record_purity_test():
    """Record a purity test"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    data = request.get_json()
    required = ['batch_id', 'purity_percent', 'quality_grade']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    batch = Batch.get(data['batch_id'])
    if not batch:
        return jsonify({'error': 'Batch not found'}), 404
    test = PurityTest.record(
        batch_id=data['batch_id'],
        purity_percent=float(data['purity_percent']),
        quality_grade=data['quality_grade'],
        mq135_aqi=data.get('mq135_aqi'),
        mq3_voltage=data.get('mq3_voltage'),
        gas_resistance_kOhm=data.get('gas_resistance_kOhm'),
        binary_classification=data.get('binary_classification'),
        confidence=data.get('confidence'),
        tester_name=data.get('tester_name'),
        test_location=data.get('test_location'),
        notes=data.get('notes')
    )
    return jsonify({'success': True, 'message': 'Purity test recorded', 'test': test}), 201

@app.route('/api/handlers', methods=['GET'])
def get_handlers():
    """Get all handlers"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    handler_type = request.args.get('type')
    handlers = Handler.get_all(handler_type)
    return jsonify({'success': True, 'count': len(handlers), 'handlers': handlers})

@app.route('/api/handlers', methods=['POST'])
@require_role(['admin'])
def create_handler():
    """Register a new handler"""
    if not TRACEABILITY_ENABLED:
        return jsonify({'error': 'Traceability not enabled'}), 503
    data = request.get_json()
    required = ['name', 'handler_type']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    if data['handler_type'] not in Handler.TYPES:
        return jsonify({'error': f'Invalid handler type. Must be one of: {Handler.TYPES}'}), 400
    handler = Handler.create(
        name=data['name'],
        handler_type=data['handler_type'],
        organization=data.get('organization'),
        location=data.get('location'),
        contact_email=data.get('contact_email'),
        contact_phone=data.get('contact_phone'),
        license_number=data.get('license_number')
    )
    return jsonify({'success': True, 'message': 'Handler registered', 'handler': handler}), 201

# ============================================================
# AUTH API ROUTES
# ============================================================

@app.route('/api/auth/register', methods=['POST'])
@require_role(['admin'])
def register_user():
    """Register a new user - Admin only"""
    if not FIREBASE_ENABLED:
        return jsonify({'error': 'Firebase Auth not enabled'}), 503
    data = request.get_json()
    required = ['email', 'password', 'role']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    try:
        user = create_user(
            email=data['email'],
            password=data['password'],
            display_name=data.get('display_name'),
            role=data['role']
        )
        return jsonify({'success': True, 'message': 'User created', 'user': user}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/auth/set-role', methods=['POST'])
@require_role(['admin'])
def update_user_role():
    """Update user role - Admin only"""
    if not FIREBASE_ENABLED:
        return jsonify({'error': 'Firebase Auth not enabled'}), 503
    data = request.get_json()
    if 'uid' not in data or 'role' not in data:
        return jsonify({'error': 'Missing uid or role'}), 400
    try:
        set_user_role(data['uid'], data['role'])
        return jsonify({'success': True, 'message': f"Role updated to {data['role']}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/auth/users', methods=['GET'])
@require_role(['admin'])
def get_all_users():
    """List all users - Admin only"""
    if not FIREBASE_ENABLED:
        return jsonify({'error': 'Firebase Auth not enabled'}), 503
    try:
        users = list_all_users()
        return jsonify({'success': True, 'count': len(users), 'users': users})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_user():
    """Get current authenticated user info"""
    return jsonify({
        'success': True,
        'user': {
            'uid': request.user.get('uid'),
            'email': request.user.get('email'),
            'role': request.user.get('role', 'viewer')
        }
    })

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*50)
    print("🌶️  SPICE PURITY SERVER")
    print("="*50)
    print(f"Dashboard: http://localhost:{port}")
    print(f"API Endpoint: http://localhost:{port}/api/predict")
    print("="*50 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
