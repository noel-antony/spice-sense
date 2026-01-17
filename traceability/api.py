"""
Spice Traceability - REST API
Flask API for supply chain tracking with Firebase Authentication
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import os
import qrcode
import io
import base64

from models import (
    init_db, Batch, SupplyChainEvent, PurityTest, Handler,
    generate_batch_id, generate_handler_id
)

from firebase_config import (
    require_auth, require_role, create_user, 
    set_user_role, get_user_by_email, list_all_users
)

app = Flask(__name__)
CORS(app)

# Initialize database
init_db()


# ============================================================
# HELPER FUNCTIONS
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
# BATCH ENDPOINTS
# ============================================================

@app.route('/api/batches', methods=['GET'])
def get_batches():
    """Get all batches - Public endpoint"""
    batches = Batch.get_all()
    return jsonify({
        'success': True,
        'count': len(batches),
        'batches': batches
    })


@app.route('/api/batches', methods=['POST'])
@require_role(['admin', 'farmer'])  # Only admins and farmers can create batches
def create_batch():
    """Create a new batch - Requires farmer or admin role"""
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
    
    # Generate QR code
    qr_data = f"https://spice-sense.vercel.app/track/{batch['id']}"
    batch['qr_code_base64'] = generate_qr_code(qr_data)
    
    return jsonify({
        'success': True,
        'message': 'Batch created successfully',
        'batch': batch
    }), 201


@app.route('/api/batches/<batch_id>', methods=['GET'])
def get_batch(batch_id):
    """Get batch by ID"""
    batch = Batch.get(batch_id)
    if not batch:
        return jsonify({'error': 'Batch not found'}), 404
    
    return jsonify({
        'success': True,
        'batch': batch
    })


@app.route('/api/batches/<batch_id>/journey', methods=['GET'])
def get_batch_journey(batch_id):
    """Get complete journey of a batch"""
    journey = Batch.get_full_journey(batch_id)
    if not journey:
        return jsonify({'error': 'Batch not found'}), 404
    
    return jsonify({
        'success': True,
        **journey
    })


@app.route('/api/batches/<batch_id>/qr', methods=['GET'])
def get_batch_qr(batch_id):
    """Get QR code for batch"""
    batch = Batch.get(batch_id)
    if not batch:
        return jsonify({'error': 'Batch not found'}), 404
    
    qr_data = f"https://spice-sense.vercel.app/track/{batch_id}"
    qr_base64 = generate_qr_code(qr_data)
    
    return jsonify({
        'success': True,
        'batch_id': batch_id,
        'qr_url': qr_data,
        'qr_code_base64': qr_base64
    })


# ============================================================
# SUPPLY CHAIN EVENT ENDPOINTS
# ============================================================

@app.route('/api/events', methods=['POST'])
@require_auth  # Any authenticated user can record events
def record_event():
    """Record a supply chain event - Requires authentication"""
    data = request.get_json()
    
    required = ['batch_id', 'event_type', 'stage']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Verify batch exists
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
    
    return jsonify({
        'success': True,
        'message': 'Event recorded',
        'event': event
    }), 201


@app.route('/api/batches/<batch_id>/events', methods=['GET'])
def get_batch_events(batch_id):
    """Get all events for a batch"""
    events = SupplyChainEvent.get_by_batch(batch_id)
    return jsonify({
        'success': True,
        'batch_id': batch_id,
        'count': len(events),
        'events': events
    })


@app.route('/api/batches/<batch_id>/verify', methods=['GET'])
def verify_batch_chain(batch_id):
    """Verify hash chain integrity for a batch"""
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


# ============================================================
# PURITY TEST ENDPOINTS
# ============================================================

@app.route('/api/purity-tests', methods=['POST'])
@require_role(['admin', 'tester'])  # Only admins and testers can record purity tests
def record_purity_test():
    """Record a purity test for a batch - Requires tester or admin role"""
    data = request.get_json()
    
    required = ['batch_id', 'purity_percent', 'quality_grade']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Verify batch exists
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
    
    return jsonify({
        'success': True,
        'message': 'Purity test recorded',
        'test': test
    }), 201


@app.route('/api/batches/<batch_id>/purity-tests', methods=['GET'])
def get_batch_purity_tests(batch_id):
    """Get all purity tests for a batch"""
    tests = PurityTest.get_by_batch(batch_id)
    return jsonify({
        'success': True,
        'batch_id': batch_id,
        'count': len(tests),
        'tests': tests
    })


# ============================================================
# HANDLER ENDPOINTS
# ============================================================

@app.route('/api/handlers', methods=['GET'])
def get_handlers():
    """Get all handlers"""
    handler_type = request.args.get('type')
    handlers = Handler.get_all(handler_type)
    return jsonify({
        'success': True,
        'count': len(handlers),
        'handlers': handlers
    })


@app.route('/api/handlers', methods=['POST'])
@require_role(['admin'])  # Only admins can register handlers
def create_handler():
    """Register a new handler - Admin only"""
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
    
    return jsonify({
        'success': True,
        'message': 'Handler registered',
        'handler': handler
    }), 201


@app.route('/api/handlers/<handler_id>', methods=['GET'])
def get_handler(handler_id):
    """Get handler by ID"""
    handler = Handler.get(handler_id)
    if not handler:
        return jsonify({'error': 'Handler not found'}), 404
    
    return jsonify({
        'success': True,
        'handler': handler
    })


# ============================================================
# TRANSFER ENDPOINT (Common operation)
# ============================================================

@app.route('/api/transfer', methods=['POST'])
def transfer_batch():
    """Transfer batch from one handler to another"""
    data = request.get_json()
    
    required = ['batch_id', 'from_handler', 'to_handler', 'to_handler_type', 'location']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    batch = Batch.get(data['batch_id'])
    if not batch:
        return jsonify({'error': 'Batch not found'}), 404
    
    # Determine new stage based on handler type
    stage_map = {
        'processor': 'processing',
        'tester': 'tested',
        'packager': 'packaged',
        'distributor': 'at_distributor',
        'retailer': 'at_retailer'
    }
    new_stage = stage_map.get(data['to_handler_type'], 'in_transit')
    
    event = SupplyChainEvent.record(
        batch_id=data['batch_id'],
        event_type='transferred',
        stage=new_stage,
        location=data['location'],
        handler_name=data['to_handler'],
        handler_type=data['to_handler_type'],
        details=f"Transferred from {data['from_handler']} to {data['to_handler']}",
        quantity_kg=data.get('quantity_kg'),
        temperature_c=data.get('temperature_c'),
        humidity_percent=data.get('humidity_percent')
    )
    
    return jsonify({
        'success': True,
        'message': f"Batch transferred to {data['to_handler']}",
        'new_stage': new_stage,
        'event': event
    })


# ============================================================
# CONSUMER TRACKING ENDPOINT
# ============================================================

@app.route('/api/track/<batch_id>', methods=['GET'])
def track_batch_public(batch_id):
    """Public endpoint for consumers to track a batch"""
    journey = Batch.get_full_journey(batch_id)
    if not journey:
        return jsonify({'error': 'Batch not found'}), 404
    
    # Simplify for consumer view
    batch = journey['batch']
    events = journey['journey']
    tests = journey['purity_tests']
    
    # Build timeline
    timeline = []
    for event in events:
        timeline.append({
            'date': event['timestamp'][:10],
            'time': event['timestamp'][11:19],
            'stage': event['stage'].replace('_', ' ').title(),
            'location': event['location'] or 'N/A',
            'handler': event['handler_name'] or 'N/A',
            'details': event['details']
        })
    
    # Latest purity
    latest_purity = tests[-1] if tests else None
    
    return jsonify({
        'success': True,
        'product': {
            'type': batch['spice_type'].title(),
            'batch_id': batch['id'],
            'origin': batch['origin_farm'],
            'location': batch['origin_location'],
            'harvest_date': batch['harvest_date'],
            'current_stage': batch['current_stage'].replace('_', ' ').title(),
            'current_holder': batch['current_holder']
        },
        'quality': {
            'purity_percent': latest_purity['purity_percent'] if latest_purity else None,
            'grade': latest_purity['quality_grade'] if latest_purity else 'Not tested',
            'test_date': latest_purity['test_timestamp'][:10] if latest_purity else None
        },
        'timeline': timeline,
        'verified': journey['chain_valid']
    })


# ============================================================
# STATS ENDPOINT
# ============================================================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    batches = Batch.get_all(active_only=False)
    handlers = Handler.get_all()
    
    # Count by stage
    stage_counts = {}
    for batch in batches:
        stage = batch['current_stage']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    # Count by handler type
    handler_counts = {}
    for handler in handlers:
        h_type = handler['handler_type']
        handler_counts[h_type] = handler_counts.get(h_type, 0) + 1
    
    return jsonify({
        'success': True,
        'stats': {
            'total_batches': len(batches),
            'active_batches': len([b for b in batches if b['is_active']]),
            'total_handlers': len(handlers),
            'batches_by_stage': stage_counts,
            'handlers_by_type': handler_counts
        }
    })


# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.route('/api/auth/register', methods=['POST'])
@require_role(['admin'])  # Only admins can register new users
def register_user():
    """Register a new user - Admin only"""
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
        return jsonify({
            'success': True,
            'message': 'User created successfully',
            'user': user
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/auth/set-role', methods=['POST'])
@require_role(['admin'])  # Only admins can set roles
def update_user_role():
    """Update user role - Admin only"""
    data = request.get_json()
    
    if 'uid' not in data or 'role' not in data:
        return jsonify({'error': 'Missing uid or role'}), 400
    
    try:
        set_user_role(data['uid'], data['role'])
        return jsonify({
            'success': True,
            'message': f"Role updated to {data['role']}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/auth/users', methods=['GET'])
@require_role(['admin'])  # Only admins can list users
def get_all_users():
    """List all users - Admin only"""
    try:
        users = list_all_users()
        return jsonify({
            'success': True,
            'count': len(users),
            'users': users
        })
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
# HEALTH CHECK
# ============================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'service': 'Spice Traceability API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("\n" + "="*50)
    print("🌶️  SPICE TRACEABILITY API")
    print("="*50)
    print(f"API running on: http://localhost:{port}")
    print(f"Health check: http://localhost:{port}/api/health")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)
