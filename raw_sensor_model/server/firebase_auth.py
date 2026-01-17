"""
Firebase Admin SDK Configuration for Production
Uses environment variable for credentials
"""
import os
import json
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
from flask import request, jsonify

def init_firebase():
    """Initialize Firebase from environment variable or file"""
    if firebase_admin._apps:
        return  # Already initialized
    
    # Try environment variable first (for production)
    firebase_creds = os.environ.get('FIREBASE_CREDENTIALS')
    
    if firebase_creds:
        # Parse JSON from environment variable
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("✓ Firebase initialized from environment variable")
    else:
        # Try local file (for development)
        cred_path = os.path.join(os.path.dirname(__file__), '..', '..', 'traceability', 'serviceAccountKey.json')
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print("✓ Firebase initialized from local file")
        else:
            print("⚠ Firebase not initialized - no credentials found")
            print("  Set FIREBASE_CREDENTIALS env var or add serviceAccountKey.json")

# Initialize on import
init_firebase()

def verify_token(id_token):
    """Verify Firebase ID token and return user info"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None

def require_auth(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        
        parts = auth_header.split()
        if len(parts) != 2 or parts[0] != 'Bearer':
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        token = parts[1]
        user_info = verify_token(token)
        
        if not user_info:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.user = user_info
        return f(*args, **kwargs)
    
    return decorated_function

def require_role(allowed_roles):
    """Decorator to require specific roles"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({'error': 'No authorization header'}), 401
            
            parts = auth_header.split()
            if len(parts) != 2 or parts[0] != 'Bearer':
                return jsonify({'error': 'Invalid authorization header format'}), 401
            
            token = parts[1]
            user_info = verify_token(token)
            
            if not user_info:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            user_role = user_info.get('role', 'viewer')
            
            if user_role not in allowed_roles:
                return jsonify({
                    'error': f'Access denied. Required roles: {allowed_roles}, your role: {user_role}'
                }), 403
            
            request.user = user_info
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def set_user_role(uid, role):
    """Set custom claims (role) for a user"""
    valid_roles = ['admin', 'farmer', 'processor', 'tester', 'distributor', 'retailer', 'viewer']
    
    if role not in valid_roles:
        raise ValueError(f"Invalid role. Must be one of: {valid_roles}")
    
    auth.set_custom_user_claims(uid, {'role': role})
    return True

def create_user(email, password, display_name=None, role='viewer'):
    """Create a new user with optional role"""
    try:
        user = auth.create_user(
            email=email,
            password=password,
            display_name=display_name
        )
        set_user_role(user.uid, role)
        return {'uid': user.uid, 'email': user.email, 'role': role}
    except Exception as e:
        raise Exception(f"Failed to create user: {e}")

def list_all_users():
    """List all users"""
    users = []
    page = auth.list_users()
    
    for user in page.iterate_all():
        users.append({
            'uid': user.uid,
            'email': user.email,
            'display_name': user.display_name,
            'role': user.custom_claims.get('role', 'viewer') if user.custom_claims else 'viewer'
        })
    
    return users
