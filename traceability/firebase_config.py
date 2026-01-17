"""
Firebase Admin SDK Configuration
For server-side authentication and user management
"""
import os
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
from flask import request, jsonify

# Initialize Firebase Admin SDK
cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

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
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        
        # Expected format: "Bearer <token>"
        parts = auth_header.split()
        if len(parts) != 2 or parts[0] != 'Bearer':
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        token = parts[1]
        
        # Verify the token
        user_info = verify_token(token)
        if not user_info:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Add user info to request context
        request.user = user_info
        return f(*args, **kwargs)
    
    return decorated_function

def require_role(allowed_roles):
    """Decorator to require specific roles"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # First check auth
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
            
            # Check role from custom claims
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
        # Set role
        set_user_role(user.uid, role)
        return {'uid': user.uid, 'email': user.email, 'role': role}
    except Exception as e:
        raise Exception(f"Failed to create user: {e}")

def get_user_by_email(email):
    """Get user info by email"""
    try:
        user = auth.get_user_by_email(email)
        return {
            'uid': user.uid,
            'email': user.email,
            'display_name': user.display_name,
            'role': user.custom_claims.get('role', 'viewer') if user.custom_claims else 'viewer'
        }
    except Exception as e:
        return None

def list_all_users():
    """List all users (admin only)"""
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
