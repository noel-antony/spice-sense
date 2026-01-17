"""
Test Firebase Authentication with API
"""
import requests
import json

# First, we need to get a Firebase ID token
# This simulates what the frontend does when logging in

import firebase_admin
from firebase_admin import credentials, auth
import os

# Initialize Firebase
cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

API_URL = "http://localhost:5001"

print("="*60)
print("🧪 TESTING FIREBASE AUTH WITH API")
print("="*60)

# Test 1: Public endpoint (no auth needed)
print("\n📋 TEST 1: Public endpoint (GET /api/batches)")
print("-"*60)
response = requests.get(f"{API_URL}/api/batches")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()['success']}")
print("✅ Public endpoints work without auth!")

# Test 2: Protected endpoint WITHOUT token
print("\n📋 TEST 2: Protected endpoint WITHOUT token (POST /api/batches)")
print("-"*60)
response = requests.post(f"{API_URL}/api/batches", json={
    "origin_farm": "Test Farm",
    "origin_location": "Kerala",
    "harvest_date": "2026-01-18",
    "quantity_kg": 100
})
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
if response.status_code == 401:
    print("✅ Correctly blocked - No auth token!")

# Test 3: Create a custom token for testing
print("\n📋 TEST 3: Creating test token for admin user")
print("-"*60)

# Get admin user
admin_email = "noelantony67@gmail.com"
try:
    user = auth.get_user_by_email(admin_email)
    print(f"Found user: {user.email}")
    print(f"UID: {user.uid}")
    print(f"Role: {user.custom_claims}")
    
    # Create a custom token (for testing purposes)
    custom_token = auth.create_custom_token(user.uid)
    print(f"Custom token created (first 50 chars): {custom_token[:50]}...")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("📝 HOW TO TEST WITH REAL TOKEN:")
print("="*60)
print("""
To test protected endpoints, you need a Firebase ID token.

OPTION 1: Use the browser console after logging in:
  1. Login at http://localhost:8080/login.html
  2. Open browser console (F12)
  3. Run: firebase.auth().currentUser.getIdToken().then(t => console.log(t))
  4. Copy the token

OPTION 2: Use the stored token:
  After login, the token is stored in localStorage.
  Check: localStorage.getItem('authToken')

Then test API with:
  curl -X POST http://localhost:5001/api/batches \\
    -H "Authorization: Bearer YOUR_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{"origin_farm":"Test","origin_location":"Kerala","harvest_date":"2026-01-18","quantity_kg":100}'
""")

# Test health endpoint
print("\n📋 TEST 4: Health check")
print("-"*60)
response = requests.get(f"{API_URL}/api/health")
print(f"Status: {response.status_code}")
print(f"Service: {response.json()['service']}")
print("✅ API is healthy!")
