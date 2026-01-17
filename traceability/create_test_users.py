"""
Quick script to create all test users
"""
import firebase_admin
from firebase_admin import credentials, auth
import os

# Initialize Firebase
cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

# Test users to create
TEST_USERS = [
    ("farmer@spicesense.com", "farmer123", "Test Farmer", "farmer"),
    ("processor@spicesense.com", "processor123", "Test Processor", "processor"),
    ("tester@spicesense.com", "tester123", "Test Tester", "tester"),
    ("distributor@spicesense.com", "distributor123", "Test Distributor", "distributor"),
    ("retailer@spicesense.com", "retailer123", "Test Retailer", "retailer"),
    ("viewer@spicesense.com", "viewer123", "Test Viewer", "viewer"),
]

print("="*60)
print("🌶️  Creating Test Users for Spice Sense")
print("="*60)

for email, password, name, role in TEST_USERS:
    try:
        # Check if exists
        try:
            existing = auth.get_user_by_email(email)
            print(f"⚠️  {email} exists - updating role to {role}")
            auth.set_custom_user_claims(existing.uid, {'role': role})
            continue
        except auth.UserNotFoundError:
            pass
        
        # Create user
        user = auth.create_user(
            email=email,
            password=password,
            display_name=name
        )
        auth.set_custom_user_claims(user.uid, {'role': role})
        print(f"✅ Created: {email} | Password: {password} | Role: {role}")
        
    except Exception as e:
        print(f"❌ Error creating {email}: {e}")

# List all users
print("\n" + "="*60)
print("📋 ALL USERS")
print("="*60)
print(f"{'Email':<35} {'Role':<12} {'Password'}")
print("-"*60)

# Show admin
print(f"{'noelantony67@gmail.com':<35} {'admin':<12} noel67")

# Show test users
for email, password, name, role in TEST_USERS:
    print(f"{email:<35} {role:<12} {password}")

print("\n✅ All test users ready!")
print("You can now login with any of these accounts.")
