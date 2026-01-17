"""
Create First Admin User
Run this once to create your first admin account
"""
import firebase_admin
from firebase_admin import credentials, auth
import os

# Initialize Firebase if not already
cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

def create_admin_user(email, password, display_name="Admin"):
    """Create an admin user"""
    try:
        # Check if user already exists
        try:
            existing = auth.get_user_by_email(email)
            print(f"User {email} already exists!")
            print(f"UID: {existing.uid}")
            
            # Update role to admin
            auth.set_custom_user_claims(existing.uid, {'role': 'admin'})
            print(f"Role updated to: admin")
            return existing.uid
            
        except auth.UserNotFoundError:
            pass
        
        # Create user
        user = auth.create_user(
            email=email,
            password=password,
            display_name=display_name
        )
        print(f"✅ User created: {email}")
        print(f"   UID: {user.uid}")
        
        # Set admin role
        auth.set_custom_user_claims(user.uid, {'role': 'admin'})
        print(f"   Role: admin")
        
        return user.uid
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == '__main__':
    print("="*50)
    print("🌶️  SPICE SENSE - Create Admin User")
    print("="*50)
    
    email = input("\nEnter admin email: ").strip()
    password = input("Enter password (min 6 chars): ").strip()
    name = input("Enter display name (or press Enter for 'Admin'): ").strip() or "Admin"
    
    if len(password) < 6:
        print("❌ Password must be at least 6 characters!")
    else:
        print("\nCreating admin user...")
        uid = create_admin_user(email, password, name)
        
        if uid:
            print("\n" + "="*50)
            print("✅ Admin user ready!")
            print("="*50)
            print(f"\nYou can now login at the frontend with:")
            print(f"   Email: {email}")
            print(f"   Password: {password}")
