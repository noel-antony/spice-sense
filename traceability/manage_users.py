"""
User Management Script
Create, list, and manage users with different roles
"""
import firebase_admin
from firebase_admin import credentials, auth
import os

# Initialize Firebase
cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

VALID_ROLES = ['admin', 'farmer', 'processor', 'tester', 'distributor', 'retailer', 'viewer']

def create_user(email, password, display_name, role):
    """Create a new user with role"""
    if role not in VALID_ROLES:
        print(f"❌ Invalid role. Choose from: {VALID_ROLES}")
        return None
    
    try:
        # Check if exists
        try:
            existing = auth.get_user_by_email(email)
            print(f"⚠️  User {email} already exists! Updating role to {role}...")
            auth.set_custom_user_claims(existing.uid, {'role': role})
            return existing.uid
        except auth.UserNotFoundError:
            pass
        
        # Create new user
        user = auth.create_user(
            email=email,
            password=password,
            display_name=display_name
        )
        auth.set_custom_user_claims(user.uid, {'role': role})
        print(f"✅ Created: {email} | Role: {role}")
        return user.uid
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def list_users():
    """List all users"""
    print("\n" + "="*70)
    print("📋 ALL USERS")
    print("="*70)
    print(f"{'Email':<35} {'Name':<15} {'Role':<12} {'UID'}")
    print("-"*70)
    
    for user in auth.list_users().iterate_all():
        role = user.custom_claims.get('role', 'viewer') if user.custom_claims else 'viewer'
        name = user.display_name or '-'
        print(f"{user.email:<35} {name:<15} {role:<12} {user.uid[:20]}...")

def delete_user(email):
    """Delete a user"""
    try:
        user = auth.get_user_by_email(email)
        auth.delete_user(user.uid)
        print(f"✅ Deleted user: {email}")
    except auth.UserNotFoundError:
        print(f"❌ User not found: {email}")
    except Exception as e:
        print(f"❌ Error: {e}")

def update_role(email, new_role):
    """Update user's role"""
    if new_role not in VALID_ROLES:
        print(f"❌ Invalid role. Choose from: {VALID_ROLES}")
        return
    
    try:
        user = auth.get_user_by_email(email)
        auth.set_custom_user_claims(user.uid, {'role': new_role})
        print(f"✅ Updated {email} role to: {new_role}")
    except auth.UserNotFoundError:
        print(f"❌ User not found: {email}")
    except Exception as e:
        print(f"❌ Error: {e}")

def create_test_users():
    """Create sample users for each role"""
    test_users = [
        ("farmer@test.com", "test123", "Test Farmer", "farmer"),
        ("processor@test.com", "test123", "Test Processor", "processor"),
        ("tester@test.com", "test123", "Test Tester", "tester"),
        ("distributor@test.com", "test123", "Test Distributor", "distributor"),
        ("retailer@test.com", "test123", "Test Retailer", "retailer"),
        ("viewer@test.com", "test123", "Test Viewer", "viewer"),
    ]
    
    print("\n🔧 Creating test users...")
    print("-"*50)
    for email, password, name, role in test_users:
        create_user(email, password, name, role)

def interactive_menu():
    """Interactive menu"""
    while True:
        print("\n" + "="*50)
        print("🌶️  SPICE SENSE - User Management")
        print("="*50)
        print("1. List all users")
        print("2. Create new user")
        print("3. Create test users (all roles)")
        print("4. Update user role")
        print("5. Delete user")
        print("6. Exit")
        print("-"*50)
        
        choice = input("Choose option (1-6): ").strip()
        
        if choice == '1':
            list_users()
            
        elif choice == '2':
            print(f"\nAvailable roles: {VALID_ROLES}")
            email = input("Email: ").strip()
            password = input("Password (min 6 chars): ").strip()
            name = input("Display name: ").strip()
            role = input("Role: ").strip()
            
            if len(password) >= 6:
                create_user(email, password, name, role)
            else:
                print("❌ Password must be at least 6 characters!")
                
        elif choice == '3':
            create_test_users()
            
        elif choice == '4':
            print(f"\nAvailable roles: {VALID_ROLES}")
            email = input("Email: ").strip()
            role = input("New role: ").strip()
            update_role(email, role)
            
        elif choice == '5':
            email = input("Email to delete: ").strip()
            confirm = input(f"Are you sure you want to delete {email}? (yes/no): ").strip()
            if confirm.lower() == 'yes':
                delete_user(email)
                
        elif choice == '6':
            print("Bye! 👋")
            break
        else:
            print("Invalid choice!")

if __name__ == '__main__':
    interactive_menu()
