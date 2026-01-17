"""
Demo: How Hash Chain Detects Tampering
This script shows step-by-step how tampering breaks the chain
"""

import sqlite3
import hashlib
import json
from datetime import datetime

# Use a separate test database
TEST_DB = 'tampering_demo.db'


def calculate_hash(data: dict, previous_hash: str = "") -> str:
    """Calculate SHA-256 hash"""
    data_string = json.dumps(data, sort_keys=True) + previous_hash
    return hashlib.sha256(data_string.encode()).hexdigest()


def setup_demo_db():
    """Create a fresh demo database"""
    conn = sqlite3.connect(TEST_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS events')
    cursor.execute('''
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            batch_id TEXT,
            event_type TEXT,
            stage TEXT,
            timestamp TEXT,
            location TEXT,
            handler_name TEXT,
            quantity_kg REAL,
            purity_score REAL,
            details TEXT,
            previous_hash TEXT,
            event_hash TEXT
        )
    ''')
    conn.commit()
    return conn


def add_event(conn, batch_id, event_type, stage, location, handler_name, 
              quantity_kg=None, purity_score=None, details=None):
    """Add an event with proper hash chain"""
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    # Get previous hash
    cursor.execute('SELECT event_hash FROM events WHERE batch_id = ? ORDER BY id DESC LIMIT 1', (batch_id,))
    row = cursor.fetchone()
    previous_hash = row['event_hash'] if row else "GENESIS"
    
    # Create event data for hashing (use float for numeric values)
    event_data = {
        'batch_id': batch_id,
        'event_type': event_type,
        'stage': stage,
        'timestamp': timestamp,
        'location': location,
        'handler_name': handler_name,
        'quantity_kg': float(quantity_kg) if quantity_kg else None,
        'purity_score': float(purity_score) if purity_score else None,
        'details': details
    }
    
    # Calculate hash
    event_hash = calculate_hash(event_data, previous_hash)
    
    cursor.execute('''
        INSERT INTO events (batch_id, event_type, stage, timestamp, location, 
                           handler_name, quantity_kg, purity_score, details,
                           previous_hash, event_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (batch_id, event_type, stage, timestamp, location, handler_name,
          quantity_kg, purity_score, details, previous_hash, event_hash))
    
    conn.commit()
    return event_hash


def verify_chain(conn, batch_id):
    """Verify hash chain integrity"""
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM events WHERE batch_id = ? ORDER BY id', (batch_id,))
    events = cursor.fetchall()
    
    print("\n" + "="*70)
    print("🔍 VERIFYING HASH CHAIN")
    print("="*70)
    
    for i, event in enumerate(events):
        # Rebuild event data (ensure types match what was stored)
        event_data = {
            'batch_id': event['batch_id'],
            'event_type': event['event_type'],
            'stage': event['stage'],
            'timestamp': event['timestamp'],
            'location': event['location'],
            'handler_name': event['handler_name'],
            'quantity_kg': float(event['quantity_kg']) if event['quantity_kg'] else None,
            'purity_score': float(event['purity_score']) if event['purity_score'] else None,
            'details': event['details']
        }
        
        # Recalculate hash
        calculated_hash = calculate_hash(event_data, event['previous_hash'])
        stored_hash = event['event_hash']
        
        print(f"\nEvent {i+1}: {event['event_type']}")
        print(f"  Location: {event['location']}")
        print(f"  Quantity: {event['quantity_kg']} kg")
        print(f"  Stored hash:     {stored_hash[:30]}...")
        print(f"  Calculated hash: {calculated_hash[:30]}...")
        
        if calculated_hash == stored_hash:
            print(f"  ✅ MATCH - Event {i+1} is valid")
        else:
            print(f"  ❌ MISMATCH - TAMPERING DETECTED!")
            print(f"\n{'='*70}")
            print("🚨 CHAIN INTEGRITY COMPROMISED!")
            print(f"{'='*70}")
            return False
    
    print(f"\n{'='*70}")
    print("✅ CHAIN VERIFIED - ALL EVENTS AUTHENTIC")
    print(f"{'='*70}")
    return True


def tamper_event(conn, event_id, new_quantity):
    """Simulate tampering by directly modifying database"""
    cursor = conn.cursor()
    cursor.execute('UPDATE events SET quantity_kg = ? WHERE id = ?', (new_quantity, event_id))
    conn.commit()


def main():
    print("\n" + "🌶️"*30)
    print("    HASH CHAIN TAMPERING DETECTION DEMO")
    print("🌶️"*30)
    
    # Setup
    conn = setup_demo_db()
    batch_id = "BTH-DEMO-001"
    
    # ========================================
    # PHASE 1: Create legitimate supply chain
    # ========================================
    print("\n\n" + "="*70)
    print("PHASE 1: Creating legitimate supply chain events")
    print("="*70)
    
    print("\n📝 Adding Event 1: Harvest (500 kg)...")
    h1 = add_event(conn, batch_id, "batch_created", "harvested", 
                   "Kerala Farm", "Farmer Rajan", quantity_kg=500)
    print(f"   Hash: {h1[:40]}...")
    
    print("\n📝 Adding Event 2: Processing (480 kg after waste)...")
    h2 = add_event(conn, batch_id, "processing_completed", "processing",
                   "Kochi Mill", "Spice Mills", quantity_kg=480)
    print(f"   Hash: {h2[:40]}...")
    
    print("\n📝 Adding Event 3: Quality Testing (92.5% purity)...")
    h3 = add_event(conn, batch_id, "quality_tested", "tested",
                   "Chennai Lab", "Quality Labs", quantity_kg=480, purity_score=92.5)
    print(f"   Hash: {h3[:40]}...")
    
    # ========================================
    # PHASE 2: Verify original chain
    # ========================================
    print("\n\n" + "="*70)
    print("PHASE 2: Verifying original (untampered) chain")
    print("="*70)
    
    verify_chain(conn, batch_id)
    
    # ========================================
    # PHASE 3: Simulate tampering
    # ========================================
    print("\n\n" + "="*70)
    print("PHASE 3: 😈 SIMULATING TAMPERING ATTACK")
    print("="*70)
    
    print("\n🚨 Attacker modifies Event 2:")
    print("   Original quantity: 480 kg")
    print("   Tampered quantity: 600 kg (added 120kg cheap filler!)")
    print("\n   Executing: UPDATE events SET quantity_kg = 600 WHERE id = 2")
    
    tamper_event(conn, 2, 600)  # Change 480 to 600
    
    print("\n   ✓ Database modified directly (like a hacker would)")
    
    # ========================================
    # PHASE 4: Detect tampering
    # ========================================
    print("\n\n" + "="*70)
    print("PHASE 4: Verifying chain after tampering")
    print("="*70)
    
    verify_chain(conn, batch_id)
    
    # ========================================
    # Explanation
    # ========================================
    print("\n\n" + "="*70)
    print("📚 WHAT HAPPENED?")
    print("="*70)
    print("""
    1. When Event 2 was created, we hashed:
       {"quantity_kg": 480, ...} + previous_hash
       
    2. This produced hash: "k1l2m3..." (stored in database)
    
    3. Attacker changed quantity_kg from 480 → 600
    
    4. When we verify, we recalculate:
       {"quantity_kg": 600, ...} + previous_hash
       
    5. This produces a DIFFERENT hash: "XXXYYY..."
    
    6. "XXXYYY..." ≠ "k1l2m3..." → TAMPERING DETECTED!
    
    💡 KEY INSIGHT:
       - The attacker CAN change the data
       - But they CANNOT change the hash to match
       - SHA-256 is a one-way function (can't reverse it)
       - Even a tiny change creates a completely different hash
    """)
    
    # Cleanup
    conn.close()
    import os
    os.remove(TEST_DB)
    print("\n✓ Demo database cleaned up")


if __name__ == '__main__':
    main()
