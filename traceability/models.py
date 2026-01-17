"""
Spice Traceability - Database Models
Tracks spices (turmeric) through the entire supply chain
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'traceability.db')


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Batches table - each batch of turmeric
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            id TEXT PRIMARY KEY,
            spice_type TEXT NOT NULL DEFAULT 'turmeric',
            origin_farm TEXT NOT NULL,
            origin_location TEXT NOT NULL,
            harvest_date TEXT NOT NULL,
            quantity_kg REAL NOT NULL,
            farmer_name TEXT,
            farmer_contact TEXT,
            initial_purity REAL,
            current_stage TEXT DEFAULT 'harvested',
            current_holder TEXT,
            created_at TEXT NOT NULL,
            qr_code_data TEXT,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    # Supply chain events - immutable log of all events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS supply_chain_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            stage TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            location TEXT,
            handler_name TEXT,
            handler_id TEXT,
            handler_type TEXT,
            details TEXT,
            purity_score REAL,
            purity_grade TEXT,
            quantity_kg REAL,
            temperature_c REAL,
            humidity_percent REAL,
            previous_hash TEXT,
            event_hash TEXT NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES batches(id)
        )
    ''')
    
    # Purity tests linked to batches
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS purity_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            test_timestamp TEXT NOT NULL,
            mq135_aqi REAL,
            mq3_voltage REAL,
            gas_resistance_kOhm REAL,
            purity_percent REAL NOT NULL,
            quality_grade TEXT NOT NULL,
            binary_classification TEXT,
            confidence REAL,
            tester_name TEXT,
            test_location TEXT,
            notes TEXT,
            FOREIGN KEY (batch_id) REFERENCES batches(id)
        )
    ''')
    
    # Handlers/Actors in the supply chain
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS handlers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            handler_type TEXT NOT NULL,
            organization TEXT,
            location TEXT,
            contact_email TEXT,
            contact_phone TEXT,
            license_number TEXT,
            is_verified INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized")


# ============================================================
# HASH FUNCTIONS FOR DATA INTEGRITY
# ============================================================

def calculate_hash(data: Dict[str, Any], previous_hash: str = "") -> str:
    """Calculate SHA-256 hash for data integrity (like blockchain)"""
    data_string = json.dumps(data, sort_keys=True) + previous_hash
    return hashlib.sha256(data_string.encode()).hexdigest()


def generate_batch_id(farm: str, date: str) -> str:
    """Generate unique batch ID"""
    unique_string = f"{farm}-{date}-{datetime.now().timestamp()}"
    return "BTH-" + hashlib.md5(unique_string.encode()).hexdigest()[:12].upper()


def generate_handler_id(name: str, handler_type: str) -> str:
    """Generate unique handler ID"""
    unique_string = f"{name}-{handler_type}-{datetime.now().timestamp()}"
    prefix = handler_type[:3].upper()
    return f"{prefix}-" + hashlib.md5(unique_string.encode()).hexdigest()[:8].upper()


# ============================================================
# BATCH OPERATIONS
# ============================================================

class Batch:
    """Represents a batch of spice in the supply chain"""
    
    STAGES = [
        'harvested',      # At farm
        'processing',     # Being processed/ground
        'tested',         # Quality tested
        'packaged',       # Packaged for distribution
        'in_transit',     # Being transported
        'at_distributor', # At distribution center
        'at_retailer',    # At retail store
        'sold'            # Sold to consumer
    ]
    
    @staticmethod
    def create(
        origin_farm: str,
        origin_location: str,
        harvest_date: str,
        quantity_kg: float,
        farmer_name: str = None,
        farmer_contact: str = None,
        initial_purity: float = None,
        spice_type: str = 'turmeric'
    ) -> Dict[str, Any]:
        """Create a new batch"""
        conn = get_db()
        cursor = conn.cursor()
        
        batch_id = generate_batch_id(origin_farm, harvest_date)
        created_at = datetime.now().isoformat()
        
        # QR code data
        qr_data = json.dumps({
            'batch_id': batch_id,
            'spice': spice_type,
            'origin': origin_farm,
            'harvest': harvest_date,
            'track_url': f'https://spice-sense.vercel.app/track/{batch_id}'
        })
        
        cursor.execute('''
            INSERT INTO batches 
            (id, spice_type, origin_farm, origin_location, harvest_date, 
             quantity_kg, farmer_name, farmer_contact, initial_purity, 
             created_at, qr_code_data, current_holder)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            batch_id, spice_type, origin_farm, origin_location, harvest_date,
            quantity_kg, farmer_name, farmer_contact, initial_purity,
            created_at, qr_data, farmer_name or origin_farm
        ))
        
        conn.commit()
        
        # Record initial event
        SupplyChainEvent.record(
            batch_id=batch_id,
            event_type='batch_created',
            stage='harvested',
            location=origin_location,
            handler_name=farmer_name or origin_farm,
            handler_type='farmer',
            details=f"Batch created: {quantity_kg}kg of {spice_type} harvested",
            quantity_kg=quantity_kg
        )
        
        batch = Batch.get(batch_id)
        conn.close()
        return batch
    
    @staticmethod
    def get(batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch by ID"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM batches WHERE id = ?', (batch_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    @staticmethod
    def get_all(active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all batches"""
        conn = get_db()
        cursor = conn.cursor()
        if active_only:
            cursor.execute('SELECT * FROM batches WHERE is_active = 1 ORDER BY created_at DESC')
        else:
            cursor.execute('SELECT * FROM batches ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    @staticmethod
    def update_stage(batch_id: str, new_stage: str, holder: str = None):
        """Update batch stage"""
        conn = get_db()
        cursor = conn.cursor()
        
        if holder:
            cursor.execute(
                'UPDATE batches SET current_stage = ?, current_holder = ? WHERE id = ?',
                (new_stage, holder, batch_id)
            )
        else:
            cursor.execute(
                'UPDATE batches SET current_stage = ? WHERE id = ?',
                (new_stage, batch_id)
            )
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_full_journey(batch_id: str) -> Dict[str, Any]:
        """Get complete journey of a batch"""
        batch = Batch.get(batch_id)
        if not batch:
            return None
        
        events = SupplyChainEvent.get_by_batch(batch_id)
        purity_tests = PurityTest.get_by_batch(batch_id)
        
        return {
            'batch': batch,
            'journey': events,
            'purity_tests': purity_tests,
            'total_events': len(events),
            'chain_valid': SupplyChainEvent.verify_chain(batch_id)
        }


# ============================================================
# SUPPLY CHAIN EVENTS
# ============================================================

class SupplyChainEvent:
    """Immutable supply chain events with hash chain"""
    
    EVENT_TYPES = [
        'batch_created',
        'transferred',
        'processing_started',
        'processing_completed',
        'quality_tested',
        'packaged',
        'shipped',
        'received',
        'stored',
        'sold'
    ]
    
    @staticmethod
    def record(
        batch_id: str,
        event_type: str,
        stage: str,
        location: str = None,
        handler_name: str = None,
        handler_id: str = None,
        handler_type: str = None,
        details: str = None,
        purity_score: float = None,
        purity_grade: str = None,
        quantity_kg: float = None,
        temperature_c: float = None,
        humidity_percent: float = None
    ) -> Dict[str, Any]:
        """Record a new supply chain event"""
        conn = get_db()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Get previous hash
        cursor.execute(
            'SELECT event_hash FROM supply_chain_events WHERE batch_id = ? ORDER BY id DESC LIMIT 1',
            (batch_id,)
        )
        row = cursor.fetchone()
        previous_hash = row['event_hash'] if row else "GENESIS"
        
        # Create event data for hashing
        event_data = {
            'batch_id': batch_id,
            'event_type': event_type,
            'stage': stage,
            'timestamp': timestamp,
            'location': location,
            'handler_name': handler_name,
            'details': details,
            'purity_score': purity_score,
            'quantity_kg': quantity_kg
        }
        
        event_hash = calculate_hash(event_data, previous_hash)
        
        cursor.execute('''
            INSERT INTO supply_chain_events
            (batch_id, event_type, stage, timestamp, location, handler_name,
             handler_id, handler_type, details, purity_score, purity_grade,
             quantity_kg, temperature_c, humidity_percent, previous_hash, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            batch_id, event_type, stage, timestamp, location, handler_name,
            handler_id, handler_type, details, purity_score, purity_grade,
            quantity_kg, temperature_c, humidity_percent, previous_hash, event_hash
        ))
        
        conn.commit()
        
        # Update batch stage
        Batch.update_stage(batch_id, stage, handler_name)
        
        # Get created event
        event_id = cursor.lastrowid
        cursor.execute('SELECT * FROM supply_chain_events WHERE id = ?', (event_id,))
        event = dict(cursor.fetchone())
        conn.close()
        
        return event
    
    @staticmethod
    def get_by_batch(batch_id: str) -> List[Dict[str, Any]]:
        """Get all events for a batch"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM supply_chain_events WHERE batch_id = ? ORDER BY timestamp ASC',
            (batch_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    @staticmethod
    def verify_chain(batch_id: str) -> bool:
        """Verify the hash chain integrity for a batch"""
        events = SupplyChainEvent.get_by_batch(batch_id)
        
        if not events:
            return True
        
        for i, event in enumerate(events):
            # Recreate hash
            event_data = {
                'batch_id': event['batch_id'],
                'event_type': event['event_type'],
                'stage': event['stage'],
                'timestamp': event['timestamp'],
                'location': event['location'],
                'handler_name': event['handler_name'],
                'details': event['details'],
                'purity_score': event['purity_score'],
                'quantity_kg': event['quantity_kg']
            }
            
            expected_hash = calculate_hash(event_data, event['previous_hash'])
            
            if expected_hash != event['event_hash']:
                print(f"❌ Chain broken at event {event['id']}")
                return False
        
        return True


# ============================================================
# PURITY TESTS
# ============================================================

class PurityTest:
    """Purity test records linked to batches"""
    
    @staticmethod
    def record(
        batch_id: str,
        purity_percent: float,
        quality_grade: str,
        mq135_aqi: float = None,
        mq3_voltage: float = None,
        gas_resistance_kOhm: float = None,
        binary_classification: str = None,
        confidence: float = None,
        tester_name: str = None,
        test_location: str = None,
        notes: str = None
    ) -> Dict[str, Any]:
        """Record a purity test"""
        conn = get_db()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO purity_tests
            (batch_id, test_timestamp, mq135_aqi, mq3_voltage, gas_resistance_kOhm,
             purity_percent, quality_grade, binary_classification, confidence,
             tester_name, test_location, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            batch_id, timestamp, mq135_aqi, mq3_voltage, gas_resistance_kOhm,
            purity_percent, quality_grade, binary_classification, confidence,
            tester_name, test_location, notes
        ))
        
        conn.commit()
        test_id = cursor.lastrowid
        
        # Also record as supply chain event
        SupplyChainEvent.record(
            batch_id=batch_id,
            event_type='quality_tested',
            stage='tested',
            location=test_location,
            handler_name=tester_name,
            handler_type='tester',
            details=f"Purity test: {purity_percent}% - {quality_grade}",
            purity_score=purity_percent,
            purity_grade=quality_grade
        )
        
        cursor.execute('SELECT * FROM purity_tests WHERE id = ?', (test_id,))
        test = dict(cursor.fetchone())
        conn.close()
        
        return test
    
    @staticmethod
    def get_by_batch(batch_id: str) -> List[Dict[str, Any]]:
        """Get all purity tests for a batch"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM purity_tests WHERE batch_id = ? ORDER BY test_timestamp ASC',
            (batch_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


# ============================================================
# HANDLERS (Supply Chain Actors)
# ============================================================

class Handler:
    """Supply chain participants"""
    
    TYPES = ['farmer', 'processor', 'tester', 'packager', 'distributor', 'retailer']
    
    @staticmethod
    def create(
        name: str,
        handler_type: str,
        organization: str = None,
        location: str = None,
        contact_email: str = None,
        contact_phone: str = None,
        license_number: str = None
    ) -> Dict[str, Any]:
        """Create a new handler"""
        conn = get_db()
        cursor = conn.cursor()
        
        handler_id = generate_handler_id(name, handler_type)
        created_at = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO handlers
            (id, name, handler_type, organization, location, 
             contact_email, contact_phone, license_number, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            handler_id, name, handler_type, organization, location,
            contact_email, contact_phone, license_number, created_at
        ))
        
        conn.commit()
        cursor.execute('SELECT * FROM handlers WHERE id = ?', (handler_id,))
        handler = dict(cursor.fetchone())
        conn.close()
        
        return handler
    
    @staticmethod
    def get(handler_id: str) -> Optional[Dict[str, Any]]:
        """Get handler by ID"""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM handlers WHERE id = ?', (handler_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    @staticmethod
    def get_all(handler_type: str = None) -> List[Dict[str, Any]]:
        """Get all handlers"""
        conn = get_db()
        cursor = conn.cursor()
        if handler_type:
            cursor.execute('SELECT * FROM handlers WHERE handler_type = ?', (handler_type,))
        else:
            cursor.execute('SELECT * FROM handlers')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


# Initialize database on import
if __name__ == '__main__':
    init_db()
    print("Database ready!")
