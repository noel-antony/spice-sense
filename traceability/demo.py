"""
Demo script - Simulate a complete turmeric supply chain journey
"""

import requests
import time
import json

BASE_URL = "http://localhost:5001/api"


def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"📦 {title}")
    print('='*60)
    print(json.dumps(response.json(), indent=2))


def demo_supply_chain():
    """Simulate complete supply chain flow"""
    
    print("\n" + "🌶️"*20)
    print("  TURMERIC SUPPLY CHAIN DEMO")
    print("🌶️"*20)
    
    # ========================================
    # STEP 1: Register handlers
    # ========================================
    print("\n\n" + "="*60)
    print("STEP 1: Registering Supply Chain Participants")
    print("="*60)
    
    handlers = [
        {"name": "Ramesh Kumar", "handler_type": "farmer", "organization": "Kumar Farms", "location": "Erode, Tamil Nadu"},
        {"name": "Spice Masters Pvt Ltd", "handler_type": "processor", "organization": "Spice Masters", "location": "Coimbatore, Tamil Nadu"},
        {"name": "Quality Labs India", "handler_type": "tester", "organization": "QLI", "location": "Chennai, Tamil Nadu"},
        {"name": "Fresh Pack Industries", "handler_type": "packager", "organization": "FPI", "location": "Chennai, Tamil Nadu"},
        {"name": "National Spice Distributors", "handler_type": "distributor", "organization": "NSD", "location": "Mumbai, Maharashtra"},
        {"name": "SuperMart Retail", "handler_type": "retailer", "organization": "SuperMart", "location": "Bangalore, Karnataka"},
    ]
    
    for handler in handlers:
        resp = requests.post(f"{BASE_URL}/handlers", json=handler)
        print(f"  ✓ Registered: {handler['name']} ({handler['handler_type']})")
    
    # ========================================
    # STEP 2: Create batch at farm
    # ========================================
    print("\n\n" + "="*60)
    print("STEP 2: Farmer Creates New Batch")
    print("="*60)
    
    batch_data = {
        "origin_farm": "Kumar Farms",
        "origin_location": "Erode, Tamil Nadu",
        "harvest_date": "2026-01-15",
        "quantity_kg": 500,
        "farmer_name": "Ramesh Kumar",
        "farmer_contact": "+91-98765-43210",
        "spice_type": "turmeric"
    }
    
    resp = requests.post(f"{BASE_URL}/batches", json=batch_data)
    batch = resp.json()['batch']
    batch_id = batch['id']
    
    print(f"  ✓ Batch Created: {batch_id}")
    print(f"  📍 Origin: {batch['origin_farm']}, {batch['origin_location']}")
    print(f"  ⚖️  Quantity: {batch['quantity_kg']} kg")
    
    time.sleep(1)
    
    # ========================================
    # STEP 3: Transfer to processor
    # ========================================
    print("\n\n" + "="*60)
    print("STEP 3: Transfer to Processing Unit")
    print("="*60)
    
    transfer_data = {
        "batch_id": batch_id,
        "from_handler": "Ramesh Kumar",
        "to_handler": "Spice Masters Pvt Ltd",
        "to_handler_type": "processor",
        "location": "Coimbatore, Tamil Nadu",
        "temperature_c": 28.5,
        "humidity_percent": 45
    }
    
    resp = requests.post(f"{BASE_URL}/transfer", json=transfer_data)
    print(f"  ✓ Transferred to: Spice Masters Pvt Ltd")
    print(f"  📍 Location: Coimbatore, Tamil Nadu")
    
    time.sleep(1)
    
    # Record processing events
    events = [
        {"event_type": "processing_started", "stage": "processing", "details": "Cleaning and sorting started"},
        {"event_type": "processing_completed", "stage": "processing", "details": "Grinding completed, yield: 485kg powder"},
    ]
    
    for event in events:
        event['batch_id'] = batch_id
        event['location'] = "Coimbatore, Tamil Nadu"
        event['handler_name'] = "Spice Masters Pvt Ltd"
        event['handler_type'] = "processor"
        requests.post(f"{BASE_URL}/events", json=event)
        print(f"  ✓ Event: {event['event_type']}")
        time.sleep(0.5)
    
    # ========================================
    # STEP 4: Quality Testing
    # ========================================
    print("\n\n" + "="*60)
    print("STEP 4: Quality Testing")
    print("="*60)
    
    # Transfer to tester
    transfer_data = {
        "batch_id": batch_id,
        "from_handler": "Spice Masters Pvt Ltd",
        "to_handler": "Quality Labs India",
        "to_handler_type": "tester",
        "location": "Chennai, Tamil Nadu"
    }
    resp = requests.post(f"{BASE_URL}/transfer", json=transfer_data)
    
    # Record purity test
    test_data = {
        "batch_id": batch_id,
        "purity_percent": 92.5,
        "quality_grade": "BEST",
        "mq135_aqi": 45.2,
        "mq3_voltage": 0.85,
        "gas_resistance_kOhm": 52.3,
        "binary_classification": "high_purity",
        "confidence": 98.5,
        "tester_name": "Quality Labs India",
        "test_location": "Chennai, Tamil Nadu",
        "notes": "Premium grade turmeric, excellent curcumin content"
    }
    
    resp = requests.post(f"{BASE_URL}/purity-tests", json=test_data)
    print(f"  ✓ Purity Test Completed")
    print(f"  🎯 Purity: {test_data['purity_percent']}%")
    print(f"  ⭐ Grade: {test_data['quality_grade']}")
    
    time.sleep(1)
    
    # ========================================
    # STEP 5: Packaging
    # ========================================
    print("\n\n" + "="*60)
    print("STEP 5: Packaging")
    print("="*60)
    
    transfer_data = {
        "batch_id": batch_id,
        "from_handler": "Quality Labs India",
        "to_handler": "Fresh Pack Industries",
        "to_handler_type": "packager",
        "location": "Chennai, Tamil Nadu"
    }
    resp = requests.post(f"{BASE_URL}/transfer", json=transfer_data)
    
    event = {
        "batch_id": batch_id,
        "event_type": "packaged",
        "stage": "packaged",
        "location": "Chennai, Tamil Nadu",
        "handler_name": "Fresh Pack Industries",
        "handler_type": "packager",
        "details": "Packed into 500g retail packs, 970 units produced"
    }
    requests.post(f"{BASE_URL}/events", json=event)
    print(f"  ✓ Packaged: 970 units of 500g packs")
    
    time.sleep(1)
    
    # ========================================
    # STEP 6: Distribution
    # ========================================
    print("\n\n" + "="*60)
    print("STEP 6: Distribution")
    print("="*60)
    
    transfer_data = {
        "batch_id": batch_id,
        "from_handler": "Fresh Pack Industries",
        "to_handler": "National Spice Distributors",
        "to_handler_type": "distributor",
        "location": "Mumbai, Maharashtra",
        "temperature_c": 25.0,
        "humidity_percent": 40
    }
    resp = requests.post(f"{BASE_URL}/transfer", json=transfer_data)
    print(f"  ✓ Shipped to: National Spice Distributors")
    print(f"  📍 Location: Mumbai, Maharashtra")
    
    time.sleep(1)
    
    # ========================================
    # STEP 7: Retail
    # ========================================
    print("\n\n" + "="*60)
    print("STEP 7: Retail Distribution")
    print("="*60)
    
    transfer_data = {
        "batch_id": batch_id,
        "from_handler": "National Spice Distributors",
        "to_handler": "SuperMart Retail",
        "to_handler_type": "retailer",
        "location": "Bangalore, Karnataka"
    }
    resp = requests.post(f"{BASE_URL}/transfer", json=transfer_data)
    print(f"  ✓ Received by: SuperMart Retail")
    print(f"  📍 Location: Bangalore, Karnataka")
    
    # ========================================
    # FINAL: View Complete Journey
    # ========================================
    print("\n\n" + "="*60)
    print("COMPLETE JOURNEY - Consumer View")
    print("="*60)
    
    resp = requests.get(f"{BASE_URL}/track/{batch_id}")
    track_data = resp.json()
    
    print(f"\n🌶️ Product: {track_data['product']['type']}")
    print(f"🏷️ Batch ID: {track_data['product']['batch_id']}")
    print(f"🌾 Origin: {track_data['product']['origin']}, {track_data['product']['location']}")
    print(f"📅 Harvested: {track_data['product']['harvest_date']}")
    print(f"📍 Current: {track_data['product']['current_stage']} - {track_data['product']['current_holder']}")
    
    print(f"\n⭐ Quality:")
    print(f"   Purity: {track_data['quality']['purity_percent']}%")
    print(f"   Grade: {track_data['quality']['grade']}")
    
    print(f"\n📜 Timeline:")
    for event in track_data['timeline']:
        print(f"   {event['date']} | {event['stage']:20} | {event['location']}")
    
    print(f"\n✅ Chain Verified: {track_data['verified']}")
    
    # ========================================
    # Verify chain integrity
    # ========================================
    print("\n\n" + "="*60)
    print("VERIFY CHAIN INTEGRITY")
    print("="*60)
    
    resp = requests.get(f"{BASE_URL}/batches/{batch_id}/verify")
    verify_data = resp.json()
    print(f"  {verify_data['integrity']}")
    
    # Get QR Code
    print("\n\n" + "="*60)
    print("QR CODE FOR CONSUMER")
    print("="*60)
    
    resp = requests.get(f"{BASE_URL}/batches/{batch_id}/qr")
    qr_data = resp.json()
    print(f"  Scan URL: {qr_data['qr_url']}")
    print(f"  (QR Code image generated as base64)")
    
    print("\n\n" + "🎉"*20)
    print("  DEMO COMPLETE!")
    print("🎉"*20)
    print(f"\n  Batch {batch_id} successfully tracked through entire supply chain!")
    print(f"  Consumer can verify authenticity at: {qr_data['qr_url']}")


if __name__ == '__main__':
    demo_supply_chain()
