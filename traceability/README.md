# 🌶️ Spice Traceability System

A practical, database-backed supply chain tracking system for spices (turmeric). This is a simpler alternative to blockchain that provides:

- ✅ **Immutable event logging** with hash chains
- ✅ **Complete journey tracking** from farm to consumer
- ✅ **QR code generation** for easy scanning
- ✅ **Purity test integration** with the detection system
- ✅ **Chain integrity verification**

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FARMER    │ ──▶ │  PROCESSOR  │ ──▶ │   TESTER    │ ──▶ │  PACKAGER   │
│  (Harvest)  │     │  (Grinding) │     │  (Quality)  │     │  (Packing)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
        ┌───────────────────────────────────────────────────────────┘
        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ DISTRIBUTOR │ ──▶ │  RETAILER   │ ──▶ │  CONSUMER   │
│  (Shipping) │     │   (Store)   │     │  (QR Scan)  │
└─────────────┘     └─────────────┘     └─────────────┘

        All events stored in ──▶  [ Traceability Database ]
                                         │
                                  ┌──────┴──────┐
                                  │  REST API   │
                                  │  + QR Codes │
                                  └─────────────┘
```

## Why Not Blockchain?

| Feature | Blockchain | This System |
|---------|------------|-------------|
| Immutability | ✅ Distributed | ✅ Hash chains |
| Verification | ✅ Consensus | ✅ Hash verification |
| Complexity | ❌ High | ✅ Simple |
| Cost | ❌ Gas fees | ✅ Free |
| Speed | ❌ Slow | ✅ Instant |
| Scalability | ❌ Limited | ✅ High |
| Deployment | ❌ Complex | ✅ Easy |

## Quick Start

### 1. Install dependencies

```bash
cd traceability
pip install -r requirements.txt
```

### 2. Start the API server

```bash
python api.py
```

Server runs on `http://localhost:5001`

### 3. Run the demo

```bash
python demo.py
```

This simulates a complete supply chain journey from farm to retail.

## API Endpoints

### Batches

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/batches` | List all batches |
| POST | `/api/batches` | Create new batch |
| GET | `/api/batches/{id}` | Get batch details |
| GET | `/api/batches/{id}/journey` | Get complete journey |
| GET | `/api/batches/{id}/qr` | Get QR code |
| GET | `/api/batches/{id}/verify` | Verify chain integrity |

### Events

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/events` | Record supply chain event |
| GET | `/api/batches/{id}/events` | Get batch events |

### Purity Tests

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/purity-tests` | Record purity test |
| GET | `/api/batches/{id}/purity-tests` | Get batch tests |

### Handlers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/handlers` | List all handlers |
| POST | `/api/handlers` | Register handler |
| GET | `/api/handlers/{id}` | Get handler details |

### Transfer & Track

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/transfer` | Transfer batch between handlers |
| GET | `/api/track/{id}` | Consumer tracking (public) |

## Example: Create a Batch

```python
import requests

# Create batch at farm
response = requests.post('http://localhost:5001/api/batches', json={
    'origin_farm': 'Kumar Farms',
    'origin_location': 'Erode, Tamil Nadu',
    'harvest_date': '2026-01-15',
    'quantity_kg': 500,
    'farmer_name': 'Ramesh Kumar',
    'spice_type': 'turmeric'
})

batch = response.json()['batch']
print(f"Created batch: {batch['id']}")
```

## Example: Transfer Batch

```python
# Transfer to processor
requests.post('http://localhost:5001/api/transfer', json={
    'batch_id': 'BTH-XXXXXXXXXXXX',
    'from_handler': 'Ramesh Kumar',
    'to_handler': 'Spice Masters Pvt Ltd',
    'to_handler_type': 'processor',
    'location': 'Coimbatore, Tamil Nadu'
})
```

## Example: Record Purity Test

```python
# Record test results from ESP32 sensor
requests.post('http://localhost:5001/api/purity-tests', json={
    'batch_id': 'BTH-XXXXXXXXXXXX',
    'purity_percent': 92.5,
    'quality_grade': 'BEST',
    'mq135_aqi': 45.2,
    'mq3_voltage': 0.85,
    'gas_resistance_kOhm': 52.3,
    'tester_name': 'Quality Labs India',
    'test_location': 'Chennai, Tamil Nadu'
})
```

## Hash Chain Integrity

Every event is linked to the previous one via SHA-256 hashes:

```
Event 1 (Genesis)
   hash: abc123...
          │
          ▼
Event 2 
   previous_hash: abc123...
   hash: def456...
          │
          ▼
Event 3
   previous_hash: def456...
   hash: ghi789...
```

Tampering with any event breaks the chain:

```python
# Verify chain integrity
response = requests.get('http://localhost:5001/api/batches/BTH-XXX/verify')
print(response.json()['integrity'])  # "VERIFIED ✓" or "COMPROMISED ✗"
```

## Supply Chain Stages

1. **harvested** - At farm
2. **processing** - Being processed/ground
3. **tested** - Quality tested
4. **packaged** - Packaged for distribution
5. **in_transit** - Being transported
6. **at_distributor** - At distribution center
7. **at_retailer** - At retail store
8. **sold** - Sold to consumer

## Integration with Purity Detection

When ESP32 tests a batch, record the results:

```python
# After ML prediction from ESP32
requests.post('http://localhost:5001/api/purity-tests', json={
    'batch_id': batch_id,
    'purity_percent': prediction['regression']['purity_percent'],
    'quality_grade': prediction['multiclass']['label'],
    'mq135_aqi': raw_features['mq135_aqi'],
    'mq3_voltage': raw_features['mq3_voltage'],
    'gas_resistance_kOhm': raw_features['gas_resistance_kOhm'],
    'confidence': prediction['multiclass']['confidence']
})
```

## Files

```
traceability/
├── models.py         # Database models & operations
├── api.py            # Flask REST API
├── demo.py           # Demo script
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── traceability.db   # SQLite database (auto-created)
```
