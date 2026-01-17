"""
Recalibration Analysis for Spice Purity Detection
Compares training data with current test readings to find calibration offset
"""

import pandas as pd
import numpy as np

# Training data statistics
training_stats = {
    'sensor_combined': {
        100: {'min': 33.13, 'max': 41.63, 'mean': 37.35},
        75:  {'min': 42.21, 'max': 48.37, 'mean': 47.54},
        50:  {'min': 42.72, 'max': 53.52, 'mean': 51.34},
        25:  {'min': 47.65, 'max': 57.70, 'mean': 55.49}
    },
    'mq135_ratio': {
        100: {'min': 0.49, 'max': 1.20, 'mean': 0.90},
        75:  {'min': 1.51, 'max': 1.65, 'mean': 1.58},
        50:  {'min': 1.62, 'max': 1.84, 'mean': 1.76},
        25:  {'min': 1.94, 'max': 2.01, 'mean': 1.97}
    }
}

# Current test readings (from your attached files)
current_readings = {
    100: {'sensor_combined': 42.0, 'mq135_ratio': 1.45},
    50:  {'sensor_combined': 43.5, 'mq135_ratio': 1.47},
    25:  {'sensor_combined': 52.0, 'mq135_ratio': 1.97},
    0:   {'sensor_combined': 345.0, 'mq135_ratio': 31.0}  # Pure maida
}

print("=" * 70)
print("CALIBRATION DRIFT ANALYSIS")
print("=" * 70)

print("\n📊 sensor_combined Analysis:")
print("-" * 50)
for purity in [100, 50, 25]:
    training_mean = training_stats['sensor_combined'][purity]['mean']
    current = current_readings[purity]['sensor_combined']
    offset = current - training_mean
    print(f"  {purity}% purity:")
    print(f"    Training mean: {training_mean:.2f}")
    print(f"    Current:       {current:.2f}")
    print(f"    Offset:        {offset:+.2f}")

print("\n📊 MQ135 Ratio Analysis:")
print("-" * 50)
for purity in [100, 50, 25]:
    training_mean = training_stats['mq135_ratio'][purity]['mean']
    current = current_readings[purity]['mq135_ratio']
    offset = current - training_mean
    ratio = current / training_mean
    print(f"  {purity}% purity:")
    print(f"    Training mean: {training_mean:.2f}")
    print(f"    Current:       {current:.2f}")
    print(f"    Offset:        {offset:+.2f} (ratio: {ratio:.2f}x)")

# Calculate average offsets
sc_offsets = []
for purity in [100, 50, 25]:
    sc_offsets.append(current_readings[purity]['sensor_combined'] - 
                      training_stats['sensor_combined'][purity]['mean'])

print("\n" + "=" * 70)
print("RECOMMENDED CALIBRATION OFFSETS")
print("=" * 70)
print(f"\nAverage sensor_combined offset: {np.mean(sc_offsets):+.2f}")
print(f"Offset range: {min(sc_offsets):+.2f} to {max(sc_offsets):+.2f}")

print("\n⚠️  PROBLEM: Offsets are NOT consistent!")
print("   100% pure: offset = +4.65 (small drift)")
print("   50% pure:  offset = -7.84 (large negative drift!)")
print("   25% pure:  offset = -3.49 (moderate negative drift)")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("""
1. BEST OPTION: Recollect training data with current sensor state
   - This ensures model matches current sensor behavior
   - Include 0% pure (100% maida) samples this time!

2. ALTERNATIVE: Apply linear recalibration
   - But offsets are inconsistent, so this may not work well

3. QUICK FIX: Adjust thresholds based on current readings
   - 100% pure → sensor_combined ~42
   - 50% pure  → sensor_combined ~43.5
   - 25% pure  → sensor_combined ~52
   - 0% pure   → sensor_combined ~345

NEW SUGGESTED THRESHOLDS (based on current data):
   THRESHOLD_BEST_GOOD  = 42.5  (midpoint 100% and 50%)
   THRESHOLD_GOOD_POOR  = 47.5  (estimated 75% region)
   THRESHOLD_POOR_WORST = 100.0 (to separate 25% from 0%)

NEW INTERPOLATION POINTS:
   SC_POINTS = {42.0, 43.5, 52.0, 345.0}
   PURITY_POINTS = {100, 50, 25, 0}
""")

# What the readings SHOULD predict based on pattern
print("\n" + "=" * 70)
print("WHAT YOUR CURRENT READINGS SHOULD PREDICT")
print("=" * 70)

def predict_with_new_calibration(sensor_combined):
    """Using new interpolation based on current test data"""
    sc_pts = [42.0, 43.5, 52.0, 345.0]
    purity_pts = [100, 50, 25, 0]
    
    if sensor_combined <= sc_pts[0]:
        return 100.0
    if sensor_combined >= sc_pts[3]:
        return 0.0
    
    for i in range(len(sc_pts) - 1):
        if sc_pts[i] <= sensor_combined < sc_pts[i+1]:
            fraction = (sensor_combined - sc_pts[i]) / (sc_pts[i+1] - sc_pts[i])
            purity = purity_pts[i] + fraction * (purity_pts[i+1] - purity_pts[i])
            return purity
    return 50.0

# Test the new calibration
test_cases = [
    (42.0, "100% pure sample"),
    (43.5, "50% pure sample"),
    (52.0, "25% pure sample"),
    (345.0, "0% pure (100% maida)"),
]

for sc, desc in test_cases:
    predicted = predict_with_new_calibration(sc)
    print(f"  {desc}: sensor_combined={sc:.1f} → {predicted:.1f}% purity")
