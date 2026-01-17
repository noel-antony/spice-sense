"""
Extract ACCURATE model parameters using the actual training data
This will give us real regression coefficients for ESP32
"""

import pickle
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# File paths
CSV_FILE = "E:/spicesniff csv data/spice_data.csv"
BINARY_MODEL = "E:/spicesniff csv data/binary_classifier.pkl"
MULTI_MODEL = "E:/spicesniff csv data/multiclass_classifier.pkl"
REG_MODEL = "E:/spicesniff csv data/regression_model.pkl"
METADATA = "E:/spicesniff csv data/model_metadata.json"

def load_and_prepare_data():
    """Load data and prepare features exactly as in training"""
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)
    
    # Engineer features (same as train.py)
    df['mq_voc_ratio'] = df['mq3_voltage'] / (df['mq135_voltage'] + 0.001)
    df['gas_temp_normalized'] = df['gas_resistance_kOhm'] / df['temp_C']
    df['humidity_factor'] = df['humidity_%'] / 100.0
    df['sensor_combined'] = (df['gas_resistance_kOhm'] * 0.5 + 
                             df['mq135_ratio'] * 10 + 
                             df['mq3_ratio'] * 10)
    df['mq_raw_ratio'] = df['mq135_raw'] / (df['mq3_raw'] + 1)
    df['env_stability'] = df['temp_C'] * df['humidity_factor']
    
    print(f"Dataset shape: {df.shape}")
    return df

def extract_decision_rules(model, feature_names, X, y):
    """Extract actual decision rules from Random Forest"""
    print("\n" + "="*70)
    print("ANALYZING DECISION RULES")
    print("="*70)
    
    # Get predictions for different contamination levels
    contamination_levels = [100, 75, 50, 25]
    
    print("\nAnalyzing sensor_combined values for each class:")
    for level in contamination_levels:
        mask = y == level
        if mask.sum() > 0:
            sensor_combined_values = X[mask, 1]  # sensor_combined is feature 1
            print(f"\n{level}% purity:")
            print(f"  sensor_combined range: {sensor_combined_values.min():.2f} - {sensor_combined_values.max():.2f}")
            print(f"  sensor_combined mean:  {sensor_combined_values.mean():.2f}")
            print(f"  sensor_combined median: {np.median(sensor_combined_values):.2f}")
    
    # Find optimal thresholds
    print("\n" + "-"*70)
    print("OPTIMAL THRESHOLDS")
    print("-"*70)
    
    # Binary: HIGH (>=75%) vs LOW (<75%)
    binary_threshold = X[(y == 75) | (y == 100), 1].mean()
    print(f"\nBinary (HIGH/LOW): sensor_combined = {binary_threshold:.2f}")
    
    # Multi-class boundaries
    threshold_best_good = (X[y == 100, 1].mean() + X[y == 75, 1].mean()) / 2
    threshold_good_poor = (X[y == 75, 1].mean() + X[y == 50, 1].mean()) / 2
    threshold_poor_worst = (X[y == 50, 1].mean() + X[y == 25, 1].mean()) / 2
    
    print(f"BEST/GOOD boundary:  sensor_combined = {threshold_best_good:.2f}")
    print(f"GOOD/POOR boundary:  sensor_combined = {threshold_good_poor:.2f}")
    print(f"POOR/WORST boundary: sensor_combined = {threshold_poor_worst:.2f}")
    
    return {
        'binary': binary_threshold,
        'best_good': threshold_best_good,
        'good_poor': threshold_good_poor,
        'poor_worst': threshold_poor_worst
    }

def extract_regression_coefficients(model, feature_names, X, y):
    """Extract regression parameters"""
    print("\n" + "="*70)
    print("REGRESSION ANALYSIS")
    print("="*70)
    
    # Get RF predictions
    y_pred_rf = model.predict(X)
    
    print("\nRandom Forest Performance:")
    print(f"  MAE: {mean_absolute_error(y, y_pred_rf):.2f}%")
    print(f"  R²:  {r2_score(y, y_pred_rf):.4f}")
    
    # Fit linear approximation
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    
    print("\nLinear Approximation:")
    print(f"  Intercept: {linear_model.intercept_:.4f}")
    for name, coef in zip(feature_names, linear_model.coef_):
        print(f"  {name}: {coef:.4f}")
    
    print(f"\nLinear Model Performance:")
    print(f"  MAE: {mean_absolute_error(y, y_pred_linear):.2f}%")
    print(f"  R²:  {r2_score(y, y_pred_linear):.4f}")
    
    # Test on extreme points
    print("\nTesting on extreme values:")
    for level in [100, 75, 50, 25]:
        mask = y == level
        if mask.sum() > 0:
            sample = X[mask][0:1]
            rf_pred = model.predict(sample)[0]
            lin_pred = linear_model.predict(sample)[0]
            print(f"  {level}% actual -> RF: {rf_pred:.1f}%, Linear: {lin_pred:.1f}%")
    
    return {
        'intercept': float(linear_model.intercept_),
        'coefficients': [float(c) for c in linear_model.coef_],
        'r2': float(r2_score(y, y_pred_linear)),
        'mae': float(mean_absolute_error(y, y_pred_linear))
    }

def extract_lookup_table(X, y):
    """Create lookup table based on sensor_combined ranges for each purity level"""
    print("\n" + "="*70)
    print("LOOKUP TABLE FOR REGRESSION (sensor_combined based)")
    print("="*70)
    
    lookup = {}
    for level in [100, 75, 50, 25]:
        mask = y == level
        if mask.sum() > 0:
            sc_values = X[mask, 1]  # sensor_combined is feature 1
            lookup[level] = {
                'min': float(sc_values.min()),
                'max': float(sc_values.max()),
                'mean': float(sc_values.mean()),
                'median': float(np.median(sc_values))
            }
            print(f"\n{level}% purity:")
            print(f"  sensor_combined: {sc_values.min():.2f} - {sc_values.max():.2f} (mean: {sc_values.mean():.2f})")
    
    # Create interpolation points (use means)
    # sensor_combined -> purity mapping
    sc_points = [lookup[100]['mean'], lookup[75]['mean'], lookup[50]['mean'], lookup[25]['mean']]
    purity_points = [100, 75, 50, 25]
    
    print("\n" + "-"*70)
    print("INTERPOLATION POINTS (for ESP32)")
    print("-"*70)
    print(f"\nconst float SC_POINTS[4] = {{{', '.join([f'{x:.2f}' for x in sc_points])}}};")
    print(f"const float PURITY_POINTS[4] = {{{', '.join([f'{x:.0f}' for x in purity_points])}}};")
    
    return {
        'lookup': lookup,
        'interpolation': {
            'sensor_combined_points': sc_points,
            'purity_points': purity_points
        }
    }

def main():
    # Load data
    df = load_and_prepare_data()
    
    # Load metadata
    with open(METADATA, 'r') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']  # Fixed: was 'features'
    
    print(f"\nUsing features: {feature_names}")
    
    # Prepare ML features
    X = df[feature_names].values
    y = df['contamination_level_%'].values
    
    # Load models
    print("\nLoading models...")
    with open(BINARY_MODEL, 'rb') as f:
        binary_data = pickle.load(f)
        scaler = binary_data['scaler']
    
    with open(REG_MODEL, 'rb') as f:
        reg_data = pickle.load(f)
        reg_model = reg_data['model']
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Extract thresholds
    thresholds = extract_decision_rules(reg_model, feature_names, X, y)
    
    # Extract regression coefficients
    reg_params = extract_regression_coefficients(reg_model, feature_names, X_scaled, y)
    
    # Extract lookup table for better regression
    lookup_params = extract_lookup_table(X, y)
    
    # Prepare output
    output = {
        'feature_names': feature_names,
        'scaling': {
            'mean': [float(x) for x in scaler.mean_],
            'std': [float(x) for x in scaler.scale_]
        },
        'thresholds': {
            'binary_high_low': float(thresholds['binary']),
            'best_good': float(thresholds['best_good']),
            'good_poor': float(thresholds['good_poor']),
            'poor_worst': float(thresholds['poor_worst'])
        },
        'regression': reg_params,
        'lookup_table': lookup_params
    }
    
    # Save
    output_file = "E:/spicesniff csv data/esp32_parameters.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print("ESP32 CODE PARAMETERS")
    print("="*70)
    
    print("\n// Feature scaling")
    print(f"const float FEATURE_MEAN[3] = {{{', '.join([f'{x:.2f}' for x in scaler.mean_])}}};")
    print(f"const float FEATURE_STD[3] = {{{', '.join([f'{x:.2f}' for x in scaler.scale_])}}};")
    
    print("\n// Classification thresholds (UNSCALED sensor_combined)")
    print(f"const float THRESHOLD_HIGH_LOW = {thresholds['binary']:.2f};")
    print(f"const float THRESHOLD_BEST_GOOD = {thresholds['best_good']:.2f};")
    print(f"const float THRESHOLD_GOOD_POOR = {thresholds['good_poor']:.2f};")
    print(f"const float THRESHOLD_POOR_WORST = {thresholds['poor_worst']:.2f};")
    
    print("\n// Regression via INTERPOLATION (better than linear!)")
    sc_pts = lookup_params['interpolation']['sensor_combined_points']
    pu_pts = lookup_params['interpolation']['purity_points']
    print(f"const float SC_POINTS[4] = {{{', '.join([f'{x:.2f}' for x in sc_pts])}}};")
    print(f"const float PURITY_POINTS[4] = {{{', '.join([f'{int(x)}' for x in pu_pts])}}};")
    
    print("\n// Linear regression coefficients (backup, less accurate)")
    print(f"const float REG_INTERCEPT = {reg_params['intercept']:.2f};")
    coefs = ', '.join([f'{c:.2f}' for c in reg_params['coefficients']])
    print(f"const float REG_COEF[3] = {{{coefs}}};")
    
    print(f"\n✅ Complete parameters saved to: {output_file}")
    print(f"\n📊 Linear regression quality: R²={reg_params['r2']:.4f}, MAE={reg_params['mae']:.2f}%")
    print("⚠️  Recommendation: Use interpolation instead of linear regression for better accuracy!")

if __name__ == "__main__":
    main()