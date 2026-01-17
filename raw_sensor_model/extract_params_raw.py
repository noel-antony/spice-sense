"""
Extract model parameters from raw sensor models for ESP32 deployment.
Outputs parameters that can be hardcoded into Arduino/ESP32 code.
"""

import pickle
import json
import numpy as np

def load_models():
    """Load all trained raw sensor models"""
    with open('binary_classifier_raw.pkl', 'rb') as f:
        binary_data = pickle.load(f)
    
    with open('multiclass_classifier_raw.pkl', 'rb') as f:
        multi_data = pickle.load(f)
    
    with open('regression_model_raw.pkl', 'rb') as f:
        reg_data = pickle.load(f)
    
    with open('model_metadata_raw.json', 'r') as f:
        metadata = json.load(f)
    
    return binary_data, multi_data, reg_data, metadata

def extract_scaling_params(metadata):
    """Extract feature scaling parameters"""
    print("="*70)
    print("FEATURE SCALING PARAMETERS")
    print("="*70)
    
    features = metadata['feature_names']
    means = metadata['scaling']['mean']
    stds = metadata['scaling']['std']
    
    print(f"\nFeatures: {features}")
    print(f"\n// Feature means (for standardization)")
    print(f"const float FEATURE_MEAN[3] = {{{means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f}}};")
    print(f"\n// Feature stds (for standardization)")
    print(f"const float FEATURE_STD[3] = {{{stds[0]:.4f}, {stds[1]:.4f}, {stds[2]:.4f}}};")
    
    return means, stds

def extract_rf_thresholds(multi_data, metadata):
    """Extract decision thresholds from Random Forest for multi-class"""
    print("\n" + "="*70)
    print("MULTI-CLASS CLASSIFICATION THRESHOLDS")
    print("="*70)
    
    # Since we have 4 classes (best, good, poor, worst), we need to find
    # feature-based thresholds. For ESP32, we'll use gas_resistance as primary
    # since it has highest correlation (0.9345)
    
    # Load training data to find class boundaries
    import pandas as pd
    df = pd.read_csv('../spice_data.csv')
    df['mq135_aqi'] = 100.0 / df['mq135_ratio']
    
    features = metadata['feature_names']
    
    print("\n--- Class Boundaries (Gas Resistance) ---")
    for level in [100, 75, 50, 25]:
        subset = df[df['contamination_level_%'] == level]
        gas_mean = subset['gas_resistance_kOhm'].mean()
        gas_std = subset['gas_resistance_kOhm'].std()
        print(f"   Purity {level}%: gas_resistance = {gas_mean:.2f} ± {gas_std:.2f}")
    
    # Calculate thresholds as midpoints between class means
    class_means = {}
    for level in [100, 75, 50, 25]:
        class_means[level] = df[df['contamination_level_%'] == level]['gas_resistance_kOhm'].mean()
    
    # Thresholds
    thresh_best_good = (class_means[100] + class_means[75]) / 2
    thresh_good_poor = (class_means[75] + class_means[50]) / 2
    thresh_poor_worst = (class_means[50] + class_means[25]) / 2
    
    print(f"\n// Gas resistance thresholds for classification")
    print(f"const float GAS_BEST_GOOD = {thresh_best_good:.2f};   // 100% vs 75%")
    print(f"const float GAS_GOOD_POOR = {thresh_good_poor:.2f};   // 75% vs 50%")
    print(f"const float GAS_POOR_WORST = {thresh_poor_worst:.2f}; // 50% vs 25%")
    
    # Also for binary: high (>=75%) vs low (<75%)
    high_mean = df[df['contamination_level_%'] >= 75]['gas_resistance_kOhm'].mean()
    low_mean = df[df['contamination_level_%'] < 75]['gas_resistance_kOhm'].mean()
    thresh_binary = (high_mean + low_mean) / 2
    
    print(f"\n// Binary threshold (high purity vs low purity)")
    print(f"const float GAS_HIGH_LOW = {thresh_binary:.2f};")
    
    return {
        'best_good': thresh_best_good,
        'good_poor': thresh_good_poor,
        'poor_worst': thresh_poor_worst,
        'high_low': thresh_binary
    }

def extract_purity_lookup(metadata):
    """Create lookup table for purity estimation from gas resistance"""
    print("\n" + "="*70)
    print("PURITY LOOKUP TABLE (INTERPOLATION)")
    print("="*70)
    
    import pandas as pd
    df = pd.read_csv('../spice_data.csv')
    
    # Get mean gas resistance for each purity level
    gas_points = []
    purity_points = []
    
    for level in [100, 75, 50, 25]:
        gas_mean = df[df['contamination_level_%'] == level]['gas_resistance_kOhm'].mean()
        gas_points.append(gas_mean)
        purity_points.append(level)
    
    print("\n// Lookup table: Gas Resistance -> Purity %")
    print(f"const int NUM_POINTS = 4;")
    print(f"const float GAS_POINTS[4] = {{{gas_points[0]:.2f}, {gas_points[1]:.2f}, {gas_points[2]:.2f}, {gas_points[3]:.2f}}};")
    print(f"const float PURITY_POINTS[4] = {{{purity_points[0]}, {purity_points[1]}, {purity_points[2]}, {purity_points[3]}}};")
    
    # Note: Gas resistance DECREASES with lower purity
    print("\n// Note: Lower gas resistance = Lower purity")
    print("// Interpolation: gas < 27.86 -> 100%, gas > 39.06 -> 25%")
    
    return gas_points, purity_points

def extract_regression_coefficients(reg_data, metadata):
    """Extract regression model parameters if linear, or create lookup otherwise"""
    print("\n" + "="*70)
    print("REGRESSION MODEL PARAMETERS")
    print("="*70)
    
    model = reg_data['model']
    model_type = type(model).__name__
    print(f"\nModel type: {model_type}")
    
    # For Random Forest, we can't extract simple coefficients
    # Instead, provide a linear approximation based on gas resistance
    
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    
    df = pd.read_csv('../spice_data.csv')
    df['mq135_aqi'] = 100.0 / df['mq135_ratio']
    
    # Fit a simple linear model using gas_resistance (highest correlation)
    X = df['gas_resistance_kOhm'].values.reshape(-1, 1)
    y = df['contamination_level_%'].values
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    print(f"\n// Simple linear regression on gas_resistance")
    print(f"// purity = intercept + coef * gas_resistance")
    print(f"const float REG_INTERCEPT = {lr.intercept_:.4f};")
    print(f"const float REG_COEF_GAS = {lr.coef_[0]:.4f};")
    
    # Also fit multi-feature linear regression
    X_multi = df[['mq135_aqi', 'mq3_voltage', 'gas_resistance_kOhm']].values
    lr_multi = LinearRegression()
    lr_multi.fit(X_multi, y)
    
    print(f"\n// Multi-feature linear regression (fallback)")
    print(f"// purity = intercept + c1*aqi + c2*mq3_v + c3*gas_r")
    print(f"const float REG_MULTI_INTERCEPT = {lr_multi.intercept_:.4f};")
    print(f"const float REG_MULTI_COEF[3] = {{{lr_multi.coef_[0]:.4f}, {lr_multi.coef_[1]:.4f}, {lr_multi.coef_[2]:.4f}}};")
    
    from sklearn.metrics import r2_score
    r2_simple = r2_score(y, lr.predict(X))
    r2_multi = r2_score(y, lr_multi.predict(X_multi))
    
    print(f"\n// Linear regression R² scores:")
    print(f"// Simple (gas only): {r2_simple:.4f}")
    print(f"// Multi-feature: {r2_multi:.4f}")
    
    return {
        'simple': {'intercept': lr.intercept_, 'coef': lr.coef_[0]},
        'multi': {'intercept': lr_multi.intercept_, 'coef': lr_multi.coef_.tolist()}
    }

def generate_esp32_params_json(metadata, thresholds, gas_points, purity_points, regression):
    """Generate JSON file with all ESP32 parameters"""
    print("\n" + "="*70)
    print("GENERATING ESP32 PARAMETERS FILE")
    print("="*70)
    
    params = {
        'model_type': 'raw_sensor_features',
        'features': metadata['feature_names'],
        'scaling': {
            'mean': metadata['scaling']['mean'],
            'std': metadata['scaling']['std']
        },
        'classification_thresholds': {
            'gas_based': thresholds,
            'note': 'Lower gas_resistance = Lower purity'
        },
        'purity_lookup': {
            'gas_points': gas_points,
            'purity_points': purity_points
        },
        'regression': regression,
        'class_labels': {
            'binary': ['high_purity', 'low_purity'],
            'multi': ['best', 'good', 'poor', 'worst']
        },
        'aqi_calculation': {
            'formula': 'AQI = 100.0 / mq135_ratio',
            'note': 'Higher AQI = Higher purity (cleaner air from pure spice)'
        }
    }
    
    with open('esp32_parameters_raw.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    print("✓ Saved esp32_parameters_raw.json")
    
    return params

def print_arduino_code():
    """Print ready-to-use Arduino code snippet"""
    print("\n" + "="*70)
    print("ESP32 ARDUINO CODE SNIPPET")
    print("="*70)
    
    code = '''
// ============================================================
// RAW SENSOR MODEL PARAMETERS - Copy to ESP32 Code
// ============================================================

// Features: [mq135_aqi, mq3_voltage, gas_resistance_kOhm]
const int NUM_FEATURES = 3;

// Feature scaling (StandardScaler parameters)
const float FEATURE_MEAN[3] = {66.6355, 1.2879, 33.7827};
const float FEATURE_STD[3] = {17.6961, 0.0374, 4.4636};

// Gas resistance thresholds for classification
const float GAS_BEST_GOOD = 30.33;   // < this = 100% pure
const float GAS_GOOD_POOR = 34.11;   // < this = 75%+ 
const float GAS_POOR_WORST = 37.23;  // < this = 50%+, else 25%

// Binary threshold
const float GAS_HIGH_LOW = 34.11;    // >= 75% vs < 75%

// Purity lookup table (gas resistance -> purity %)
const int NUM_POINTS = 4;
const float GAS_POINTS[4] = {27.86, 32.81, 35.41, 39.06};
const float PURITY_POINTS[4] = {100.0, 75.0, 50.0, 25.0};

// Linear regression: purity = intercept + coef * gas_resistance
const float REG_INTERCEPT = 185.91;
const float REG_COEF_GAS = -4.00;

// ============================================================
// HELPER FUNCTIONS
// ============================================================

float calculateAQI(float mq135_ratio) {
    if (mq135_ratio <= 0) return 500.0;
    return 100.0 / mq135_ratio;
}

float interpolatePurity(float gas_resistance) {
    // Gas resistance decreases with lower purity
    if (gas_resistance <= GAS_POINTS[0]) return PURITY_POINTS[0];  // 100%
    if (gas_resistance >= GAS_POINTS[3]) return PURITY_POINTS[3];  // 25%
    
    for (int i = 0; i < NUM_POINTS - 1; i++) {
        if (gas_resistance >= GAS_POINTS[i] && gas_resistance < GAS_POINTS[i+1]) {
            float ratio = (gas_resistance - GAS_POINTS[i]) / 
                         (GAS_POINTS[i+1] - GAS_POINTS[i]);
            return PURITY_POINTS[i] + ratio * (PURITY_POINTS[i+1] - PURITY_POINTS[i]);
        }
    }
    return 50.0;  // Default
}

String getQualityCategory(float gas_resistance) {
    if (gas_resistance < GAS_BEST_GOOD) return "BEST";
    if (gas_resistance < GAS_GOOD_POOR) return "GOOD";
    if (gas_resistance < GAS_POOR_WORST) return "POOR";
    return "WORST";
}

bool isHighPurity(float gas_resistance) {
    return gas_resistance < GAS_HIGH_LOW;
}
'''
    print(code)

def main():
    """Main extraction pipeline"""
    print("\n" + "="*70)
    print("EXTRACTING RAW SENSOR MODEL PARAMETERS FOR ESP32")
    print("="*70)
    
    # Load models
    binary_data, multi_data, reg_data, metadata = load_models()
    
    # Extract parameters
    means, stds = extract_scaling_params(metadata)
    thresholds = extract_rf_thresholds(multi_data, metadata)
    gas_points, purity_points = extract_purity_lookup(metadata)
    regression = extract_regression_coefficients(reg_data, metadata)
    
    # Generate ESP32 parameters JSON
    params = generate_esp32_params_json(metadata, thresholds, gas_points, 
                                        purity_points, regression)
    
    # Print Arduino code
    print_arduino_code()
    
    print("\n" + "="*70)
    print("✓ PARAMETER EXTRACTION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
