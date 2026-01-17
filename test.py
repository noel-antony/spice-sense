"""
Spice Purity Inference - Make Predictions on New Samples
Load trained model and predict purity from sensor readings
"""

import pickle
import numpy as np
import json

def load_models():
    """Load all trained models"""
    print("Loading models...")
    
    # Load binary classifier
    with open('binary_classifier.pkl', 'rb') as f:
        binary_data = pickle.load(f)
    print("✓ Binary classifier loaded")
    
    # Load multi-class classifier
    with open('multiclass_classifier.pkl', 'rb') as f:
        multi_data = pickle.load(f)
    print("✓ Multi-class classifier loaded")
    
    # Load regression model
    with open('regression_model.pkl', 'rb') as f:
        regression_data = pickle.load(f)
    print("✓ Regression model loaded")
    
    # Load metadata
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    print("✓ Metadata loaded")
    
    return binary_data, multi_data, regression_data, metadata

def calculate_features(sensor_data):
    """
    Calculate the 3 required features from raw sensor data
    
    Input: Dictionary with sensor readings
    {
        'temp_C': 30.5,
        'humidity_%': 73.2,
        'gas_resistance_kOhm': 26.5,
        'mq135_ratio': 0.55,
        'mq3_voltage': 1.31,
        'mq3_ratio': 1.51  (optional)
    }
    
    Output: Array of 3 features [env_stability, sensor_combined, mq3_voltage]
    """
    
    # Feature 1: env_stability = temp_C * (humidity_% / 100)
    humidity_factor = sensor_data['humidity_%'] / 100.0
    env_stability = sensor_data['temp_C'] * humidity_factor
    
    # Feature 2: sensor_combined = weighted combination
    # Uses gas_resistance and ratios with specific weights
    sensor_combined = (
        sensor_data['gas_resistance_kOhm'] * 0.5 + 
        sensor_data['mq135_ratio'] * 10 +
        sensor_data.get('mq3_ratio', sensor_data['mq3_voltage']) * 10
    )
    
    # Feature 3: mq3_voltage (direct)
    mq3_voltage = sensor_data['mq3_voltage']
    
    return np.array([[env_stability, sensor_combined, mq3_voltage]])

def predict_all(sensor_data, binary_data, multi_data, regression_data):
    """
    Run all three prediction types
    
    Returns complete analysis of the sample
    """
    
    # Calculate features
    features = calculate_features(sensor_data)
    
    # === BINARY CLASSIFICATION ===
    features_scaled = binary_data['scaler'].transform(features)
    binary_pred = binary_data['model'].predict(features_scaled)[0]
    binary_proba = binary_data['model'].predict_proba(features_scaled)[0]
    binary_class = binary_data['label_encoder'].inverse_transform([binary_pred])[0]
    binary_confidence = binary_proba.max() * 100
    
    # === MULTI-CLASS CLASSIFICATION ===
    multi_pred = multi_data['model'].predict(features_scaled)[0]
    multi_proba = multi_data['model'].predict_proba(features_scaled)[0]
    multi_class = multi_data['label_encoder'].inverse_transform([multi_pred])[0]
    multi_confidence = multi_proba.max() * 100
    
    # === REGRESSION (EXACT PURITY %) ===
    features_scaled_reg = regression_data['scaler'].transform(features)
    purity_percentage = regression_data['model'].predict(features_scaled_reg)[0]
    
    # Map quality to descriptive text
    quality_map = {
        'best': '🟢 BEST - Pure/Premium Quality',
        'good': '🟡 GOOD - Slight contamination',
        'poor': '🟠 POOR - Moderate contamination',
        'worst': '🔴 WORST - Heavy contamination',
        'high_purity': '✅ HIGH PURITY (75-100%)',
        'low_purity': '⚠️ LOW PURITY (25-50%)'
    }
    
    return {
        'binary': {
            'class': binary_class,
            'description': quality_map.get(binary_class, binary_class),
            'confidence': binary_confidence,
            'probabilities': {
                binary_data['label_encoder'].classes_[0]: binary_proba[0] * 100,
                binary_data['label_encoder'].classes_[1]: binary_proba[1] * 100
            }
        },
        'multiclass': {
            'class': multi_class,
            'description': quality_map.get(multi_class, multi_class),
            'confidence': multi_confidence,
            'probabilities': {
                cls: prob * 100 
                for cls, prob in zip(multi_data['label_encoder'].classes_, multi_proba)
            }
        },
        'regression': {
            'predicted_purity': purity_percentage,
            'interpretation': f"{purity_percentage:.1f}% pure turmeric, {100-purity_percentage:.1f}% contamination"
        },
        'features_used': {
            'env_stability': features[0][0],
            'sensor_combined': features[0][1],
            'mq3_voltage': features[0][2]
        }
    }

def print_prediction_report(sample_name, sensor_data, results):
    """Print a formatted prediction report"""
    print("\n" + "="*70)
    print(f"ANALYSIS: {sample_name}")
    print("="*70)
    
    print("\n📊 SENSOR READINGS:")
    for key, value in sensor_data.items():
        print(f"   {key:25s} = {value}")
    
    print("\n🔬 CALCULATED FEATURES:")
    for key, value in results['features_used'].items():
        print(f"   {key:25s} = {value:.4f}")
    
    print("\n" + "-"*70)
    print("PREDICTION RESULTS")
    print("-"*70)
    
    print("\n1️⃣  BINARY CLASSIFICATION:")
    print(f"   Result: {results['binary']['description']}")
    print(f"   Confidence: {results['binary']['confidence']:.2f}%")
    print(f"   Probabilities:")
    for cls, prob in results['binary']['probabilities'].items():
        print(f"      {cls}: {prob:.2f}%")
    
    print("\n2️⃣  MULTI-CLASS QUALITY GRADE:")
    print(f"   Grade: {results['multiclass']['description']}")
    print(f"   Confidence: {results['multiclass']['confidence']:.2f}%")
    print(f"   Probabilities:")
    for cls, prob in sorted(results['multiclass']['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob/5)
        print(f"      {cls:10s}: {prob:5.2f}% {bar}")
    
    print("\n3️⃣  EXACT PURITY PREDICTION:")
    print(f"   Purity: {results['regression']['predicted_purity']:.1f}%")
    print(f"   {results['regression']['interpretation']}")
    
    print("\n" + "="*70)

def main():
    """Example usage with test samples"""
    print("="*70)
    print("SPICE PURITY PREDICTION SYSTEM")
    print("="*70)
    
    # Load all models
    binary_data, multi_data, regression_data, metadata = load_models()
    
    print("\n📋 MODEL INFORMATION:")
    print(f"   Features used: {', '.join(metadata['feature_names'])}")
    print(f"   Binary classes: {', '.join(metadata['binary_classes'])}")
    print(f"   Multi classes: {', '.join(metadata['multi_classes'])}")
    print(f"\n   Model Performance:")
    print(f"   - Binary accuracy: {metadata['model_performance']['binary_accuracy']:.4f}")
    print(f"   - Multi accuracy: {metadata['model_performance']['multi_accuracy']:.4f}")
    print(f"   - Regression RMSE: {metadata['model_performance']['regression_rmse']:.2f}%")
    print(f"   - Regression R²: {metadata['model_performance']['regression_r2']:.4f}")
    
    # Test samples (replace with real ESP32 data)
    test_samples = [
        {
            'name': 'Sample 1: Pure Turmeric (Expected: 100%)',
            'data': {
                'temp_C': 30.65,
                'humidity_%': 74.36,
                'gas_resistance_kOhm': 26.00,
                'mq135_ratio': 0.5489,
                'mq3_voltage': 1.309,
                'mq3_ratio': 1.5203
            }
        },
        {
            'name': 'Sample 2: Contaminated (Expected: 25%)',
            'data': {
                'temp_C': 32.06,
                'humidity_%': 72.74,
                'gas_resistance_kOhm': 33.01,
                'mq135_ratio': 1.5917,
                'mq3_voltage': 1.314,
                'mq3_ratio': 1.5115
            }
        },
        {
            'name': 'Sample 3: Medium Purity (Expected: ~50-75%)',
            'data': {
                'temp_C': 31.5,
                'humidity_%': 73.0,
                'gas_resistance_kOhm': 29.5,
                'mq135_ratio': 1.0,
                'mq3_voltage': 1.31,
                'mq3_ratio': 1.51
            }
        }
    ]
    
    # Make predictions
    for sample in test_samples:
        results = predict_all(
            sample['data'],
            binary_data,
            multi_data,
            regression_data
        )
        print_prediction_report(sample['name'], sample['data'], results)
    
    print("\n" + "="*70)
    print("✓ ALL PREDICTIONS COMPLETE")
    print("="*70)
    print("\n💡 TIP: Replace test_samples with real ESP32 sensor data")
    print("   for live predictions!\n")

if __name__ == "__main__":
    main()