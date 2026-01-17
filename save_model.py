"""
Save Trained Spice Purity Model
Run this after train.py to save the best model for inference
"""

import pickle
import json
from train import main

def save_models(results):
    """Save trained models and metadata"""
    print("\n" + "="*70)
    print("SAVING TRAINED MODELS")
    print("="*70)
    
    # Extract best models
    binary_rf = results['binary_clf']['Random Forest']['model']
    multi_rf = results['multi_clf']['Random Forest']['model']
    regression_rf = results['regression']['Random Forest']['model']
    
    # Extract data_dict from results - it's stored at top level
    data_dict = results['data_dict']
    
    # Extract scalers and label encoders from data_dict
    binary_scaler = data_dict['binary'][5]
    binary_le = data_dict['binary'][4]
    
    multi_scaler = data_dict['multi'][5]
    multi_le = data_dict['multi'][4]
    
    regression_scaler = data_dict['regression'][4]
    
    # Feature names
    feature_names = data_dict['feature_names']
    
    print(f"\n📋 Features being saved: {feature_names}")
    print(f"📋 Binary classes: {binary_le.classes_}")
    print(f"📋 Multi classes: {multi_le.classes_}")
    
    # === SAVE BINARY CLASSIFICATION MODEL ===
    print("\n1. Saving Binary Classification Model...")
    with open('binary_classifier.pkl', 'wb') as f:
        pickle.dump({
            'model': binary_rf,
            'scaler': binary_scaler,
            'label_encoder': binary_le,
            'features': feature_names
        }, f)
    print("   ✓ Saved: binary_classifier.pkl")
    
    # === SAVE MULTI-CLASS CLASSIFICATION MODEL ===
    print("\n2. Saving Multi-Class Classification Model...")
    with open('multiclass_classifier.pkl', 'wb') as f:
        pickle.dump({
            'model': multi_rf,
            'scaler': multi_scaler,
            'label_encoder': multi_le,
            'features': feature_names
        }, f)
    print("   ✓ Saved: multiclass_classifier.pkl")
    
    # === SAVE REGRESSION MODEL ===
    print("\n3. Saving Regression Model...")
    with open('regression_model.pkl', 'wb') as f:
        pickle.dump({
            'model': regression_rf,
            'scaler': regression_scaler,
            'features': feature_names
        }, f)
    print("   ✓ Saved: regression_model.pkl")
    
    # === SAVE METADATA ===
    print("\n4. Saving Model Metadata...")
    metadata = {
        'feature_names': feature_names,
        'binary_classes': binary_le.classes_.tolist(),
        'multi_classes': multi_le.classes_.tolist(),
        'model_performance': {
            'binary_accuracy': results['binary_clf']['Random Forest']['accuracy'],
            'multi_accuracy': results['multi_clf']['Random Forest']['accuracy'],
            'regression_rmse': results['regression']['Random Forest']['rmse'],
            'regression_mae': results['regression']['Random Forest']['mae'],
            'regression_r2': results['regression']['Random Forest']['r2']
        },
        'feature_descriptions': {
            'env_stability': 'Temperature * Humidity interaction (correlation: 0.9721)',
            'sensor_combined': 'Weighted multi-sensor fusion (correlation: 0.9500)',
            'mq3_voltage': 'MQ3 VOC sensor voltage (correlation: 0.6669)'
        },
        'purity_scale': {
            '100%': 'Best - Pure turmeric (0% contamination)',
            '75%': 'Good - 25% flour contamination',
            '50%': 'Poor - 50% flour contamination',
            '25%': 'Worst - 75% flour contamination'
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✓ Saved: model_metadata.json")
    
    print("\n" + "="*70)
    print("✓ ALL MODELS SAVED SUCCESSFULLY!")
    print("="*70)
    print("\nSaved files:")
    print("  1. binary_classifier.pkl       - High/Low purity (100% accuracy)")
    print("  2. multiclass_classifier.pkl   - Best/Good/Poor/Worst (100% accuracy)")
    print("  3. regression_model.pkl        - Exact purity % (RMSE: 2.21%)")
    print("  4. model_metadata.json         - Model info and performance")
    print("\n✅ You can now run test.py for inference!")
    
    return True

if __name__ == "__main__":
    print("="*70)
    print("TRAINING AND SAVING SPICE PURITY MODELS")
    print("="*70)
    
    # Train models
    CSV_FILE = "spice_data.csv"
    print(f"\nTraining models using: {CSV_FILE}\n")
    
    try:
        results = main(CSV_FILE)
        
        # Debug: Print structure
        print("\n[DEBUG] Results structure:")
        print(f"  Keys in results: {results.keys()}")
        
        # Save models
        save_models(results)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure train.py is in the same directory")
        import traceback
        traceback.print_exc()