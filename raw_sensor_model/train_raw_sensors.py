"""
Spice Purity Testing - Raw Sensor Model
Uses only 3 direct sensor readings:
1. MQ135 AQI (calculated from mq135_ratio)
2. MQ3 Voltage
3. BME688 Gas Resistance

This is an alternative to the engineered feature model.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, mean_squared_error, r2_score,
                             mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Path to parent directory CSV
CSV_FILE = "../spice_data.csv"

# ============================================================================
# AQI CALCULATION FROM MQ135
# ============================================================================

def calculate_aqi_from_mq135(mq135_ratio):
    """
    Calculate Air Quality Index from MQ135 sensor ratio.
    
    MQ135 detects: NH3, NOx, Alcohol, Benzene, Smoke, CO2
    
    This is a simplified AQI based on Rs/R0 ratio:
    - Lower ratio = Higher gas concentration = Higher AQI (worse air)
    - Higher ratio = Lower gas concentration = Lower AQI (better air)
    
    AQI Scale (simplified):
    - 0-50: Good
    - 51-100: Moderate  
    - 101-150: Unhealthy for sensitive groups
    - 151-200: Unhealthy
    - 201-300: Very unhealthy
    - 301-500: Hazardous
    """
    # Convert ratio to approximate PPM (simplified model)
    # In clean air, Rs/R0 ≈ 3.6 for MQ135
    # Lower ratio = more gas
    
    # Simplified conversion: AQI ≈ (1/ratio) * scaling_factor
    # Adjust scaling to get reasonable AQI range
    
    if mq135_ratio <= 0:
        return 500  # Max AQI for invalid reading
    
    # Inverse relationship: lower ratio = higher AQI
    # Scale to typical AQI range (0-500)
    aqi = (1.0 / mq135_ratio) * 100
    
    # Clamp to valid range
    aqi = max(0, min(500, aqi))
    
    return aqi

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load CSV and prepare the 3 raw sensor features"""
    print("="*70)
    print("LOADING AND PREPARING DATA")
    print("="*70)
    
    df = pd.read_csv(CSV_FILE)
    print(f"\n✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Create AQI from MQ135 ratio
    df['mq135_aqi'] = df['mq135_ratio'].apply(calculate_aqi_from_mq135)
    
    print("\n--- Raw Sensor Features ---")
    print("1. mq135_aqi (calculated from mq135_ratio)")
    print("2. mq3_voltage")
    print("3. gas_resistance_kOhm")
    
    # Check contamination level distribution
    print("\n--- Contamination Level Distribution ---")
    print(df['contamination_level_%'].value_counts().sort_index())
    
    return df

# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_data(df):
    """Clean data: remove missing values, duplicates, and outliers"""
    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Remove missing values
    df_clean = df_clean.dropna()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Remove outliers using IQR method (3*IQR)
    features = ['gas_resistance_kOhm', 'mq3_voltage', 'mq135_aqi']
    for col in features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                           (df_clean[col] <= upper_bound)]
    
    print(f"\n✓ Cleaned: {initial_rows} → {len(df_clean)} rows")
    print(f"  Removed: {initial_rows - len(df_clean)} rows ({(initial_rows - len(df_clean))/initial_rows*100:.1f}%)")
    
    return df_clean

# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

def analyze_features(df):
    """Analyze correlation of raw sensor features with target"""
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)
    
    features = ['mq135_aqi', 'mq3_voltage', 'gas_resistance_kOhm']
    target = 'contamination_level_%'
    
    print("\n📊 Feature Correlations with Target:")
    correlations = {}
    for feat in features:
        corr = abs(df[feat].corr(df[target]))
        correlations[feat] = corr
        print(f"   {feat}: {corr:.4f}")
    
    print("\n📊 Feature Statistics by Purity Level:")
    for feat in features:
        print(f"\n{feat}:")
        stats = df.groupby(target)[feat].agg(['mean', 'std', 'min', 'max'])
        print(stats)
    
    return features, correlations

# ============================================================================
# ML DATA PREPARATION
# ============================================================================

def prepare_ml_data(df, features):
    """Prepare train/test splits for all tasks"""
    print("\n" + "="*70)
    print("PREPARING ML DATASETS")
    print("="*70)
    
    X = df[features].values
    
    # Binary classification: High purity (>=75%) vs Low purity (<75%)
    df['binary_class'] = df['contamination_level_%'].apply(
        lambda x: 'high_purity' if x >= 75 else 'low_purity'
    )
    
    # Multi-class classification
    def categorize(level):
        if level == 100:
            return 'best'
        elif level >= 75:
            return 'good'
        elif level >= 50:
            return 'poor'
        else:
            return 'worst'
    
    df['multi_class'] = df['contamination_level_%'].apply(categorize)
    
    le_binary = LabelEncoder()
    le_multi = LabelEncoder()
    y_binary = le_binary.fit_transform(df['binary_class'])
    y_multi = le_multi.fit_transform(df['multi_class'])
    y_regression = df['contamination_level_%'].values
    
    # Split data
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_clf = StandardScaler()
    scaler_reg = StandardScaler()
    
    X_train_bin_scaled = scaler_clf.fit_transform(X_train_bin)
    X_test_bin_scaled = scaler_clf.transform(X_test_bin)
    X_train_multi_scaled = scaler_clf.transform(X_train_multi)
    X_test_multi_scaled = scaler_clf.transform(X_test_multi)
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    print(f"\n✓ Binary: {X_train_bin_scaled.shape[0]} train, {X_test_bin_scaled.shape[0]} test")
    print(f"✓ Multi-Class: {X_train_multi_scaled.shape[0]} train, {X_test_multi_scaled.shape[0]} test")
    print(f"✓ Regression: {X_train_reg_scaled.shape[0]} train, {X_test_reg_scaled.shape[0]} test")
    
    return {
        'binary': (X_train_bin_scaled, X_test_bin_scaled, y_train_bin, y_test_bin, le_binary, scaler_clf),
        'multi': (X_train_multi_scaled, X_test_multi_scaled, y_train_multi, y_test_multi, le_multi, scaler_clf),
        'regression': (X_train_reg_scaled, X_test_reg_scaled, y_train_reg, y_test_reg, scaler_reg),
        'feature_names': features
    }

# ============================================================================
# TRAIN CLASSIFICATION MODELS
# ============================================================================

def train_classification_models(data_dict, task='binary'):
    """Train multiple classification models"""
    print("\n" + "="*70)
    print(f"TRAINING {task.upper()} CLASSIFICATION MODELS")
    print("="*70)
    
    if task == 'binary':
        X_train, X_test, y_train, y_test, le, scaler = data_dict['binary']
    else:
        X_train, X_test, y_train, y_test, le, scaler = data_dict['multi']
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        print(f"✓ Test Accuracy: {accuracy:.4f}")
        print(f"✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'predictions': y_pred
        }
    
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\n{'='*70}")
    print(f"🏆 BEST {task.upper()} MODEL: {best_model_name} ({results[best_model_name]['accuracy']:.4f})")
    print(f"{'='*70}")
    
    return results

# ============================================================================
# TRAIN REGRESSION MODELS
# ============================================================================

def train_regression_models(data_dict):
    """Train multiple regression models"""
    print("\n" + "="*70)
    print("TRAINING REGRESSION MODELS")
    print("="*70)
    
    X_train, X_test, y_train, y_test, scaler = data_dict['regression']
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'SVR': SVR(kernel='rbf'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✓ RMSE: {rmse:.2f}%")
        print(f"✓ MAE: {mae:.2f}%")
        print(f"✓ R² Score: {r2:.4f}")
        
        results[name] = {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    print(f"\n{'='*70}")
    print(f"🏆 BEST REGRESSION MODEL: {best_model_name} (RMSE: {results[best_model_name]['rmse']:.2f}%)")
    print(f"{'='*70}")
    
    return results

# ============================================================================
# SAVE MODELS
# ============================================================================

def save_models(data_dict, binary_results, multi_results, reg_results):
    """Save trained models to pickle files"""
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    # Get best models
    best_binary = max(binary_results, key=lambda x: binary_results[x]['accuracy'])
    best_multi = max(multi_results, key=lambda x: multi_results[x]['accuracy'])
    best_reg = min(reg_results, key=lambda x: reg_results[x]['rmse'])
    
    # Get scalers and encoders
    _, _, _, _, le_binary, scaler_clf = data_dict['binary']
    _, _, _, _, le_multi, _ = data_dict['multi']
    _, _, _, _, scaler_reg = data_dict['regression']
    
    # Save binary classifier
    binary_data = {
        'model': binary_results[best_binary]['model'],
        'scaler': scaler_clf,
        'label_encoder': le_binary,
        'accuracy': binary_results[best_binary]['accuracy']
    }
    with open('binary_classifier_raw.pkl', 'wb') as f:
        pickle.dump(binary_data, f)
    print("✓ Saved binary_classifier_raw.pkl")
    
    # Save multi-class classifier
    multi_data = {
        'model': multi_results[best_multi]['model'],
        'scaler': scaler_clf,
        'label_encoder': le_multi,
        'accuracy': multi_results[best_multi]['accuracy']
    }
    with open('multiclass_classifier_raw.pkl', 'wb') as f:
        pickle.dump(multi_data, f)
    print("✓ Saved multiclass_classifier_raw.pkl")
    
    # Save regression model
    reg_data = {
        'model': reg_results[best_reg]['model'],
        'scaler': scaler_reg,
        'rmse': reg_results[best_reg]['rmse'],
        'r2': reg_results[best_reg]['r2']
    }
    with open('regression_model_raw.pkl', 'wb') as f:
        pickle.dump(reg_data, f)
    print("✓ Saved regression_model_raw.pkl")
    
    # Save metadata
    metadata = {
        'feature_names': data_dict['feature_names'],
        'feature_descriptions': {
            'mq135_aqi': 'Air Quality Index calculated from MQ135 ratio (100/ratio)',
            'mq3_voltage': 'Direct MQ3 sensor voltage (0-3.3V)',
            'gas_resistance_kOhm': 'BME688 gas resistance in kOhm'
        },
        'scaling': {
            'mean': scaler_clf.mean_.tolist(),
            'std': scaler_clf.scale_.tolist()
        },
        'binary_classes': le_binary.classes_.tolist(),
        'multi_classes': le_multi.classes_.tolist(),
        'model_performance': {
            'binary_accuracy': binary_results[best_binary]['accuracy'],
            'multi_accuracy': multi_results[best_multi]['accuracy'],
            'regression_rmse': reg_results[best_reg]['rmse'],
            'regression_r2': reg_results[best_reg]['r2']
        },
        'model_type': 'raw_sensor_features'
    }
    with open('model_metadata_raw.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved model_metadata_raw.json")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline using raw sensor values"""
    print("\n" + "="*70)
    print("SPICE PURITY ML PIPELINE - RAW SENSOR FEATURES")
    print("Features: MQ135 AQI, MQ3 Voltage, Gas Resistance")
    print("="*70)
    
    # Load and prepare data
    df = load_and_prepare_data()
    df_clean = clean_data(df)
    
    # Analyze features
    features, correlations = analyze_features(df_clean)
    
    # Prepare ML data
    data_dict = prepare_ml_data(df_clean, features)
    
    # Train models
    binary_results = train_classification_models(data_dict, task='binary')
    multi_results = train_classification_models(data_dict, task='multi')
    regression_results = train_regression_models(data_dict)
    
    # Save models
    save_models(data_dict, binary_results, multi_results, regression_results)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    
    # Summary comparison
    print("\n📊 RAW SENSOR MODEL SUMMARY:")
    print(f"   Features: {features}")
    print(f"   Binary Accuracy: {max(binary_results[m]['accuracy'] for m in binary_results):.4f}")
    print(f"   Multi Accuracy: {max(multi_results[m]['accuracy'] for m in multi_results):.4f}")
    print(f"   Regression RMSE: {min(regression_results[m]['rmse'] for m in regression_results):.2f}%")
    
    return {
        'data_dict': data_dict,
        'binary': binary_results,
        'multi': multi_results,
        'regression': regression_results,
        'features': features,
        'correlations': correlations
    }

if __name__ == "__main__":
    results = main()
