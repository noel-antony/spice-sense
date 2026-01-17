"""
Spice Purity Testing - Optimized ML Pipeline
Incorporates best practices for data preprocessing and feature selection

IMPROVEMENTS:
1. Duplicate removal
2. Feature redundancy elimination
3. Correlation-based feature selection
4. More robust outlier detection
5. Better feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

# ============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================================================

def load_and_explore_data(csv_file):
    """Load CSV and perform initial exploration"""
    print("="*70)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("="*70)
    
    df = pd.read_csv(csv_file)
    print(f"\n✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found ✓")
    
    print("\n--- Contamination Level Distribution ---")
    print(df['contamination_level_%'].value_counts().sort_index())
    
    return df

# ============================================================================
# PART 2: ENHANCED DATA CLEANING
# ============================================================================

def clean_data(df):
    """Clean data: remove missing values, duplicates, and outliers"""
    print("\n" + "="*70)
    print("STEP 2: ENHANCED DATA CLEANING")
    print("="*70)
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # 1. Remove rows with missing values
    before_na = len(df_clean)
    df_clean = df_clean.dropna()
    removed_na = before_na - len(df_clean)
    print(f"\n✓ Removed {removed_na} rows with missing values")
    
    # 2. Remove duplicate rows (FRIEND'S ADVICE)
    before_dup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_dup = before_dup - len(df_clean)
    print(f"✓ Removed {removed_dup} duplicate rows")
    
    # 3. Remove exact duplicate sensor readings (potential sensor freeze)
    sensor_cols = ['gas_resistance_kOhm', 'mq135_raw', 'mq3_raw']
    before_sensor_dup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=sensor_cols, keep='first')
    removed_sensor_dup = before_sensor_dup - len(df_clean)
    print(f"✓ Removed {removed_sensor_dup} rows with identical sensor readings")
    
    # 4. Remove outliers using IQR method (FRIEND'S ADVICE)
    sensor_cols_full = ['gas_resistance_kOhm', 'mq135_raw', 'mq135_voltage', 
                        'mq135_ratio', 'mq3_raw', 'mq3_voltage', 'mq3_ratio']
    
    before_outlier = len(df_clean)
    for col in sensor_cols_full:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                               (df_clean[col] <= upper_bound)]
    
    removed_outliers = before_outlier - len(df_clean)
    print(f"✓ Removed {removed_outliers} outliers (IQR method, 3*IQR)")
    
    print(f"\n📊 Cleaning Summary:")
    print(f"   Initial rows: {initial_rows}")
    print(f"   Final rows: {len(df_clean)}")
    print(f"   Total removed: {initial_rows - len(df_clean)} ({(initial_rows - len(df_clean))/initial_rows*100:.1f}%)")
    
    return df_clean

# ============================================================================
# PART 3: SMART FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create meaningful derived features"""
    print("\n" + "="*70)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*70)
    
    df_feat = df.copy()
    
    # 1. VOC signature ratio
    df_feat['mq_voc_ratio'] = df_feat['mq3_voltage'] / (df_feat['mq135_voltage'] + 0.001)
    
    # 2. Temperature-normalized gas resistance
    df_feat['gas_temp_normalized'] = df_feat['gas_resistance_kOhm'] / df_feat['temp_C']
    
    # 3. Humidity correction
    df_feat['humidity_factor'] = df_feat['humidity_%'] / 100.0
    
    # 4. Combined sensor response (weighted)
    df_feat['sensor_combined'] = (
        df_feat['gas_resistance_kOhm'] * 0.5 + 
        df_feat['mq135_ratio'] * 10 +
        df_feat['mq3_ratio'] * 10
    )
    
    # 5. MQ sensor raw ratio
    df_feat['mq_raw_ratio'] = df_feat['mq135_raw'] / (df_feat['mq3_raw'] + 1)
    
    # 6. Environmental stability indicator
    df_feat['env_stability'] = df_feat['temp_C'] * df_feat['humidity_factor']
    
    print("\n✓ Created 6 engineered features")
    
    return df_feat

# ============================================================================
# PART 4: REMOVE REDUNDANT FEATURES (FRIEND'S KEY ADVICE)
# ============================================================================

def remove_redundant_features(df, threshold=0.95):
    """Remove highly correlated redundant features"""
    print("\n" + "="*70)
    print("STEP 4: REMOVING REDUNDANT FEATURES")
    print("="*70)
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols 
                   if col not in ['timestamp', 'contamination_level_%']]
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Identify redundant features
    redundant_features = []
    print(f"\n🔍 Identifying features with correlation > {threshold}:")
    
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][upper_triangle[column] > threshold]
        if len(correlated) > 0:
            for idx, corr_value in correlated.items():
                print(f"   • {column} ↔ {idx}: {corr_value:.3f}")
                # Keep the feature with higher correlation to target
                target_corr_1 = abs(df[column].corr(df['contamination_level_%']))
                target_corr_2 = abs(df[idx].corr(df['contamination_level_%']))
                
                if target_corr_1 < target_corr_2:
                    redundant_features.append(column)
                else:
                    redundant_features.append(idx)
    
    # Remove duplicates
    redundant_features = list(set(redundant_features))
    
    print(f"\n✓ Removing {len(redundant_features)} redundant features:")
    for feat in redundant_features:
        print(f"   - {feat}")
    
    # Keep only non-redundant features
    features_to_keep = [col for col in feature_cols if col not in redundant_features]
    
    return features_to_keep, redundant_features

# ============================================================================
# PART 5: CORRELATION-BASED FEATURE SELECTION
# ============================================================================

def select_features_by_correlation(df, features, target_col='contamination_level_%', 
                                   min_correlation=0.1):
    """Remove features with low correlation to target (CORRECTED FRIEND'S ADVICE)"""
    print("\n" + "="*70)
    print("STEP 5: CORRELATION-BASED FEATURE SELECTION")
    print("="*70)
    
    print(f"\n🎯 Analyzing correlation with target: {target_col}")
    print(f"   Minimum correlation threshold: {min_correlation}")
    
    # Calculate correlation with target
    correlations = {}
    for feature in features:
        if feature in df.columns:
            corr = abs(df[feature].corr(df[target_col]))
            correlations[feature] = corr
    
    # Sort by correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 Feature Correlations with Target:")
    for feat, corr in sorted_corr:
        status = "✓" if corr >= min_correlation else "✗"
        print(f"   {status} {feat}: {corr:.4f}")
    
    # Select features above threshold
    selected_features = [feat for feat, corr in sorted_corr if corr >= min_correlation]
    removed_features = [feat for feat, corr in sorted_corr if corr < min_correlation]
    
    print(f"\n✓ Selected {len(selected_features)} features above threshold")
    print(f"✗ Removed {len(removed_features)} features below threshold")
    
    return selected_features, removed_features

# ============================================================================
# PART 6: FINAL FEATURE SET SELECTION
# ============================================================================

def select_final_features(df):
    """Select final optimized feature set"""
    print("\n" + "="*70)
    print("STEP 6: FINAL FEATURE SET SELECTION")
    print("="*70)
    
    # Start with all numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_features = [col for col in numeric_cols 
                         if col not in ['timestamp', 'contamination_level_%']]
    
    # Apply redundancy removal
    features_after_redundancy, redundant = remove_redundant_features(df, threshold=0.95)
    
    # Apply correlation filtering
    final_features, low_corr = select_features_by_correlation(
        df, features_after_redundancy, min_correlation=0.05
    )
    
    print(f"\n📋 Final Feature Set ({len(final_features)} features):")
    for i, feat in enumerate(final_features, 1):
        corr = abs(df[feat].corr(df['contamination_level_%']))
        print(f"   {i}. {feat} (corr: {corr:.4f})")
    
    return final_features

# ============================================================================
# PART 7: PREPARE ML DATA
# ============================================================================

def prepare_ml_data(df, ml_features):
    """Prepare train/test splits for classification and regression"""
    print("\n" + "="*70)
    print("STEP 7: PREPARING ML DATASETS")
    print("="*70)
    
    print("\n📌 Note: contamination_level_% represents % of PURE turmeric")
    print("   100% = pure turmeric (best)")
    print("   75%  = 25% flour contamination (good)")
    print("   50%  = 50% flour contamination (poor)")
    print("   25%  = 75% flour contamination (worst)")
    
    X = df[ml_features].values
    
    # Binary classification: High purity (>=75%) vs Low purity (<75%)
    df['binary_class'] = df['contamination_level_%'].apply(
        lambda x: 'high_purity' if x >= 75 else 'low_purity'
    )
    
    # Multi-class classification based on purity levels
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
    
    # Split and scale
    X_train_clf, X_test_clf, y_train_binary, y_test_binary = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    scaler_clf = StandardScaler()
    scaler_reg = StandardScaler()
    
    X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler_clf.transform(X_test_clf)
    X_train_multi_scaled = scaler_clf.transform(X_train_multi)
    X_test_multi_scaled = scaler_clf.transform(X_test_multi)
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    print(f"\n✓ Binary Classification: {X_train_clf_scaled.shape[0]} train, {X_test_clf_scaled.shape[0]} test")
    print(f"✓ Multi-Class Classification: {X_train_multi_scaled.shape[0]} train, {X_test_multi_scaled.shape[0]} test")
    print(f"✓ Regression: {X_train_reg_scaled.shape[0]} train, {X_test_reg_scaled.shape[0]} test")
    
    return {
        'binary': (X_train_clf_scaled, X_test_clf_scaled, y_train_binary, y_test_binary, le_binary, scaler_clf),
        'multi': (X_train_multi_scaled, X_test_multi_scaled, y_train_multi, y_test_multi, le_multi, scaler_clf),
        'regression': (X_train_reg_scaled, X_test_reg_scaled, y_train_reg, y_test_reg, scaler_reg),
        'feature_names': ml_features
    }

# ============================================================================
# PART 8: TRAIN CLASSIFICATION MODELS
# ============================================================================

def train_classification_models(data_dict, task='binary'):
    """Train multiple classification models (FRIEND'S ADVICE)"""
    print("\n" + "="*70)
    print(f"STEP 8: TRAINING {task.upper()} CLASSIFICATION MODELS")
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
# PART 9: TRAIN REGRESSION MODELS
# ============================================================================

def train_regression_models(data_dict):
    """Train multiple regression models (FRIEND'S ADVICE)"""
    print("\n" + "="*70)
    print("STEP 9: TRAINING REGRESSION MODELS")
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
# MAIN PIPELINE
# ============================================================================

def main(csv_file):
    """Optimized ML pipeline incorporating friend's advice"""
    print("\n" + "="*70)
    print("SPICE PURITY ML PIPELINE - OPTIMIZED")
    print("="*70)
    
    df = load_and_explore_data(csv_file)
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    ml_features = select_final_features(df_feat)
    data_dict = prepare_ml_data(df_feat, ml_features)
    
    binary_results = train_classification_models(data_dict, task='binary')
    multi_results = train_classification_models(data_dict, task='multi')
    regression_results = train_regression_models(data_dict)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    
    return {
        'data': df_feat,
        'binary_clf': binary_results,
        'multi_clf': multi_results,
        'regression': regression_results,
        'data_dict': data_dict,  # ← Added this!
        'features': ml_features
    }

if __name__ == "__main__":
    CSV_FILE = "spice_data.csv"
    results = main(CSV_FILE)