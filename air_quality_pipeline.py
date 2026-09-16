#!/usr/bin/env python3
"""
NLP-Based Air Quality Health Advisory Generator
11-Step ML Pipeline for Indian CPCB Standards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


CPCB_CATEGORIES = ['Good', 'Satisfactory', 'Moderately Polluted', 'Poor', 'Very Poor', 'Severe']
DEMOGRAPHIC_GROUPS = ['General', 'Children', 'Elderly', 'Sensitive', 'Asthma']


def classify_aqi(aqi_value):
    if aqi_value <= 50:
        return 'Good'
    elif aqi_value <= 100:
        return 'Satisfactory'
    elif aqi_value <= 200:
        return 'Moderately Polluted'
    elif aqi_value <= 300:
        return 'Poor'
    elif aqi_value <= 400:
        return 'Very Poor'
    return 'Severe'


def generate_advisory(aqi, category, dominant_pollutant, temperature, humidity,
                      wind_speed, location, target_group='General'):
    base_advisories = {
        'Good': f"Air quality in {location} is Good ({aqi} AQI). Minimal health impact; safe for outdoor activities.",
        'Satisfactory': f"Air quality in {location} is Satisfactory ({aqi} AQI). May cause minor breathing discomfort to sensitive people.",
        'Moderately Polluted': f"Air quality in {location} is Moderately Polluted ({aqi} AQI). Discomfort to people with lung/heart disease, children, and older adults.",
        'Poor': f"Air quality in {location} is Poor ({aqi} AQI). Breathing discomfort to most people on prolonged exposure.",
        'Very Poor': f"Air quality in {location} is Very Poor ({aqi} AQI). Respiratory illness on prolonged exposure; severe impact on sensitive groups.",
        'Severe': f"Air quality in {location} is Severe ({aqi} AQI). Affects healthy people and seriously impacts those with pre-existing conditions."
    }

    advisory = base_advisories.get(category, "")
    advisory += f" The dominant pollutant is {dominant_pollutant}."

    weather_parts = []
    if temperature > 35:
        weather_parts.append(f"High temperature ({temperature}°C) increases respiratory stress.")
    if humidity > 70:
        weather_parts.append(f"Elevated humidity ({humidity}%) aggravates respiratory conditions.")
    if wind_speed and wind_speed < 5:
        weather_parts.append(f"Low wind speed ({wind_speed} km/h) causes pollutant accumulation.")
    if weather_parts:
        advisory += " " + " ".join(weather_parts)

    demographic_advice = {
        'Children': "Children should limit outdoor activities and ensure proper ventilation indoors.",
        'Elderly': "Elderly individuals should avoid prolonged outdoor exposure and keep medications ready.",
        'Sensitive': "Individuals with respiratory conditions should stay indoors and use air purifiers.",
        'Asthma': "Asthma patients should carry inhalers and avoid outdoor exercise. Keep windows closed.",
        'General': ""
    }
    mod = demographic_advice.get(target_group, "")
    if mod:
        advisory += " " + mod

    recommendations = {
        'Good': "No restrictions. Enjoy outdoor activities safely.",
        'Satisfactory': "Reduce intense outdoor activity if you have respiratory sensitivities.",
        'Moderately Polluted': "Reduce prolonged exertion. Keep windows closed if dust is suspected.",
        'Poor': "Avoid outdoor activities. Use N95 masks if going outside is necessary.",
        'Very Poor': "Stay indoors with windows closed. Use air purifiers if available.",
        'Severe': "Remain indoors. Use high-efficiency air filters. Seek medical attention if symptoms appear."
    }
    advisory += " " + recommendations.get(category, "")

    return advisory.strip()


def load_all_city_data(data_dir):
    city_files = {
        'Bangalore': 'Bangalore_AQI_Dataset.csv',
        'Chennai': 'Chennai_AQI_Dataset.csv',
        'Delhi': 'Delhi_AQI_Dataset.csv',
        'Hyderabad': 'Hyderabad_AQI_Dataset.csv',
        'Mumbai': 'Mumbai_AQI_Dataset.csv'
    }

    all_data = {}
    for city, filename in city_files.items():
        filepath = Path(data_dir) / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['City'] = city
            all_data[city] = df
            print(f"Loaded {city}: {len(df)} records")
        else:
            print(f"Not found: {filepath}")

    combined = pd.concat(all_data.values(), ignore_index=True)
    print(f"Total: {len(combined):,} samples across {len(all_data)} cities\n")
    return combined, all_data


def perform_eda(df):
    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDescriptive Statistics:")
    print(df.describe())

    df['CPCB_Category'] = df['AQI'].apply(classify_aqi)
    print("\nCPCB Category Distribution:")
    print(df['CPCB_Category'].value_counts())

    return df


def visualize_data(df, save_dir=None):
    print("\n" + "=" * 60)
    print("STEP 4: DATA VISUALIZATION")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    sns.countplot(data=df, x='CPCB_Category', palette='viridis', ax=axes[0, 0],
                  order=CPCB_CATEGORIES)
    axes[0, 0].set_title('CPCB AQI Category Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)

    axes[0, 1].hist(df['AQI'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('AQI Distribution')
    axes[0, 1].set_xlabel('AQI')
    axes[0, 1].set_ylabel('Frequency')

    axes[0, 2].hist(df['PM2.5'], bins=50, color='orange', edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('PM2.5 Distribution')
    axes[0, 2].set_xlabel('PM2.5 (µg/m³)')

    sns.boxplot(data=df[['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']], palette='Set2', ax=axes[1, 0])
    axes[1, 0].set_title('Outlier Detection - Key Pollutants')
    axes[1, 0].tick_params(axis='x', rotation=45)

    corr_cols = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=axes[1, 1])
    axes[1, 1].set_title('Pollutant Correlation Heatmap')

    sns.boxplot(data=df, x='City', y='AQI', palette='pastel', ax=axes[1, 2])
    axes[1, 2].set_title('AQI by City')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / 'eda_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


def preprocess_data(df, noise_level=0.18, random_state=42):
    print("\n" + "=" * 60)
    print("STEP 5: DATA PREPROCESSING")
    print("=" * 60)

    df = df.copy()
    # AQI kept for label creation only — never used as a model feature
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    numerical_cols = ['AQI'] + pollutant_cols

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: imputed with median ({median_val:.2f})")

    # Labels from clean AQI (before any feature noise)
    df['CPCB_Category'] = df['AQI'].apply(classify_aqi)

    # Dataset pollutants are exact linear transforms of AQI (corr=1.0).
    # Inject independent measurement noise so models cannot trivially invert AQI.
    rng = np.random.default_rng(random_state)
    for col in pollutant_cols:
        relative = rng.normal(0.0, noise_level, size=len(df))
        df[col] = (df[col] * (1.0 + relative)).clip(lower=0)
    print(f"  Injected ~{noise_level*100:.0f}% independent measurement noise per pollutant")
    print(f"  (breaks synthetic AQI-pollutant collinearity; simulates sensor error)")
    print(f"  Classes after cleaning: {df['CPCB_Category'].value_counts().to_dict()}")

    scaler = MinMaxScaler()
    df[pollutant_cols] = scaler.fit_transform(df[pollutant_cols])
    print(f"  Scaled {len(pollutant_cols)} pollutant features (AQI excluded)")

    le = LabelEncoder()
    if 'City' in df.columns:
        df['City_Encoded'] = le.fit_transform(df['City'])
        print(f"  City: {len(le.classes_)} categories encoded")

    return df, scaler, le


def separate_features_target(df):
    print("\n" + "=" * 60)
    print("STEP 6: SEPARATE INPUT (X) AND OUTPUT (y)")
    print("=" * 60)

    # CRITICAL: exclude AQI — CPCB_Category is a direct function of AQI (target leakage)
    feature_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'City_Encoded']
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df['CPCB_Category'].copy()

    print(f"X shape: {X.shape} | Features: {list(X.columns)}")
    print(f"  (AQI excluded to prevent target leakage)")
    print(f"y shape: {y.shape} | Classes: {y.nunique()} ({list(y.unique())})")
    print(f"Class counts:\n{y.value_counts()}")

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    print("\n" + "=" * 60)
    print("STEP 7: TRAIN-TEST SPLIT (Stratified with Rare Class Handling)")
    print("=" * 60)

    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    min_class_name = class_counts.idxmin()
    
    # If rare classes exist (fewer than 5 samples), use random split with class weights
    if min_class_count < 5:
        print(f"Note: Rare class '{min_class_name}' has only {min_class_count} sample(s).")
        print("  Using random split with class weights for model training.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"  Random split applied (no stratification due to rare classes)")
    elif min_class_count < 10:
        # Medium rare: use stratified but be aware of variance
        print(f"Note: Class '{min_class_name}' has {min_class_count} sample(s).")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"  Split with stratified sampling (expect some variance on rare class)")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Stratified split applied normally")

    print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Print train/test class distribution for verification
    print("Train set class distribution:")
    print(y_train.value_counts())
    print("\nTest set class distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def train_baseline_models(X_train, y_train):
    print("\n" + "=" * 60)
    print("STEP 8: TRAIN BASELINE MODELS WITH REGULARIZATION")
    print("=" * 60)

    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    n_classes = len(le_target.classes_)

    models = {
        "Logistic Regression": LogisticRegression(
            solver='lbfgs', max_iter=2000, random_state=42,
            class_weight='balanced', penalty='l2', C=0.5
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, random_state=42, class_weight='balanced',
            min_samples_leaf=20, min_samples_split=40, ccp_alpha=0.001
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            class_weight='balanced', n_jobs=-1,
            min_samples_leaf=15, min_samples_split=30,
            max_features='sqrt'
        ),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            objective='multi:softprob', eval_metric='mlogloss',
            random_state=42, n_jobs=-1,
            reg_lambda=2.0, reg_alpha=0.5,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5
        )
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            objective='multiclass', num_class=n_classes,
            random_state=42, n_jobs=-1, verbosity=-1,
            reg_alpha=0.5, reg_lambda=2.0,
            min_child_samples=20, min_split_gain=0.01,
            subsample=0.8, colsample_bytree=0.8
        )

    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        if name in ["XGBoost", "LightGBM"]:
            model.fit(X_train, y_train_encoded)
            train_acc = model.score(X_train, y_train_encoded)
        else:
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
        trained[name] = (model, le_target if name in ["XGBoost", "LightGBM"] else None)
        print(f"  Train Accuracy: {train_acc*100:.2f}%")

    return trained


def evaluate_models(models, X_test, y_test, X_train=None, y_train=None):
    print("\n" + "=" * 60)
    print("STEP 9: MODEL EVALUATION WITH CROSS-VALIDATION INSIGHTS")
    print("=" * 60)

    results = {}
    class_names = CPCB_CATEGORIES

    for name, model_tuple in models.items():
        model, le_target = model_tuple

        if name in ["XGBoost", "LightGBM"]:
            y_pred_encoded = model.predict(X_test)
            y_pred = le_target.inverse_transform(y_pred_encoded)
            if X_train is not None:
                train_acc = model.score(X_train, le_target.transform(y_train))
            else:
                train_acc = None
        else:
            y_pred = model.predict(X_test)
            train_acc = model.score(X_train, y_train) if X_train is not None else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'f1_macro': f1_macro, 'train_accuracy': train_acc,
            'y_pred': y_pred,
            'report': classification_report(y_test, y_pred, target_names=class_names,
                                           labels=class_names, output_dict=True, zero_division=0)
        }

        gap = (train_acc - acc) if train_acc is not None else None
        print(f"\n{name}:")
        if train_acc is not None:
            print(f"  Train Acc:     {train_acc*100:.2f}%")
        print(f"  Test Acc:      {acc*100:.2f}%")
        if gap is not None:
            print(f"  Train-Test gap:{gap*100:.2f}%")
        print(f"  Precision (w): {prec*100:.2f}%")
        print(f"  Recall (w):    {rec*100:.2f}%")
        print(f"  F1 (weighted): {f1*100:.2f}%")
        print(f"  F1 (macro):    {f1_macro*100:.2f}%")

    return results


def plot_confusion_matrices(models, X_test, y_test, save_dir=None):
    print("\n" + "=" * 60)
    print("STEP 10: CONFUSION MATRICES")
    print("=" * 60)

    class_names = ['Good', 'Satisfactory', 'Moderately\nPolluted', 'Poor', 'Very\nPoor', 'Severe']
    n_models = len(models)

    fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models))
    if n_models == 1:
        axes = [axes]

    for idx, (name, model_tuple) in enumerate(models.items()):
        model, le_target = model_tuple

        if name in ["XGBoost", "LightGBM"]:
            y_pred_encoded = model.predict(X_test)
            y_pred = le_target.inverse_transform(y_pred_encoded)
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=CPCB_CATEGORIES)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(f'{name} - Confusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_models(results):
    print("\n" + "=" * 60)
    print("STEP 11: MODEL COMPARISON & SELECTION")
    print("=" * 60)

    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1-Score': [r['f1'] for r in results.values()],
        'F1-Macro': [r['f1_macro'] for r in results.values()],
        'Train-Acc': [r.get('train_accuracy') for r in results.values()],
    }).sort_values('F1-Score', ascending=False).reset_index(drop=True)

    print(f"\n{'Rank':<6} {'Model':<20} {'Train':<8} {'Test':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'MacroF1':<8}")
    print("-" * 80)
    for i, row in comparison.iterrows():
        train_s = f"{row['Train-Acc']*100:.2f}%" if row['Train-Acc'] is not None else "n/a"
        print(f"{i+1:<6} {row['Model']:<20} {train_s:<8} {row['Accuracy']*100:<8.2f}% "
              f"{row['Precision']*100:<8.2f}% {row['Recall']*100:<8.2f}% "
              f"{row['F1-Score']*100:<8.2f}% {row['F1-Macro']*100:<8.2f}%")

    best_model = comparison.iloc[0]['Model']
    print(f"\nBest Model: {best_model} (F1: {comparison.iloc[0]['F1-Score']*100:.2f}%)")

    best_report = results[best_model]['report']
    print(f"\nPer-Class Performance ({best_model}):")
    for cls in CPCB_CATEGORIES:
        if cls in best_report:
            m = best_report[cls]
            print(f"  {cls:<25}: P={m['precision']*100:>6.2f}% R={m['recall']*100:>6.2f}% F1={m['f1-score']*100:>6.2f}%")

    return comparison, best_model


def demo_advisory_generation():
    print("\n" + "=" * 60)
    print("HEALTH ADVISORY GENERATION DEMO")
    print("=" * 60)

    scenarios = [
        {'city': 'Delhi', 'aqi': 310, 'pollutant': 'PM2.5', 'temp': 32, 'humidity': 75, 'wind': 5, 'group': 'Children'},
        {'city': 'Bangalore', 'aqi': 78, 'pollutant': 'PM10', 'temp': 25, 'humidity': 60, 'wind': 10, 'group': 'Elderly'},
        {'city': 'Mumbai', 'aqi': 155, 'pollutant': 'PM2.5', 'temp': 30, 'humidity': 80, 'wind': 8, 'group': 'Asthma'},
        {'city': 'Chennai', 'aqi': 95, 'pollutant': 'O3', 'temp': 35, 'humidity': 70, 'wind': 12, 'group': 'General'},
        {'city': 'Hyderabad', 'aqi': 220, 'pollutant': 'PM2.5', 'temp': 28, 'humidity': 55, 'wind': 6, 'group': 'Sensitive'},
    ]

    for s in scenarios:
        cat = classify_aqi(s['aqi'])
        advisory = generate_advisory(s['aqi'], cat, s['pollutant'], s['temp'],
                                     s['humidity'], s['wind'], s['city'], s['group'])
        print(f"\n{s['city']} | AQI: {s['aqi']} ({cat}) | Group: {s['group']}")
        print(f"  {advisory}")


def run_full_pipeline(data_dir, save_dir=None):
    print("=" * 70)
    print("NLP-BASED AIR QUALITY HEALTH ADVISORY GENERATOR")
    print("11-Step ML Pipeline (CPCB India Standards)")
    print("=" * 70)

    combined_df, _ = load_all_city_data(data_dir)
    combined_df = perform_eda(combined_df)
    visualize_data(combined_df, save_dir)
    df_clean, scaler, encoder = preprocess_data(combined_df)
    X, y = separate_features_target(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y)
    trained_models = train_baseline_models(X_train, y_train)
    results = evaluate_models(trained_models, X_test, y_test, X_train, y_train)
    plot_confusion_matrices(trained_models, X_test, y_test, save_dir)
    comparison, best_model = compare_models(results)
    demo_advisory_generation()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Dataset: {len(combined_df):,} samples, 5 Indian cities")
    print(f"Best Model: {best_model} (F1: {comparison.iloc[0]['F1-Score']*100:.2f}%)")
    print("Advisory Engine: Rule-based, CPCB-compliant, demographic-aware")
    print("Next: LSTM forecasting + T5 fine-tuning (Phase 4)")

    return {
        'models': trained_models,
        'best_model': best_model,
        'comparison': comparison,
        'scaler': scaler,
        'encoder': encoder,
        'results': results
    }


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\wsrir\OneDrive\Desktop\coding bhai\NLP project"
    SAVE_DIR = r"C:\Users\wsrir\OneDrive\Desktop\coding bhai\NLP project\outputs"

    Path(SAVE_DIR).mkdir(exist_ok=True)
    run_full_pipeline(DATA_DIR, SAVE_DIR)