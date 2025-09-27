# src/models/proper_pesticide_models.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def build_proper_pesticide_models():
    """Build proper pesticide recommendation models using D5 data"""
    print("🐛 Building Proper Pesticide Recommendation Models...")
    
    project_root = Path(__file__).parent.parent.parent
    processed_path = project_root / "data/processed/"
    module_path = project_root / "src/modules/pesticide_recommendation/models/"
    module_path.mkdir(parents=True, exist_ok=True)
    
    # Load D5 data (indoor plants with pest information)
    d5_data = pd.read_csv(processed_path / "D5_processed.csv")
    
    print("📊 Dataset Info:")
    print(f"Shape: {d5_data.shape}")
    print(f"Pest Presence distribution:\n{d5_data['Pest_Presence'].value_counts()}")
    print(f"Pest Severity distribution:\n{d5_data['Pest_Severity'].value_counts()}")
    
    models = {}
    
    # MODEL 1: Pest Presence Classifier (What pest is present?)
    print("\n🔧 1. Training Pest Presence Classifier...")
    
    # Prepare features and target
    X = d5_data.drop(columns=['Pest_Presence', 'Pest_Severity', 'Plant_ID'])
    y = d5_data['Pest_Presence']
    
    # Handle missing values in target
    y_clean = y[y != 'Unknown']  # Remove 'Unknown' labels
    X_clean = X.loc[y_clean.index]
    
    print(f"🎯 Target: Pest Presence ({len(y_clean.unique())} types)")
    print(f"📊 Class distribution:\n{y_clean.value_counts()}")
    
    # Encode categorical features
    categorical_cols = X_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X_clean[col].nunique() <= 10:
            dummies = pd.get_dummies(X_clean[col], prefix=col, drop_first=True)
            X_clean = pd.concat([X_clean, dummies], axis=1)
            X_clean.drop(columns=[col], inplace=True)
        else:
            le = LabelEncoder()
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))
    
    # Keep only numeric columns
    X_clean = X_clean.select_dtypes(include=[np.number])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Train model
    pest_type_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pest_type_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pest_type_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Pest Presence Classifier trained!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📈 Classification Report:\n{classification_report(y_test, y_pred)}")
    
    models['pest_presence_classifier'] = (pest_type_model, accuracy)
    
    # MODEL 2: Pest Severity Classifier (How severe is the infestation?)
    print("\n🔧 2. Training Pest Severity Classifier...")
    
    # Use only samples with known pests
    severity_data = d5_data[d5_data['Pest_Presence'] != 'Unknown']
    X_sev = severity_data.drop(columns=['Pest_Presence', 'Pest_Severity', 'Plant_ID'])
    y_sev = severity_data['Pest_Severity']
    
    print(f"🎯 Target: Pest Severity ({len(y_sev.unique())} levels)")
    print(f"📊 Class distribution:\n{y_sev.value_counts()}")
    
    # Encode features (same process as above)
    categorical_cols_sev = X_sev.select_dtypes(include=['object']).columns
    for col in categorical_cols_sev:
        if X_sev[col].nunique() <= 10:
            dummies = pd.get_dummies(X_sev[col], prefix=col, drop_first=True)
            X_sev = pd.concat([X_sev, dummies], axis=1)
            X_sev.drop(columns=[col], inplace=True)
        else:
            le = LabelEncoder()
            X_sev[col] = le.fit_transform(X_sev[col].astype(str))
    
    X_sev = X_sev.select_dtypes(include=[np.number])
    
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=42, stratify=y_sev
    )
    
    severity_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    severity_model.fit(X_train_sev, y_train_sev)
    
    y_pred_sev = severity_model.predict(X_test_sev)
    accuracy_sev = accuracy_score(y_test_sev, y_pred_sev)
    
    print(f"✅ Pest Severity Classifier trained!")
    print(f"📊 Accuracy: {accuracy_sev:.4f}")
    print(f"📈 Classification Report:\n{classification_report(y_test_sev, y_pred_sev)}")
    
    models['pest_severity_classifier'] = (severity_model, accuracy_sev)
    
    # MODEL 3: Treatment Recommendation (Based on pest type and severity)
    print("\n🔧 3. Creating Treatment Recommendation Logic...")
    
    # Create treatment mapping based on pest type and severity
    treatment_rules = {
        'Aphids': {
            'Low': 'Neem oil spray',
            'Moderate': 'Insecticidal soap', 
            'High': 'Pyrethrin-based insecticide'
        },
        'Spider mites': {
            'Low': 'Water spray',
            'Moderate': 'Horticultural oil',
            'High': 'Miticide application'
        },
        'Whiteflies': {
            'Low': 'Yellow sticky traps',
            'Moderate': 'Insecticidal soap',
            'High': 'Systemic insecticide'
        },
        'Fungus gnats': {
            'Low': 'Reduce watering',
            'Moderate': 'Beneficial nematodes',
            'High': 'Bacillus thuringiensis'
        }
    }
    
    models['treatment_recommender'] = treatment_rules
    
    # Save all models
    for model_name, model_data in models.items():
        filename = module_path / f"{model_name}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model_data[0] if isinstance(model_data, tuple) else model_data, f)
        print(f"💾 Saved {model_name} to {filename}")
    
    print(f"\n🎉 Pesticide module built with {len(models)} models!")
    
    return models

if __name__ == "__main__":
    build_proper_pesticide_models()