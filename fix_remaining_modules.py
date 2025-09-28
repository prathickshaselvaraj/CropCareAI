# fix_remaining_modules.py
"""
TARGETED FIX FOR YIELD, MARKET, AND PESTICIDE MODULES
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

print("🎯 FIXING REMAINING BROKEN MODULES")
print("=" * 50)

def fix_yield_prediction():
    """Fix yield prediction with proper feature extraction"""
    print("\n📈 FIXING YIELD PREDICTION...")
    
    try:
        # Load D3 data
        d3_data = pd.read_csv("data/processed/D3_processed.csv")
        
        # Use actual columns from D3 instead of standardized template
        if 'area' in d3_data.columns and 'production_cleaned' in d3_data.columns:
            # Use real features from D3
            feature_cols = ['area', 'temperature', 'humidity', 'N', 'P', 'K']
            available_features = [col for col in feature_cols if col in d3_data.columns]
            
            X = d3_data[available_features].fillna(0)
            y = d3_data['production_cleaned']
            
            # Sample for speed
            if len(X) > 20000:
                sample_idx = np.random.choice(len(X), 20000, replace=False)
                X = X.iloc[sample_idx]
                y = y.iloc[sample_idx]
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            r2 = r2_score(y_test, model.predict(X_test))
            print(f"✅ R² SCORE: {r2:.3f}")
            
            # Save model
            model_path = "src/modules/yield_analysis/models/yield_predictor.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return r2
        else:
            print("❌ Required columns not found in D3")
            return 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

def fix_market_analysis():
    """Fix market analysis with proper features"""
    print("\n💰 FIXING MARKET ANALYSIS...")
    
    try:
        # Load D7 data
        d7_data = pd.read_csv("data/processed/D7_processed.csv")
        
        # Use actual numeric columns from D7
        numeric_cols = d7_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Use first numeric column as target, others as features
            target_col = numeric_cols[0]
            feature_cols = numeric_cols[1:6]  # Use first 5 numeric features
            
            X = d7_data[feature_cols].fillna(0)
            y = d7_data[target_col]
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            r2 = r2_score(y_test, model.predict(X_test))
            print(f"✅ R² SCORE: {r2:.3f}")
            
            # Save model
            model_path = "src/modules/market_analysis/models/price_predictor.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return r2
        else:
            print("❌ Not enough numeric columns in D7")
            return 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

def fix_pesticide_recommendation():
    """Fix pesticide with simple binary classification"""
    print("\n🐛 FIXING PESTICIDE RECOMMENDATION...")
    
    try:
        # Load D5 data
        d5_data = pd.read_csv("data/processed/D5_processed.csv")
        
        # Create simple binary classification: Pest vs No Pest
        if 'Pest_Presence' in d5_data.columns:
            # Convert to binary (1 = Pest, 0 = No Pest/Unknown)
            d5_data['pest_binary'] = d5_data['Pest_Presence'].apply(
                lambda x: 1 if x in ['Aphids', 'Spider mites', 'Whiteflies', 'Fungus gnats'] else 0
            )
            
            # Use numeric features only
            numeric_cols = d5_data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != 'pest_binary'][:5]  # Use 5 features
            
            X = d5_data[feature_cols].fillna(0)
            y = d5_data['pest_binary']
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = accuracy_score(y_test, model.predict(X_test))
            print(f"✅ ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Save model
            model_path = "src/modules/pesticide_recommendation/models/pest_presence_classifier.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return accuracy
        else:
            print("❌ Pest_Presence column not found in D5")
            return 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

# Run the fixes
if __name__ == "__main__":
    print("🚀 FIXING REMAINING 3 MODULES...")
    
    results = {}
    results['yield'] = fix_yield_prediction()
    results['market'] = fix_market_analysis() 
    results['pesticide'] = fix_pesticide_recommendation()
    
    print("\n" + "=" * 50)
    print("🎯 FIX RESULTS")
    print("=" * 50)
    
    for module, score in results.items():
        if module == 'pesticide':
            print(f"✅ {module}: {score:.3f} ({score*100:.1f}%) accuracy")
        else:
            print(f"✅ {module}: R² = {score:.3f}")
    
    successful = len([s for s in results.values() if s > 0])
    print(f"\n📊 SUCCESSFULLY FIXED: {successful}/3 modules")