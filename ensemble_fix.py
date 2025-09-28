# ensemble_fix.py
"""
COMPREHENSIVE ENSEMBLE SOLUTION FOR YIELD AND MARKET PREDICTION
Combines multiple models and approaches
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def create_ensemble_yield_predictor():
    """Create ensemble model for yield prediction"""
    print("🔄 CREATING ENSEMBLE YIELD PREDICTOR...")
    
    try:
        d3_data = pd.read_csv("data/processed/D3_processed.csv")
        
        # Feature engineering
        features = []
        if 'area' in d3_data.columns: features.append('area')
        if 'temperature' in d3_data.columns: features.append('temperature') 
        if 'humidity' in d3_data.columns: features.append('humidity')
        if 'N' in d3_data.columns: features.append('N')
        if 'P' in d3_data.columns: features.append('P')
        if 'K' in d3_data.columns: features.append('K')
        if 'precipitation' in d3_data.columns: features.append('precipitation')
        
        if not features:
            print("❌ No features available")
            return None, 0
        
        X = d3_data[features].fillna(0)
        y = d3_data['production_cleaned'] if 'production_cleaned' in d3_data.columns else d3_data.iloc[:, -1]
        
        # Remove extreme outliers
        Q1, Q3 = y.quantile(0.05), y.quantile(0.95)
        mask = (y >= Q1) & (y <= Q3)
        X, y = X[mask], y[mask]
        
        if len(X) < 1000:
            print("❌ Not enough data after cleaning")
            return None, 0
        
        # Sample for speed
        if len(X) > 20000:
            sample_idx = np.random.choice(len(X), 20000, replace=False)
            X, y = X.iloc[sample_idx], y.iloc[sample_idx]
        
        # Train multiple models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ]
        
        # Create ensemble
        ensemble = VotingRegressor(models)
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ ENSEMBLE R²: {r2:.3f}")
        
        # Save ensemble
        model_path = "src/modules/yield_analysis/models/yield_ensemble.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        
        return ensemble, r2
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, 0

# Run ensemble approach
ensemble, score = create_ensemble_yield_predictor()
if ensemble:
    print(f"🎯 Ensemble model created with R²: {score:.3f}")