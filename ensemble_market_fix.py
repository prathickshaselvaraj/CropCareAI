# ensemble_market_fix.py
"""
ENSEMBLE FIX FOR MARKET ANALYSIS MODULE
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def create_ensemble_market_predictor():
    """Create ensemble model for market analysis"""
    print("🔄 CREATING ENSEMBLE MARKET PREDICTOR...")
    
    try:
        d7_data = pd.read_csv("data/processed/D7_processed.csv")
        
        # Use multiple target options
        target_options = ['Market_Price_per_ton', 'Demand_Index', 'Supply_Index', 'Consumer_Trend_Index']
        target_col = None
        
        for option in target_options:
            if option in d7_data.columns:
                target_col = option
                break
        
        if not target_col:
            print("❌ No suitable target column found")
            return None, 0
        
        # Feature selection
        numeric_cols = d7_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col][:6]  # Use top 6 features
        
        if len(feature_cols) < 2:
            print("❌ Not enough features")
            return None, 0
        
        X = d7_data[feature_cols].fillna(0)
        y = d7_data[target_col]
        
        print(f"🎯 Target: {target_col}, Features: {len(feature_cols)}")
        
        # Remove outliers
        Q1, Q3 = y.quantile(0.1), y.quantile(0.9)
        mask = (y >= Q1) & (y <= Q3)
        X, y = X[mask], y[mask]
        
        if len(X) < 500:
            print("❌ Not enough data after cleaning")
            return None, 0
        
        # Train multiple models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('ridge', Ridge(alpha=1.0)),
            ('linear', LinearRegression())
        ]
        
        # Create ensemble
        ensemble = VotingRegressor(models)
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ ENSEMBLE R²: {r2:.3f}")
        
        # Save ensemble
        model_path = "src/modules/market_analysis/models/price_ensemble.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        
        return ensemble, r2
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, 0

# Run ensemble approach for market
ensemble, score = create_ensemble_market_predictor()
if ensemble:
    print(f"🎯 Market ensemble created with R²: {score:.3f}")
else:
    print("❌ Market ensemble failed")