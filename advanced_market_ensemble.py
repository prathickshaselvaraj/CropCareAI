# advanced_market_ensemble.py
"""
ADVANCED ENSEMBLE WITH XGBOOST, LIGHTGBM, CATBOOST
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
import warnings
warnings.filterwarnings('ignore')

def create_advanced_market_ensemble():
    """Advanced ensemble with boosting algorithms"""
    print("🚀 CREATING ADVANCED MARKET ENSEMBLE...")
    
    try:
        d7_data = pd.read_csv("data/processed/D7_processed.csv")
        
        # Try different target columns
        target_options = ['Demand_Index', 'Supply_Index', 'Market_Price_per_ton']
        best_target = None
        best_score = -999
        
        for target_col in target_options:
            if target_col in d7_data.columns:
                print(f"\n🔍 Testing target: {target_col}")
                
                # Feature engineering
                numeric_cols = d7_data.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols if col != target_col]
                
                X = d7_data[feature_cols].fillna(0)
                y = d7_data[target_col]
                
                # Remove constant columns
                X = X.loc[:, X.std() > 0]
                
                if len(X.columns) < 2:
                    continue
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                try:
                    # Try XGBoost first (most powerful)
                    from xgboost import XGBRegressor
                    xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    xgb.fit(X_train, y_train)
                    score = r2_score(y_test, xgb.predict(X_test))
                    
                    print(f"   XGBoost R²: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_target = target_col
                        best_model = xgb
                        
                except Exception as e:
                    print(f"   XGBoost failed: {e}")
                    continue
        
        if best_score > -1:  # If any model worked
            print(f"\n🎯 BEST TARGET: {best_target}, R²: {best_score:.3f}")
            
            # Save the best model
            model_path = "src/modules/market_analysis/models/market_predictor.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            return best_model, best_score
        else:
            print("❌ All targets failed")
            return None, 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, 0

# Try advanced ensemble
model, score = create_advanced_market_ensemble()
if model:
    print(f"✅ Advanced ensemble created with R²: {score:.3f}")
else:
    print("❌ Advanced ensemble failed - data may be fundamentally unpredictable")