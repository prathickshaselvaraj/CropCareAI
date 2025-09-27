# src/models/pesticide_fix.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def fix_pesticide_module():
    """Fix the pesticide recommendation module with regression"""
    print("🔧 Fixing Pesticide Recommendation Module...")
    
    project_root = Path(__file__).parent.parent.parent
    processed_path = project_root / "data/processed/"
    module_path = project_root / "src/modules/pesticide_recommendation/models/"
    module_path.mkdir(parents=True, exist_ok=True)
    
    # Load D6 data
    d6_data = pd.read_csv(processed_path / "D6_processed.csv")
    
    # Use Sustainability_Score as regression target
    target_col = "Sustainability_Score"
    X = d6_data.drop(columns=[target_col])
    y = d6_data[target_col]
    
    print(f"🎯 Regression target: {target_col}")
    print(f"📊 Value range: {y.min():.2f} to {y.max():.2f}")
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X[col].nunique() <= 10:  # One-hot encode low cardinality
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            X.drop(columns=[col], inplace=True)
        else:  # Label encode high cardinality
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train regression model
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = model.score(X_test, y_test)
    
    print(f"✅ Sustainability predictor trained!")
    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📊 R² Score: {r2:.4f}")
    
    # Save model
    filename = module_path / "sustainability_predictor.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Model saved to: {filename}")
    
    return model, rmse, r2

if __name__ == "__main__":
    fix_pesticide_module()