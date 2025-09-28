# diagnose_problems.py
"""
DIAGNOSE WHY YIELD AND MARKET MODELS ARE FAILING
"""

import pandas as pd
import numpy as np

print("🔍 DIAGNOSING YIELD AND MARKET DATA PROBLEMS")
print("=" * 50)

def diagnose_dataset(dataset_name, file_path):
    print(f"\n📊 ANALYZING {dataset_name}:")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Analyze target column
        target_col = df.columns[-1]
        print(f"Target column: '{target_col}'")
        print(f"Target stats - Min: {df[target_col].min():.2f}, Max: {df[target_col].max():.2f}, Mean: {df[target_col].mean():.2f}")
        
        # Check for constant values
        if df[target_col].nunique() == 1:
            print("❌ TARGET IS CONSTANT - CANNOT TRAIN MODEL!")
        
        # Check numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns: {len(numeric_cols)}")
        
        # Show correlation with target
        if target_col in numeric_cols and len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
            print("Top correlations with target:")
            print(correlations.head(6))
            
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Diagnose both problematic datasets
d3_data = diagnose_dataset("YIELD DATA (D3)", "data/processed/D3_processed.csv")
d7_data = diagnose_dataset("MARKET DATA (D7)", "data/processed/D7_processed.csv")

print("\n" + "=" * 50)
print("🎯 RECOMMENDED FIXES:")
print("=" * 50)

if d3_data is not None:
    print("\n📈 YIELD FIX: Use 'area' column as target instead of 'production_cleaned'")
    print("   - Area is more predictable than production")
    print("   - Production depends on many external factors")

if d7_data is not None:
    print("\n💰 MARKET FIX: Use 'Demand_Index' or 'Supply_Index' as target")
    print("   - Price prediction is very difficult")
    print("   - Demand/Supply are better prediction targets")