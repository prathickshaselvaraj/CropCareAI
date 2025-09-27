# src/models/pesticide_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_pesticide_data():
    """Analyze the pesticide data to find the correct approach"""
    print("🔍 Analyzing Pesticide Data Structure...")
    
    project_root = Path(__file__).parent.parent.parent
    processed_path = project_root / "data/processed/"
    
    # Load both pesticide datasets
    d5_data = pd.read_csv(processed_path / "D5_processed.csv")
    d6_data = pd.read_csv(processed_path / "D6_processed.csv")
    
    print("\n📊 D5 Dataset (Indoor Plants) - Shape:", d5_data.shape)
    print("Columns:", d5_data.columns.tolist())
    print("\nSample data:")
    print(d5_data.head(3))
    
    print("\n📊 D6 Dataset (Farmer Advisor) - Shape:", d6_data.shape)
    print("Columns:", d6_data.columns.tolist())
    print("\nSample data:")
    print(d6_data.head(3))
    
    # Analyze potential target columns
    print("\n🎯 POTENTIAL TARGET COLUMNS ANALYSIS:")
    
    # D5 Analysis
    print("\nD5 - Potential targets:")
    for col in d5_data.columns:
        if any(keyword in col.lower() for keyword in ['pest', 'disease', 'treatment', 'insect', 'fungus']):
            unique_vals = d5_data[col].nunique()
            print(f"  {col}: {unique_vals} unique values")
            if unique_vals <= 10:  # Show value distribution for small sets
                print(f"    Values: {d5_data[col].value_counts().to_dict()}")
    
    # D6 Analysis  
    print("\nD6 - Potential targets:")
    for col in d6_data.columns:
        if any(keyword in col.lower() for keyword in ['pest', 'disease', 'treatment', 'insect', 'fungus']):
            unique_vals = d6_data[col].nunique()
            print(f"  {col}: {unique_vals} unique values")
            if unique_vals <= 10:
                print(f"    Values: {d6_data[col].value_counts().to_dict()}")
    
    # Check for binary classification targets
    print("\n🔍 Binary Classification Candidates:")
    for df_name, df in [('D5', d5_data), ('D6', d6_data)]:
        for col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                print(f"  {df_name}.{col}: {df[col].value_counts().to_dict()}")

if __name__ == "__main__":
    analyze_pesticide_data()