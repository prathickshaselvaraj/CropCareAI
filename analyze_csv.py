import pandas as pd
import numpy as np
import os

print('🌱 ANALYZING EXISTING CSV DATASETS')
print('='*60)

# Your existing CSV files
csv_files = {
    'D1': 'D1_crop_recommendation.csv',
    'D3': 'D3_complete_dataset.csv', 
    'D4': 'D4_crop_recommendation.csv',
    'D5': 'D5_indoor_plants.csv',
    'D6': 'D6_farmer_advisor.csv',
    'D7': 'D7_market_researcher.csv'
}

def analyze_csv_dataset(dataset_key, filename):
    print(f'\\n🔍 ANALYZING: {dataset_key} - {filename}')
    print('-'*50)
    
    try:
        path = f'data/raw/{filename}'
        
        if os.path.exists(path):
            # Read CSV file
            df = pd.read_csv(path)
            print(f'✅ Loaded CSV: {path}')
            print(f'📊 Shape: {df.shape} (rows: {df.shape[0]}, cols: {df.shape[1]})')
            print(f'📋 Columns: {df.columns.tolist()}')
            print(f'🔢 Data types:')
            print(df.dtypes.value_counts())
            print(f'❓ Missing values: {df.isnull().sum().sum()}')
            print(f'🔄 Duplicates: {df.duplicated().sum()}')
            
            # Basic stats for numerical columns
            numerical = df.select_dtypes(include=[np.number])
            if not numerical.empty:
                print(f'📈 Numerical columns: {len(numerical.columns)}')
                print(f'   Sample ranges:')
                for col in numerical.columns[:3]:  # Show first 3 columns
                    if len(df[col].unique()) > 1:
                        print(f'     {col}: {df[col].min():.2f} to {df[col].max():.2f}')
            
            # Check for target column
            target_candidates = ['label', 'target', 'crop', 'disease', 'yield', 'production', 'price', 'Class', 'Type']
            target_col = None
            for candidate in target_candidates:
                if candidate in df.columns:
                    target_col = candidate
                    break
            
            if target_col:
                print(f'🎯 Target column found: {target_col}')
                print(f'   Unique values: {df[target_col].nunique()}')
                if df[target_col].nunique() < 20:
                    print(f'   Distribution:')
                    print(df[target_col].value_counts().head(10))
            else:
                print('🎯 No obvious target column found')
                # Show last column as potential target
                last_col = df.columns[-1]
                print(f'   Last column \"{last_col}\" has {df[last_col].nunique()} unique values')
            
            # Show sample data
            print(f'\\n📄 Sample data (first 2 rows):')
            print(df.head(2).to_string())
            
            return df
        else:
            print(f'❌ File not found: {path}')
            return None
    except Exception as e:
        print(f'❌ Error: {e}')
        return None

# Analyze all CSV datasets
all_data = {}
for dataset_key, filename in csv_files.items():
    df = analyze_csv_dataset(dataset_key, filename)
    all_data[dataset_key] = df

print('\\n' + '='*60)
print('📋 SUMMARY: All CSV datasets analyzed!')
print('\\\\n🎯 NEXT: Proper EDA and preprocessing for each dataset')
