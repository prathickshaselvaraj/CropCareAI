# src/models/memory_efficient_developer.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MemoryEfficientModelDeveloper:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.processed_path = self.project_root / "data/processed/"
        self.modules_path = self.project_root / "src/modules/"
        
    def load_processed_data(self):
        """Load processed datasets with memory optimization"""
        print("📊 Loading processed datasets...")
        datasets = {}
        
        try:
            # Load with optimized data types
            datasets['D1'] = pd.read_csv(self.processed_path / "D1_processed.csv")
            datasets['D2_meta'] = pd.read_csv(self.processed_path / "D2_metadata.csv")
            
            # Load D3 with memory optimization
            datasets['D3'] = pd.read_csv(self.processed_path / "D3_processed.csv", 
                                       dtype={'production_cleaned': 'float32'})
            
            datasets['D4'] = pd.read_csv(self.processed_path / "D4_processed.csv")
            datasets['D5'] = pd.read_csv(self.processed_path / "D5_processed.csv")
            datasets['D6'] = pd.read_csv(self.processed_path / "D6_processed.csv")
            datasets['D7'] = pd.read_csv(self.processed_path / "D7_processed.csv")
            
            print("✅ All processed datasets loaded successfully")
            return datasets
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return None

    def _encode_categorical_memory_safe(self, X, max_categories=50):
        """Memory-safe categorical encoding"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        for col in categorical_cols:
            # For high-cardinality columns, use label encoding instead of one-hot
            unique_count = X[col].nunique()
            
            if unique_count > max_categories:
                print(f"🔧 Encoding '{col}' with label encoding ({unique_count} categories)")
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                # Save the encoder for later use
                X_encoded[f'{col}_encoder'] = le
            else:
                # Use one-hot encoding for low-cardinality columns
                print(f"🔧 Encoding '{col}' with one-hot ({unique_count} categories)")
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded.drop(columns=[col], inplace=True)
        
        # Keep only numeric columns for modeling
        X_encoded = X_encoded.select_dtypes(include=[np.number])
        return X_encoded

    def develop_crop_recommendation_models(self, d1_data, d4_data):
        """Develop crop recommendation models (memory efficient)"""
        print("\n🌱 Developing Crop Recommendation Models...")
        
        module_path = self.modules_path / "crop_recommendation" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        models = {}
        
        # Model 1: Multi-class crop classifier
        print("🔧 Training Multi-Crop Classifier...")
        target_col = self._find_target_column(d1_data, ['label', 'crop', 'Crop_Type'])
        X = d1_data.drop(columns=[target_col])
        y = d1_data[target_col]
        
        # Memory-safe encoding
        X_encoded = self._encode_categorical_memory_safe(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)  # Reduced estimators
        rf_model.fit(X_train, y_train)
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
        
        models['crop_classifier_rf'] = (rf_model, rf_accuracy)
        
        # Save models
        for model_name, model_data in models.items():
            filename = module_path / f"{model_name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model_data[0], f)
            print(f"✅ Saved {model_name} to {filename}")
        
        return models

    def develop_yield_analysis_models(self, d3_data):
        """Memory-efficient yield analysis models"""
        print("\n📈 Developing Yield Analysis Models...")
        
        module_path = self.modules_path / "yield_analysis" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        models = {}
        
        # Use only essential columns to reduce memory
        essential_cols = ['production_cleaned']  # Start with target
        
        # Add a few key features instead of all columns
        numeric_cols = d3_data.select_dtypes(include=[np.number]).columns.tolist()
        essential_cols.extend(numeric_cols[:5])  # Use first 5 numeric columns
        
        # Filter data
        d3_reduced = d3_data[essential_cols].copy()
        
        print(f"🔧 Using reduced feature set: {len(essential_cols)-1} features")
        
        # Model 1: Yield prediction regression
        target_col = 'production_cleaned'
        X = d3_reduced.drop(columns=[target_col])
        y = d3_reduced[target_col]
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Use smaller sample for training to save memory
        if len(X) > 50000:
            sample_size = 50000
            X_sampled = X.sample(n=sample_size, random_state=42)
            y_sampled = y.loc[X_sampled.index]
            print(f"🔧 Sampling {sample_size} records for memory efficiency")
        else:
            X_sampled = X
            y_sampled = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sampled, y_sampled, test_size=0.2, random_state=42
        )
        
        # Train smaller model
        rf_model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)  # Reduced
        rf_model.fit(X_train, y_train)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))
        
        models['yield_predictor'] = (rf_model, rf_rmse)
        
        # Save models
        for model_name, model_data in models.items():
            filename = module_path / f"{model_name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model_data[0], f)
            print(f"✅ Saved {model_name} to {filename}")
        
        return models

    def develop_pesticide_recommendation_models(self, d5_data, d6_data):
        """Memory-efficient pesticide recommendation"""
        print("\n🐛 Developing Pesticide Recommendation Models...")
        
        module_path = self.modules_path / "pesticide_recommendation" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        models = {}
        
        # Use D6 (smaller than D3)
        target_col = self._find_target_column(d6_data, ['Pest_Presence', 'pest_detected'])
        X = d6_data.drop(columns=[target_col])
        y = d6_data[target_col]
        
        # Memory-safe encoding
        X_encoded = self._encode_categorical_memory_safe(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        pest_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        pest_model.fit(X_train, y_train)
        pest_accuracy = accuracy_score(y_test, pest_model.predict(X_test))
        
        models['pest_detector'] = (pest_model, pest_accuracy)
        
        # Save models
        for model_name, model_data in models.items():
            filename = module_path / f"{model_name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model_data[0], f)
            print(f"✅ Saved {model_name} to {filename}")
        
        return models

    def develop_market_analysis_models(self, d7_data):
        """Memory-efficient market analysis"""
        print("\n💰 Developing Market Analysis Models...")
        
        module_path = self.modules_path / "market_analysis" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        models = {}
        
        # Price prediction
        price_cols = [col for col in d7_data.columns if 'price' in col.lower()]
        if price_cols:
            target_col = price_cols[0]
            X = d7_data.drop(columns=[target_col])
            y = d7_data[target_col]
            
            # Memory-safe encoding
            X_encoded = self._encode_categorical_memory_safe(X)
            X_encoded = X_encoded.select_dtypes(include=[np.number])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            
            price_model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
            price_model.fit(X_train, y_train)
            price_rmse = np.sqrt(mean_squared_error(y_test, price_model.predict(X_test)))
            
            models['price_predictor'] = (price_model, price_rmse)
        
        # Save models
        for model_name, model_data in models.items():
            filename = module_path / f"{model_name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model_data[0], f)
            print(f"✅ Saved {model_name} to {filename}")
        
        return models

    def develop_disease_detection_models(self, d2_metadata):
        """Disease detection models (unchanged - memory safe)"""
        print("\n🦠 Preparing Disease Detection Models...")
        
        module_path = self.modules_path / "disease_detection" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        # Create disease classification metadata
        disease_mapping = {}
        for _, row in d2_metadata.iterrows():
            disease_mapping[row['disease_type']] = {
                'image_count': row['image_count'],
                'label': row['label'],
                'severity': self._classify_disease_severity(row['disease_type'])
            }
        
        # Save disease metadata
        with open(module_path / "disease_metadata.pkl", 'wb') as f:
            pickle.dump(disease_mapping, f)
        
        print(f"✅ Disease metadata prepared: {len(disease_mapping)} diseases")
        return disease_mapping

    def _find_target_column(self, data, possible_names):
        """Helper to find target column"""
        for name in possible_names:
            if name in data.columns:
                return name
        return data.columns[-1]

    def _classify_disease_severity(self, disease_name):
        """Classify disease severity"""
        disease_lower = disease_name.lower()
        if 'healthy' in disease_lower:
            return 'none'
        elif 'blight' in disease_lower or 'rot' in disease_lower:
            return 'high'
        else:
            return 'medium'

    def develop_all_models_memory_safe(self):
        """Main pipeline with memory optimization"""
        print("🚀 Starting Memory-Efficient Model Development...")
        print("=" * 60)
        
        # Load processed data
        datasets = self.load_processed_data()
        if datasets is None:
            return False
        
        all_models = {}
        
        try:
            # Process modules one by one with memory cleanup
            print("\n" + "="*50)
            all_models['crop_recommendation'] = self.develop_crop_recommendation_models(
                datasets['D1'], datasets['D4']
            )
            
            # Clear memory
            del datasets['D1'], datasets['D4']
            
            print("\n" + "="*50)
            all_models['disease_detection'] = self.develop_disease_detection_models(
                datasets['D2_meta']
            )
            
            del datasets['D2_meta']
            
            print("\n" + "="*50)
            all_models['yield_analysis'] = self.develop_yield_analysis_models(datasets['D3'])
            
            del datasets['D3']
            
            print("\n" + "="*50)
            all_models['pesticide_recommendation'] = self.develop_pesticide_recommendation_models(
                datasets['D5'], datasets['D6']
            )
            
            del datasets['D5'], datasets['D6']
            
            print("\n" + "="*50)
            all_models['market_analysis'] = self.develop_market_analysis_models(datasets['D7'])
            
            print("\n🎉 MEMORY-EFFICIENT MODEL DEVELOPMENT COMPLETED!")
            return all_models
            
        except Exception as e:
            print(f"❌ Error in model development: {e}")
            import traceback
            traceback.print_exc()
            return False

# Execute the memory-efficient pipeline
if __name__ == "__main__":
    developer = MemoryEfficientModelDeveloper()
    all_modules = developer.develop_all_models_memory_safe()
    
    if all_modules:
        print("\n📊 MODULES SUCCESSFULLY TRAINED:")
        for module_name, models in all_modules.items():
            print(f"  📁 {module_name}: {len(models)} models")
    else:
        print("\n❌ Model development failed.")