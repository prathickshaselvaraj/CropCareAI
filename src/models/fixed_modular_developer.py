# src/models/fixed_modular_developer.py
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

class FixedModularModelDeveloper:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.processed_path = self.project_root / "data/processed/"
        self.modules_path = self.project_root / "src/modules/"
        
    def load_processed_data(self):
        """Load processed datasets with memory optimization"""
        print("📊 Loading processed datasets...")
        datasets = {}
        
        try:
            datasets['D1'] = pd.read_csv(self.processed_path / "D1_processed.csv")
            datasets['D2_meta'] = pd.read_csv(self.processed_path / "D2_metadata.csv")
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
            unique_count = X[col].nunique()
            
            if unique_count > max_categories:
                print(f"🔧 Encoding '{col}' with label encoding ({unique_count} categories)")
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
            else:
                print(f"🔧 Encoding '{col}' with one-hot ({unique_count} categories)")
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded.drop(columns=[col], inplace=True)
        
        return X_encoded.select_dtypes(include=[np.number])

    def _safe_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Safe train-test split that handles class imbalance"""
        # Check if we can use stratification
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count >= 2:  # Can use stratification
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            print("⚠️  Class imbalance detected, using simple split")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def develop_crop_recommendation_models(self, d1_data, d4_data):
        """Develop crop recommendation models"""
        print("\n🌱 Developing Crop Recommendation Models...")
        
        module_path = self.modules_path / "crop_recommendation" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        models = {}
        
        # Model 1: Multi-class crop classifier
        print("🔧 Training Multi-Crop Classifier...")
        target_col = self._find_target_column(d1_data, ['label', 'crop', 'Crop_Type'])
        X = d1_data.drop(columns=[target_col])
        y = d1_data[target_col]
        
        print(f"🎯 Target: {target_col}, Classes: {len(y.unique())}")
        print(f"📊 Class distribution:\n{y.value_counts()}")
        
        X_encoded = self._encode_categorical_memory_safe(X)
        X_train, X_test, y_train, y_test = self._safe_train_test_split(X_encoded, y)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
        
        models['crop_classifier_rf'] = (rf_model, rf_accuracy)
        
        # Save model
        filename = module_path / "crop_classifier_rf.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"✅ Saved crop_classifier_rf with accuracy: {rf_accuracy:.4f}")
        
        return models

    def develop_yield_analysis_models(self, d3_data):
        """Memory-efficient yield analysis models"""
        print("\n📈 Developing Yield Analysis Models...")
        
        module_path = self.modules_path / "yield_analysis" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        models = {}
        
        # Use essential columns
        numeric_cols = d3_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'production_cleaned' in numeric_cols:
            target_col = 'production_cleaned'
            feature_cols = [col for col in numeric_cols if col != target_col][:5]  # Use 5 features
            essential_cols = [target_col] + feature_cols
        else:
            essential_cols = numeric_cols[:6]  # Use first 6 numeric columns
            target_col = essential_cols[0]
            feature_cols = essential_cols[1:]
        
        d3_reduced = d3_data[essential_cols].copy()
        print(f"🔧 Using {len(feature_cols)} features for yield prediction")
        
        X = d3_reduced[feature_cols]
        y = d3_reduced[target_col]
        
        # Sample if too large
        if len(X) > 50000:
            sample_size = min(50000, len(X))
            X_sampled = X.sample(n=sample_size, random_state=42)
            y_sampled = y.loc[X_sampled.index]
            print(f"🔧 Sampling {sample_size} records")
        else:
            X_sampled, y_sampled = X, y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sampled, y_sampled, test_size=0.2, random_state=42
        )
        
        rf_model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))
        
        models['yield_predictor'] = (rf_model, rf_rmse)
        
        filename = module_path / "yield_predictor.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"✅ Saved yield_predictor with RMSE: {rf_rmse:.2f}")
        
        return models

    def develop_pesticide_recommendation_models(self, d5_data, d6_data):
        """Fixed pesticide recommendation with imbalance handling"""
        print("\n🐛 Developing Pesticide Recommendation Models...")
        
        module_path = self.modules_path / "pesticide_recommendation" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        models = {}
        
        # Try D6 first (larger dataset)
        target_col = self._find_target_column(d6_data, ['Pest_Presence', 'pest_detected', 'Infestation_Level'])
        if target_col not in d6_data.columns:
            # Try D5 as fallback
            target_col = self._find_target_column(d5_data, ['Pest_Presence', 'pest_detected'])
            data_to_use = d5_data
            print("🔧 Using D5 dataset for pesticide recommendation")
        else:
            data_to_use = d6_data
            print("🔧 Using D6 dataset for pesticide recommendation")
        
        X = data_to_use.drop(columns=[target_col])
        y = data_to_use[target_col]
        
        print(f"🎯 Target: {target_col}, Classes: {len(y.unique())}")
        print(f"📊 Class distribution:\n{y.value_counts()}")
        
        # Handle extreme class imbalance
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            print("⚠️  Not enough classes for classification, skipping pesticide module")
            return models
        
        # Remove classes with only 1 sample
        y_counts = y.value_counts()
        y = y[y.isin(y_counts[y_counts > 1].index)]
        X_encoded = X_encoded.loc[y.index]

        
        X_encoded = self._encode_categorical_memory_safe(X)
        X_train, X_test, y_train, y_test = self._safe_train_test_split(X_encoded, y)
        
        pest_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        pest_model.fit(X_train, y_train)
        pest_accuracy = accuracy_score(y_test, pest_model.predict(X_test))
        
        models['pest_detector'] = (pest_model, pest_accuracy)
        
        filename = module_path / "pest_detector.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(pest_model, f)
        print(f"✅ Saved pest_detector with accuracy: {pest_accuracy:.4f}")
        
        return models

    def develop_market_analysis_models(self, d7_data):
        """Market analysis models"""
        print("\n💰 Developing Market Analysis Models...")
        
        module_path = self.modules_path / "market_analysis" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        models = {}
        
        # Find price column
        price_cols = [col for col in d7_data.columns if 'price' in col.lower() or 'value' in col.lower()]
        if price_cols:
            target_col = price_cols[0]
            X = d7_data.drop(columns=[target_col])
            y = d7_data[target_col]
            
            X_encoded = self._encode_categorical_memory_safe(X)
            X_encoded = X_encoded.select_dtypes(include=[np.number])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            
            price_model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
            price_model.fit(X_train, y_train)
            price_rmse = np.sqrt(mean_squared_error(y_test, price_model.predict(X_test)))
            
            models['price_predictor'] = (price_model, price_rmse)
            
            filename = module_path / "price_predictor.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(price_model, f)
            print(f"✅ Saved price_predictor with RMSE: {price_rmse:.2f}")
        
        return models

    def develop_disease_detection_models(self, d2_metadata):
        """Disease detection models"""
        print("\n🦠 Preparing Disease Detection Models...")
        
        module_path = self.modules_path / "disease_detection" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        disease_mapping = {}
        for _, row in d2_metadata.iterrows():
            disease_mapping[row['disease_type']] = {
                'image_count': row['image_count'],
                'label': row['label'],
                'severity': self._classify_disease_severity(row['disease_type'])
            }
        
        with open(module_path / "disease_metadata.pkl", 'wb') as f:
            pickle.dump(disease_mapping, f)
        
        print(f"✅ Disease metadata prepared: {len(disease_mapping)} diseases")
        return disease_mapping

    def _find_target_column(self, data, possible_names):
        """Helper to find target column"""
        for name in possible_names:
            if name in data.columns:
                return name
        return data.columns[-1] if len(data.columns) > 0 else None

    def _classify_disease_severity(self, disease_name):
        """Classify disease severity"""
        disease_lower = disease_name.lower()
        if 'healthy' in disease_lower:
            return 'none'
        elif 'blight' in disease_lower or 'rot' in disease_lower:
            return 'high'
        else:
            return 'medium'

    def develop_all_models_fixed(self):
        """Main pipeline with fixes for all issues"""
        print("🚀 Starting Fixed Model Development...")
        print("=" * 60)
        
        datasets = self.load_processed_data()
        if datasets is None:
            return False
        
        all_models = {}
        
        try:
            # Process modules with error handling
            modules = [
                ('crop_recommendation', lambda: self.develop_crop_recommendation_models(datasets['D1'], datasets['D4'])),
                ('disease_detection', lambda: self.develop_disease_detection_models(datasets['D2_meta'])),
                ('yield_analysis', lambda: self.develop_yield_analysis_models(datasets['D3'])),
                ('pesticide_recommendation', lambda: self.develop_pesticide_recommendation_models(datasets['D5'], datasets['D6'])),
                ('market_analysis', lambda: self.develop_market_analysis_models(datasets['D7']))
            ]
            
            for module_name, module_func in modules:
                print("\n" + "="*50)
                print(f"📊 Processing {module_name}...")
                try:
                    all_models[module_name] = module_func()
                    print(f"✅ {module_name} completed successfully")
                except Exception as e:
                    print(f"⚠️  {module_name} skipped due to error: {e}")
                    all_models[module_name] = {}
            
            print("\n🎯 MODEL DEVELOPMENT SUMMARY:")
            successful_modules = [name for name, models in all_models.items() if models]
            print(f"✅ Successful: {len(successful_modules)}/{len(modules)} modules")
            
            return all_models
            
        except Exception as e:
            print(f"❌ Error in model development: {e}")
            return False

# Execute the fixed pipeline
if __name__ == "__main__":
    developer = FixedModularModelDeveloper()
    all_modules = developer.develop_all_models_fixed()
    
    if all_modules:
        print("\n🎉 MODEL DEVELOPMENT COMPLETED!")
        print("\n📊 FINAL RESULTS:")
        for module_name, models in all_modules.items():
            if models:
                print(f"  ✅ {module_name}: {len(models)} models trained")
            else:
                print(f"  ⚠️  {module_name}: No models (skipped)")
    else:
        print("\n❌ Model development failed.")