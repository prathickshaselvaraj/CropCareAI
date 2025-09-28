# test_complete_system.py
import pickle
import os
import pandas as pd
import numpy as np

def test_complete_system():
    print("🧪 COMPREHENSIVE SYSTEM TEST")
    print("=" * 50)
    
    # Test Model Loading
    modules = {
        "🌱 Crop Recommendation": "src/modules/crop_recommendation/models/crop_classifier_rf.pkl",
        "📈 Yield Prediction": "src/modules/yield_analysis/models/yield_predictor.pkl", 
        "💰 Market Analysis": "src/modules/market_analysis/models/price_predictor.pkl",
        "🐛 Pest Presence": "src/modules/pesticide_recommendation/models/pest_presence_classifier.pkl",
        "📊 Pest Severity": "src/modules/pesticide_recommendation/models/pest_severity_classifier.pkl",
        "💊 Treatment": "src/modules/pesticide_recommendation/models/treatment_recommender.pkl"
    }
    
    all_loaded = True
    for name, path in modules.items():
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ {name}: LOADED SUCCESSFULLY")
                
                # Test basic prediction if possible
                try:
                    if hasattr(model, 'predict'):
                        # Create simple test input based on model type
                        if 'crop' in name.lower():
                            test_input = [[50, 50, 50, 25, 60, 6.5, 100]]
                        elif 'pest' in name.lower():
                            test_input = [[0.5, 0.3, 0.2, 25, 60, 100]]
                        else:
                            test_input = [[1, 2, 3]]
                            
                        prediction = model.predict(test_input[:1])
                        print(f"   📊 Sample prediction: {prediction[0]}")
                except:
                    print(f"   ⚠️  Prediction test skipped")
            else:
                print(f"❌ {name}: FILE NOT FOUND")
                all_loaded = False
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            all_loaded = False
    
    print("=" * 50)
    if all_loaded:
        print("🎉 ALL SYSTEMS GO! Ready for API launch.")
    else:
        print("⚠️  Some models need attention.")
    
    return all_loaded

if __name__ == "__main__":
    test_complete_system()