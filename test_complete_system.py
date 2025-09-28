# test_all_5_modules.py
import pickle
import os
import pandas as pd
import numpy as np

def test_all_5_modules():
    print("🧪 COMPREHENSIVE 5-MODULE SYSTEM TEST")
    print("=" * 60)
    
    # Test ALL 5 MODULES with their actual models
    modules = {
        # Module 1: Crop Recommendation
        "🌱 Crop Recommendation": {
            "path": "src/modules/crop_recommendation/models/crop_classifier_rf.pkl",
            "type": "crop"
        },
        
        # Module 2: Yield Analysis  
        "📈 Yield Prediction": {
            "path": "src/modules/yield_analysis/models/yield_predictor.pkl",
            "type": "yield"
        },
        
        # Module 3: Market Analysis
        "💰 Market Analysis": {
            "path": "src/modules/market_analysis/models/price_predictor.pkl", 
            "type": "market"
        },
        
        # Module 4: Pesticide Recommendation (3 sub-models)
        "🐛 Pest Presence Detection": {
            "path": "src/modules/pesticide_recommendation/models/pest_presence_classifier.pkl",
            "type": "pest"
        },
        "📊 Pest Severity Classification": {
            "path": "src/modules/pesticide_recommendation/models/pest_severity_classifier.pkl", 
            "type": "pest"
        },
        "💊 Treatment Recommendation": {
            "path": "src/modules/pesticide_recommendation/models/treatment_recommender.pkl",
            "type": "pest"
        },
        
        # Module 5: Disease Detection
        "🦠 Disease Detection Metadata": {
            "path": "src/modules/disease_detection/models/disease_metadata.pkl",
            "type": "disease"
        },
        "🖼️ Image Preprocessor": {
            "path": "src/modules/disease_detection/models/image_preprocessor.pkl", 
            "type": "disease"
        }
    }
    
    module_status = {
        "Crop Recommendation": False,
        "Yield Analysis": False, 
        "Market Analysis": False,
        "Pesticide Recommendation": False,
        "Disease Detection": False
    }
    
    for name, config in modules.items():
        try:
            if os.path.exists(config["path"]):
                with open(config["path"], 'rb') as f:
                    model = pickle.load(f)
                
                # Map to main modules
                if "crop" in config["type"]:
                    module_status["Crop Recommendation"] = True
                elif "yield" in config["type"]:
                    module_status["Yield Analysis"] = True  
                elif "market" in config["type"]:
                    module_status["Market Analysis"] = True
                elif "pest" in config["type"]:
                    module_status["Pesticide Recommendation"] = True
                elif "disease" in config["type"]:
                    module_status["Disease Detection"] = True
                
                print(f"✅ {name}: LOADED SUCCESSFULLY")
                
            else:
                print(f"❌ {name}: FILE NOT FOUND at {config['path']}")
                
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
    
    print("=" * 60)
    print("📊 5-MODULE STATUS SUMMARY:")
    for module, status in module_status.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {module}: {'READY' if status else 'MISSING'}")
    
    ready_modules = sum(module_status.values())
    print(f"\n🎯 TOTAL READY: {ready_modules}/5 MODULES")
    
    if ready_modules == 5:
        print("🎉 ALL 5 MODULES ARE READY FOR DEPLOYMENT!")
    else:
        print(f"⚠️  {5-ready_modules} MODULES NEED ATTENTION")
    
    return module_status

if __name__ == "__main__":
    test_all_5_modules()