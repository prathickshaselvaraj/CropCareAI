# test_model_bias.py
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('src/modules/crop_recommendation/models/final_crop_model.pkl')

print("🎯 TESTING MODEL WITH DIFFERENT NPK VALUES:")

# Test different NPK combinations
test_cases = [
    ("Very High N", [90, 20, 20]),
    ("Very High P", [20, 90, 20]), 
    ("Very High K", [20, 20, 90]),
    ("Balanced High", [80, 80, 80]),
    ("Balanced Low", [30, 30, 30]),
    ("Extreme N", [95, 10, 10]),
    ("Extreme P", [10, 95, 10]),
    ("Extreme K", [10, 10, 95])
]

for name, npk in test_cases:
    # Create a base feature set with the NPK values in correct positions
    # Based on feature names: [Soilcolor, Ph, K, P, N, Zn, S, ...]
    features = [2, 6.5, npk[2], npk[1], npk[0], 50, 50]  # First 7 known features
    # Add remaining 23 features with reasonable defaults
    features.extend([25, 25, 25, 15, 15, 15, 75, 80] + [25] * 15)  # Total 30 features
    
    try:
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        confidence = np.max(probabilities)
        
        print(f"\n{name} (N:{npk[0]}, P:{npk[1]}, K:{npk[2]}):")
        print(f"  → {prediction} (Confidence: {confidence:.4f})")
        
        # Show if Red Pepper is always top
        top_5 = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:5]
        for i, (crop, prob) in enumerate(top_5):
            marker = " 🎯" if crop == "Red Pepper" else ""
            print(f"    {i+1}. {crop}: {prob:.4f}{marker}")
            
    except Exception as e:
        print(f"❌ {name} failed: {e}")