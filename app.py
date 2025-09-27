"""Main CropCareAI Application"""
from flask import Flask, jsonify
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"🔧 Setting up paths...")
print(f"📁 Project root: {project_root}")

# Try to import the path fixer
try:
    from src.utils.path_fixer import setup_paths
    project_root, src_path = setup_paths()
    print("✅ Path fixer loaded successfully")
except ImportError as e:
    print(f"⚠️ Path fixer not available: {e}")

# Now try to import the routes
try:
    from src.api.routes.crop_routes import crop_bp
    crop_module_available = True
    print("✅ Crop routes imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import crop routes: {e}")
    crop_module_available = False

app = Flask(__name__)

# Register blueprints if available
if crop_module_available:
    app.register_blueprint(crop_bp, url_prefix='/api/crop')
    print("✅ Crop blueprint registered!")

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': '🌱 CropCareAI - Modular Structure',
        'version': '1.0.0',
        'status': 'running',
        'crop_module_available': crop_module_available,
        'endpoints': [
            'GET /',
            'GET /health',
            'POST /api/crop/recommend',
            'POST /api/crop/train',
            'GET /api/crop/info',
            'GET /api/crop/health'
        ] if crop_module_available else []
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'crop_module_available': crop_module_available,
        'timestamp': '2024-01-01T00:00:00Z'
    })

if __name__ == '__main__':
    print("\n🌱 CropCareAI Server Starting...")
    print("📊 Module Status:")
    print(f"   • Crop Recommendation: {'✅ Available' if crop_module_available else '❌ Not available'}")
    
    if crop_module_available:
        print("\n🚀 Available Endpoints:")
        print("   • GET  / - API information")
        print("   • GET  /health - Health check")
        print("   • POST /api/crop/recommend - Get crop recommendation")
        print("   • POST /api/crop/train - Train model")
        print("   • GET  /api/crop/info - Module information")
        print("   • GET  /api/crop/health - Crop module health")
    
    print(f"\n📍 Running on: http://localhost:5000")
    print("💡 Note: The model needs to be trained first using POST /api/crop/train")
    app.run(host='0.0.0.0', port=5000, debug=True)