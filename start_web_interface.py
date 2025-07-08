#!/usr/bin/env python3
"""
Web Interface Launcher for Engine Predictive Maintenance System
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'opencv-python', 'pillow', 'werkzeug'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_models():
    """Check if trained models exist"""
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found!")
        print("Please run the training script first:")
        print("python manutenzione_predittiva.py")
        return False
    
    model_files = list(models_dir.glob("model_*.h5"))
    if not model_files:
        print("❌ No trained models found!")
        print("Please run the training script first:")
        print("python manutenzione_predittiva.py")
        return False
    
    print("✓ Found trained models:")
    for model_file in model_files:
        print(f"   - {model_file.name}")
    
    return True

def main():
    print("🔧 Engine Predictive Maintenance - Web Interface")
    print("=" * 50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return
    
    # Check models
    print("\n2. Checking trained models...")
    if not check_models():
        return
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    print("✓ Uploads directory ready")
    
    # Start the web server
    print("\n3. Starting web server...")
    print("   Server will be available at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("\n" + "=" * 50)
    
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start Flask app
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 