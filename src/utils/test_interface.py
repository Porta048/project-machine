#!/usr/bin/env python3
"""
Simple test script for the web interface
"""

import requests
import time
import os

def test_web_interface():
    """Test if the web interface is running"""
    print("Testing web interface...")
    
    try:
        # Test if server is running
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code == 200:
            print("✓ Web interface is running successfully!")
            print("  - URL: http://localhost:5000")
            print("  - Status: Online")
            return True
        else:
            print(f"❌ Server responded with status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to web interface")
        print("  - Make sure the server is running: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error testing interface: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'config.py',
        'templates/index.html',
        'manutenzione_predittiva.py'
    ]
    
    required_dirs = [
        'models',
        'uploads'
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_good = False
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            all_good = False
    
    return all_good

def test_models():
    """Test if models exist"""
    print("\nTesting models...")
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("❌ Models directory not found")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    if model_files:
        print(f"✓ Found {len(model_files)} model(s):")
        for model in model_files:
            print(f"  - {model}")
        return True
    else:
        print("❌ No model files found")
        print("  - Train models first: python manutenzione_predittiva.py")
        return False

def main():
    """Run all tests"""
    print("Engine Predictive Maintenance - Interface Test")
    print("=" * 50)
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test models
    models_ok = test_models()
    
    # Test web interface
    print("\n" + "=" * 50)
    web_ok = test_web_interface()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if files_ok and models_ok and web_ok:
        print("🎉 All tests passed! Interface is ready to use.")
        print("\nNext steps:")
        print("1. Open browser to: http://localhost:5000")
        print("2. Upload an engine image")
        print("3. View analysis results")
    else:
        print("❌ Some tests failed.")
        print("\nSolutions:")
        if not files_ok:
            print("- Check file structure and missing files")
        if not models_ok:
            print("- Train models: python manutenzione_predittiva.py")
        if not web_ok:
            print("- Start server: python app.py")

if __name__ == "__main__":
    main() 