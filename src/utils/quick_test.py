#!/usr/bin/env python3
"""
Quick test to verify model loading and basic functionality
"""

import os
import numpy as np

def test_model_loading():
    """Test if models can be loaded correctly"""
    print("Testing model loading...")
    
    try:
        from manutenzione_predittiva import load_models
        
        models = load_models()
        
        if models:
            print(f"✓ Successfully loaded {len(models)} models")
            for dataset in models.keys():
                print(f"  - {dataset}: {type(models[dataset]).__name__}")
            
            # Test prediction with sample data
            print("\nTesting prediction...")
            sample_data = np.random.rand(1, 50, 24)
            
            from manutenzione_predittiva import predict_rul
            
            for dataset in list(models.keys())[:1]:  # Test first model only
                rul = predict_rul(models, sample_data, dataset)
                print(f"✓ Prediction for {dataset}: {rul}")
                break
                
            return True
        else:
            print("❌ No models loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_image_analysis():
    """Test image analysis functionality"""
    print("\nTesting image analysis...")
    
    try:
        from app import analyze_engine_image
        
        # Create a test image
        import cv2
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        test_path = "test_image.jpg"
        
        cv2.imwrite(test_path, test_image)
        
        # Test analysis
        result = analyze_engine_image(test_path)
        
        # Clean up
        os.remove(test_path)
        
        if 'error' not in result:
            print("✓ Image analysis working")
            print(f"  - Image size: {result['image_size']}")
            print(f"  - Assessment: {result['visual_assessment']}")
            return True
        else:
            print(f"❌ Image analysis failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Image analysis test failed: {e}")
        return False

def main():
    """Run quick tests"""
    print("Engine Predictive Maintenance - Quick Test")
    print("=" * 40)
    
    # Test model loading
    models_ok = test_model_loading()
    
    # Test image analysis
    image_ok = test_image_analysis()
    
    # Summary
    print("\n" + "=" * 40)
    print("QUICK TEST SUMMARY")
    print("=" * 40)
    
    if models_ok and image_ok:
        print("🎉 All tests passed! System is ready.")
        print("\nYou can now:")
        print("1. Use the web interface at http://localhost:5000")
        print("2. Upload engine images for analysis")
        print("3. Get combined visual + sensor analysis")
    else:
        print("❌ Some tests failed.")
        if not models_ok:
            print("- Check model files in models/ directory")
        if not image_ok:
            print("- Check OpenCV installation")

if __name__ == "__main__":
    main() 