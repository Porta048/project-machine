#!/usr/bin/env python3
"""
Script di Avvio Rapido - Interfaccia Web ML Core
Installa dipendenze e avvia l'interfaccia per test modelli
"""

import os
import sys
import subprocess
import time

def check_and_install_requirements():
    """Controlla e installa le dipendenze necessarie"""
    print("Checking dependencies...")
    
    try:
        import flask
        import tensorflow
        import cv2
        import PIL
        print("All dependencies are already installed")
        return True
    except ImportError as e:
        print(f"Installing missing dependencies: {e}")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements_ml_core.txt"
            ])
            print("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Error installing dependencies")
            return False

def check_models():
    """Controlla la presenza di modelli pre-addestrati"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("Warning: Models directory not found")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    if not model_files:
        print("Warning: No pre-trained models found")
        print("Tip: Run 'python train_models.py --benchmark' first to train models")
        return False
    
    print(f"Found {len(model_files)} pre-trained models")
    for model in model_files:
        print(f"   {model}")
    return True

def create_uploads_directory():
    """Crea directory per upload se non esiste"""
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"Created directory: {uploads_dir}")

def start_interface():
    """Avvia l'interfaccia web"""
    print("\nStarting web interface...")
    print("Interface will be available at: http://localhost:5000")
    print("Access from other devices: http://[COMPUTER_IP]:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Importa e avvia l'app
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

def main():
    """Funzione principale"""
    print("="*60)
    print("ML CORE - PREDICTIVE MAINTENANCE INTERFACE")
    print("="*60)
    print("Upload engine images and get RUL predictions")
    print("Interactive 3D visualization of results")
    print("="*60)
    
    # Controlla dipendenze
    if not check_and_install_requirements():
        print("Unable to install dependencies")
        return
    
    # Controlla modelli
    if not check_models():
        print("\nSUGGESTION: Train models first with:")
        print("   python train_models.py --benchmark")
        print("\nDo you want to continue anyway? (y/n): ", end="")
        
        try:
            choice = input().lower().strip()
            if choice not in ['y', 'yes', 's', 'si']:
                print("Exiting...")
                return
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Crea directory necessarie
    create_uploads_directory()
    
    # Avvia interfaccia
    start_interface()

if __name__ == "__main__":
    main()