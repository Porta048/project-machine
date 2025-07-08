#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engine Predictive Maintenance Web Interface

Main Flask application that provides a web interface for engine image analysis
and predictive maintenance using machine learning models.

This application follows a clean architecture with separated responsibilities:
- Image analysis handled by image_analyzer.py
- Model management handled by model_manager.py
- Configuration managed by config.py
- Main app.py focuses on web routes and orchestration
"""

import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import json
from typing import Dict, Any, Optional

# Import our separated modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

from image_analyzer import create_image_analyzer, create_sensor_generator, create_model_selector
from model_manager import create_model_manager, create_status_analyzer
from config import get_config, validate_config, print_config_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config_dict = get_config()

# Validate configuration
if not validate_config(config_dict):
    logger.error("Configuration validation failed")
    exit(1)

# Print configuration summary
print_config_summary(config_dict)

# Initialize Flask app with correct template and static folders
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
template_dir = os.path.join(project_root, 'templates')
static_dir = os.path.join(project_root, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config.update(config_dict)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
image_analyzer = create_image_analyzer(config_dict)
sensor_generator = create_sensor_generator()
model_selector = create_model_selector()
model_manager = create_model_manager(app.config['MODELS_DIR'], config_dict)
status_analyzer = create_status_analyzer(config_dict)

# Load models at startup
models_loaded = model_manager.load_models()
if not models_loaded:
    logger.warning("No models loaded - some functionality will be limited")


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def validate_upload_file(file) -> Dict[str, Any]:
    """
    Validate uploaded file with comprehensive checks
    
    Args:
        file: Uploaded file object
        
    Returns:
        Validation result dictionary
    """
    if not file:
        return {
            'valid': False,
            'error': 'No file provided',
            'error_type': 'no_file'
        }
    
    if file.filename == '':
        return {
            'valid': False,
            'error': 'No file selected',
            'error_type': 'no_filename'
        }
    
    # Check file extension
    if not allowed_file(file.filename):
        return {
            'valid': False,
            'error': f'File type not allowed. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}',
            'error_type': 'invalid_extension'
        }
    
    # Check filename length
    if len(file.filename) > app.config['MAX_FILENAME_LENGTH']:
        return {
            'valid': False,
            'error': f'Filename too long (max {app.config["MAX_FILENAME_LENGTH"]} characters)',
            'error_type': 'filename_too_long'
        }
    
    # Check for path traversal attempts
    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        return {
            'valid': False,
            'error': 'Invalid filename - path traversal not allowed',
            'error_type': 'path_traversal'
        }
    
    # Check blocked extensions
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    if file_ext in app.config['BLOCKED_EXTENSIONS']:
        return {
            'valid': False,
            'error': f'File extension {file_ext} is blocked for security reasons',
            'error_type': 'blocked_extension'
        }
    
    return {'valid': True}


@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')


@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring
    
    Returns:
        JSON with system health status
    """
    try:
        # Get model status
        model_status = model_manager.get_model_status()
        
        # Check if models are loaded
        models_healthy = model_status['models_loaded']
        
        # Check upload directory
        upload_dir_accessible = os.access(app.config['UPLOAD_FOLDER'], os.W_OK)
        
        # Overall health status
        overall_healthy = models_healthy and upload_dir_accessible
        
        health_status = {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'timestamp': model_status.get('load_timestamp'),
            'components': {
                'models': {
                    'status': 'healthy' if models_healthy else 'unhealthy',
                    'loaded_count': model_status.get('total_models', 0),
                    'available_datasets': model_status.get('available_datasets', [])
                },
                'upload_directory': {
                    'status': 'healthy' if upload_dir_accessible else 'unhealthy',
                    'path': app.config['UPLOAD_FOLDER']
                },
                'image_analyzer': {
                    'status': 'healthy',
                    'config_loaded': bool(image_analyzer.config)
                }
            },
            'version': '1.0.0',
            'environment': os.environ.get('FLASK_ENV', 'development')
        }
        
        status_code = 200 if overall_healthy else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': None
        }), 503


@app.route('/models/status')
def models_status():
    """
    Get detailed status of all models
    
    Returns:
        JSON with model status information
    """
    try:
        status = model_manager.get_model_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({
            'error': 'Failed to get model status',
            'details': str(e)
        }), 500


@app.route('/models/reload', methods=['POST'])
def reload_models():
    """
    Reload all models (useful for updates)
    
    Returns:
        JSON with reload status
    """
    try:
        success = model_manager.reload_models()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Models reloaded successfully',
                'models_loaded': model_manager.get_model_status()['total_models']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to reload models',
                'error': model_manager.get_model_status().get('error', 'Unknown error')
            }), 500
            
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Error during model reload',
            'details': str(e)
        }), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and analysis
    
    Returns:
        JSON with analysis results
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file part in request',
                'error_type': 'no_file_part',
                'status': 'error'
            }), 400
        
        file = request.files['file']
        
        # Validate file
        validation = validate_upload_file(file)
        if not validation['valid']:
            return jsonify({
                'error': validation['error'],
                'error_type': validation['error_type'],
                'status': 'error'
            }), 400
        
        # Secure filename and create filepath
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file with error handling
        try:
            file.save(filepath)
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return jsonify({
                'error': 'Failed to save file',
                'error_type': 'file_save_error',
                'details': str(e),
                'filename': filename,
                'status': 'error'
            }), 500
        
        # Verify file was saved correctly
        if not os.path.exists(filepath):
            return jsonify({
                'error': 'File not saved properly',
                'error_type': 'file_save_verification_failed',
                'details': 'File was not created on disk',
                'filename': filename,
                'status': 'error'
            }), 500
        
        # Analyze the image
        logger.info(f"Analyzing image: {filename}")
        image_analysis = image_analyzer.analyze_engine_image(filepath)
        
        # Check for analysis errors
        if 'error' in image_analysis:
            return jsonify({
                'error': image_analysis['error'],
                'error_type': image_analysis.get('error_type', 'analysis_error'),
                'details': image_analysis.get('details', 'Unknown analysis error'),
                'filename': filename,
                'status': 'error'
            }), 400
        
        # Generate analysis plot
        try:
            plot_data = image_analyzer.generate_analysis_plot(filepath, image_analysis)
        except Exception as e:
            plot_data = None
            logger.warning(f"Failed to generate analysis plot: {e}")
        
        # Enhanced analysis with model integration
        try:
            # Get cached models
            models = model_manager.get_models()
            
            if not models:
                # No models available - return image analysis only
                combined_analysis = {
                    'image_analysis': image_analysis,
                    'sensor_prediction': {
                        'rul': 'Models not available',
                        'dataset': 'N/A',
                        'selection_reason': 'No trained models found',
                        'error': 'Please train models first using manutenzione_predittiva.py'
                    },
                    'plot_data': plot_data,
                    'filename': filename,
                    'sensor_data_info': {
                        'simulation_info': {
                            'warning': 'No sensor data generated - models unavailable'
                        }
                    },
                    'status': 'partial_success',
                    'message': 'Image analysis completed, but predictive models are not available'
                }
                return jsonify(combined_analysis)
            
            # Choose best model with context
            user_context = request.form.get('operational_context')  # Optional user input
            dataset, selection_reason = model_selector.choose_best_model(image_analysis, user_context)
            
            # Generate sensor data with correct dataset dimensions
            sensor_data, sensor_data_info = sensor_generator.generate_sensor_data_from_image(image_analysis, dataset)
            
            # Make prediction
            try:
                rul_prediction, prediction_metadata = model_manager.predict_rul(sensor_data, dataset)
                
                # Get engine status
                engine_status = status_analyzer.get_engine_status(rul_prediction)
                
            except Exception as e:
                rul_prediction = f"Prediction failed: {str(e)}"
                prediction_metadata = {'error': str(e)}
                engine_status = status_analyzer.get_engine_status(0)  # Default status
            
            # Prepare comprehensive response
            combined_analysis = {
                'image_analysis': image_analysis,
                'sensor_prediction': {
                    'rul': rul_prediction,
                    'dataset': dataset,
                    'selection_reason': selection_reason,
                    'user_context': user_context,
                    'prediction_metadata': prediction_metadata
                },
                'engine_status': engine_status,
                'plot_data': plot_data,
                'filename': filename,
                'sensor_data_info': sensor_data_info,
                'status': 'success',
                'message': 'Analysis completed successfully'
            }
            
            return jsonify(combined_analysis)
            
        except Exception as e:
            logger.error(f"Model integration failed: {e}")
            # Model integration failed - return image analysis only
            combined_analysis = {
                'image_analysis': image_analysis,
                'sensor_prediction': {
                    'rul': 'Model integration failed',
                    'dataset': 'N/A',
                    'selection_reason': 'Error during model processing',
                    'error': str(e)
                },
                'plot_data': plot_data,
                'filename': filename,
                'sensor_data_info': {
                    'simulation_info': {
                        'warning': 'Sensor data generation failed due to model error'
                    },
                    'error': str(e)
                },
                'status': 'partial_success',
                'message': 'Image analysis completed, but model integration failed'
            }
            return jsonify(combined_analysis)
            
    except Exception as e:
        logger.error(f"Unexpected error in upload: {e}")
        return jsonify({
            'error': 'Unexpected error during file processing',
            'error_type': 'unexpected_error',
            'details': str(e),
            'status': 'error'
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        File content
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/config/summary')
def config_summary():
    """
    Get configuration summary (for debugging, remove in production)
    
    Returns:
        JSON with configuration summary
    """
    if app.config.get('DEBUG', False):
        return jsonify({
            'environment': os.environ.get('FLASK_ENV', 'development'),
            'debug_mode': app.config.get('DEBUG', False),
            'upload_folder': app.config.get('UPLOAD_FOLDER'),
            'models_dir': app.config.get('MODELS_DIR'),
            'max_file_size': app.config.get('MAX_FILE_SIZE'),
            'allowed_extensions': list(app.config.get('ALLOWED_EXTENSIONS', [])),
            'rate_limiting': app.config.get('RATE_LIMIT_ENABLED', False)
        })
    else:
        return jsonify({
            'error': 'Configuration summary not available in production'
        }), 403


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Resource not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500


@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'error': 'File too large',
        'error_type': 'file_too_large',
        'max_size_mb': app.config['MAX_FILE_SIZE'] / (1024 * 1024),
        'status': 'error'
    }), 413


if __name__ == '__main__':
    # Print startup information
    print("\n" + "="*60)
    print("ENGINE PREDICTIVE MAINTENANCE WEB INTERFACE")
    print("="*60)
    print(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"Models loaded: {model_manager.get_model_status()['total_models']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Max file size: {app.config['MAX_FILE_SIZE'] / (1024*1024):.1f} MB")
    print("="*60)
    
    # Run the application
    app.run(
        host=os.environ.get('FLASK_HOST', '0.0.0.0'),
        port=int(os.environ.get('FLASK_PORT', 5000)),
        debug=app.config.get('DEBUG', False)
    )