"""
Configuration file for the Engine Predictive Maintenance Web Interface

This module provides centralized configuration management with environment variable support
for sensitive data and deployment-specific settings.
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration class with common settings"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'engine-predictive-maintenance-2024'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Enhanced security settings
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
    MIN_FILE_SIZE = 1024  # 1KB minimum for valid images
    MAX_REASONABLE_SIZE = 50 * 1024 * 1024  # 50MB absolute maximum (DoS protection)
    MAX_FILENAME_LENGTH = 255  # Maximum filename length
    
    # Upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Security: Blocked file extensions (additional protection)
    BLOCKED_EXTENSIONS = {'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js', 'jar', 'war', 'ear', 'apk'}
    
    # Rate limiting settings (for future implementation)
    RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'false').lower() == 'true'
    RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', '100'))  # requests per minute
    RATE_LIMIT_WINDOW = int(os.environ.get('RATE_LIMIT_WINDOW', '60'))  # seconds
    
    # Model settings
    MODELS_DIR = os.environ.get('MODELS_DIR') or 'models'
    RESULTS_DIR = os.environ.get('RESULTS_DIR') or 'results'
    
    # Analysis settings
    MAX_IMAGE_SIZE = (1920, 1080)  # Max image dimensions
    MIN_IMAGE_SIZE = (100, 100)    # Min image dimensions
    
    # Rust detection thresholds (more conservative)
    RUST_COLOR_LOWER = [8, 80, 80]   # HSV lower bound for rust (more selective)
    RUST_COLOR_UPPER = [20, 255, 255] # HSV upper bound for rust
    RUST_THRESHOLD = 3.0  # Percentage of image that must be rust to trigger detection (increased)
    
    # Oil leak detection thresholds (more conservative)
    DARK_THRESHOLD = 30   # Pixel value threshold for dark areas (lower = darker needed)
    OIL_LEAK_THRESHOLD = 8.0  # Percentage of dark areas to trigger detection (increased)
    
    # Blob detection settings
    BLOB_MIN_AREA = 100
    BLOB_MAX_AREA = 5000
    
    # Engine status thresholds
    STATUS_THRESHOLDS = {
        'critical': int(os.environ.get('STATUS_CRITICAL_THRESHOLD', '20')),
        'warning': int(os.environ.get('STATUS_WARNING_THRESHOLD', '50')),
        'good': int(os.environ.get('STATUS_GOOD_THRESHOLD', '100'))
    }
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')
    
    # Database settings (for future use)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # External API settings (for future integrations)
    API_TIMEOUT = int(os.environ.get('API_TIMEOUT', '30'))  # seconds
    API_RETRY_ATTEMPTS = int(os.environ.get('API_RETRY_ATTEMPTS', '3'))
    
    # Cache settings
    CACHE_ENABLED = os.environ.get('CACHE_ENABLED', 'true').lower() == 'true'
    CACHE_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', '3600'))  # 1 hour
    
    # Model prediction settings
    PREDICTION_TIMEOUT = int(os.environ.get('PREDICTION_TIMEOUT', '30'))  # seconds
    MAX_PREDICTION_RETRIES = int(os.environ.get('MAX_PREDICTION_RETRIES', '3'))
    
    # Image analysis settings
    IMAGE_ANALYSIS_TIMEOUT = int(os.environ.get('IMAGE_ANALYSIS_TIMEOUT', '60'))  # seconds
    MAX_IMAGE_ANALYSIS_RETRIES = int(os.environ.get('MAX_IMAGE_ANALYSIS_RETRIES', '2'))
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Development-specific settings
    LOG_LEVEL = 'DEBUG'
    CACHE_ENABLED = False  # Disable cache in development
    
    # Development database
    DATABASE_URL = os.environ.get('DEV_DATABASE_URL') or 'sqlite:///dev.db'
    
    # Development model settings
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    
    # Development upload settings
    UPLOAD_FOLDER = 'uploads'
    
    # Development security (less restrictive)
    MAX_FILE_SIZE = 32 * 1024 * 1024  # 32MB for development
    RATE_LIMIT_ENABLED = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Production-specific settings
    LOG_LEVEL = 'WARNING'
    CACHE_ENABLED = True
    
    # Production database
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Production model settings
    MODELS_DIR = os.environ.get('PROD_MODELS_DIR', '/app/models')
    RESULTS_DIR = os.environ.get('PROD_RESULTS_DIR', '/app/results')
    
    # Production upload settings
    UPLOAD_FOLDER = os.environ.get('PROD_UPLOAD_FOLDER', '/app/uploads')
    
    # Production security (more restrictive)
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB for production
    RATE_LIMIT_ENABLED = True
    
    # Production timeouts
    PREDICTION_TIMEOUT = 15  # Shorter timeout in production
    IMAGE_ANALYSIS_TIMEOUT = 30  # Shorter timeout in production


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    
    # Testing-specific settings
    LOG_LEVEL = 'DEBUG'
    CACHE_ENABLED = False
    
    # Testing database
    DATABASE_URL = 'sqlite:///:memory:'  # In-memory database for testing
    
    # Testing directories
    MODELS_DIR = 'test_models'
    RESULTS_DIR = 'test_results'
    UPLOAD_FOLDER = 'test_uploads'
    
    # Testing security (very permissive)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB for testing
    RATE_LIMIT_ENABLED = False
    
    # Testing timeouts (longer for debugging)
    PREDICTION_TIMEOUT = 60
    IMAGE_ANALYSIS_TIMEOUT = 120


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config() -> Dict[str, Any]:
    """
    Get current configuration based on environment
    
    Returns:
        Configuration dictionary
    """
    config_name = os.environ.get('FLASK_ENV', 'development')
    config_class = config.get(config_name, config['default'])
    
    # Convert config class to dictionary
    config_dict = {}
    for key in dir(config_class):
        if not key.startswith('_'):
            value = getattr(config_class, key)
            if not callable(value):
                config_dict[key] = value
    
    return config_dict


def validate_config(config_dict: Dict[str, Any]) -> bool:
    """
    Validate configuration settings
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = [
        'SECRET_KEY', 'UPLOAD_FOLDER', 'MODELS_DIR', 'RESULTS_DIR',
        'MAX_FILE_SIZE', 'MIN_FILE_SIZE', 'ALLOWED_EXTENSIONS'
    ]
    
    missing_keys = [key for key in required_keys if key not in config_dict]
    if missing_keys:
        print(f"❌ Missing required configuration keys: {missing_keys}")
        return False
    
    # Validate file size settings
    if config_dict['MAX_FILE_SIZE'] <= config_dict['MIN_FILE_SIZE']:
        print("❌ MAX_FILE_SIZE must be greater than MIN_FILE_SIZE")
        return False
    
    # Validate directories
    directories = ['UPLOAD_FOLDER', 'MODELS_DIR', 'RESULTS_DIR']
    for directory in directories:
        dir_path = config_dict[directory]
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✓ Created directory: {dir_path}")
            except Exception as e:
                print(f"❌ Cannot create directory {dir_path}: {e}")
                return False
    
    print("✅ Configuration validation passed")
    return True


def get_sensitive_config() -> Dict[str, str]:
    """
    Get configuration values that should be treated as sensitive
    
    Returns:
        Dictionary of sensitive configuration values
    """
    return {
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'NOT_SET'),
        'DATABASE_URL': os.environ.get('DATABASE_URL', 'NOT_SET'),
        'API_KEYS': os.environ.get('API_KEYS', 'NOT_SET'),
        'ENCRYPTION_KEY': os.environ.get('ENCRYPTION_KEY', 'NOT_SET')
    }


def check_environment_variables() -> Dict[str, bool]:
    """
    Check if required environment variables are set
    
    Returns:
        Dictionary indicating which environment variables are set
    """
    env_vars = {
        'SECRET_KEY': 'SECRET_KEY' in os.environ,
        'DATABASE_URL': 'DATABASE_URL' in os.environ,
        'FLASK_ENV': 'FLASK_ENV' in os.environ,
        'LOG_LEVEL': 'LOG_LEVEL' in os.environ,
        'UPLOAD_FOLDER': 'UPLOAD_FOLDER' in os.environ,
        'MODELS_DIR': 'MODELS_DIR' in os.environ,
        'RATE_LIMIT_ENABLED': 'RATE_LIMIT_ENABLED' in os.environ
    }
    
    return env_vars


def print_config_summary(config_dict: Dict[str, Any]) -> None:
    """
    Print a summary of the current configuration
    
    Args:
        config_dict: Configuration dictionary
    """
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    # Environment
    env = os.environ.get('FLASK_ENV', 'development')
    print(f"Environment: {env}")
    
    # Key settings
    print(f"Debug Mode: {config_dict.get('DEBUG', False)}")
    print(f"Testing Mode: {config_dict.get('TESTING', False)}")
    print(f"Log Level: {config_dict.get('LOG_LEVEL', 'INFO')}")
    
    # File settings
    print(f"Max File Size: {config_dict.get('MAX_FILE_SIZE', 0) / (1024*1024):.1f} MB")
    print(f"Upload Folder: {config_dict.get('UPLOAD_FOLDER', 'N/A')}")
    print(f"Models Directory: {config_dict.get('MODELS_DIR', 'N/A')}")
    
    # Security settings
    print(f"Rate Limiting: {'Enabled' if config_dict.get('RATE_LIMIT_ENABLED', False) else 'Disabled'}")
    print(f"Cache: {'Enabled' if config_dict.get('CACHE_ENABLED', False) else 'Disabled'}")
    
    # Environment variables
    env_vars = check_environment_variables()
    print(f"\nEnvironment Variables:")
    for var, is_set in env_vars.items():
        status = "✓" if is_set else "✗"
        print(f"  {status} {var}")
    
    print("="*60)


# Example usage and validation
if __name__ == "__main__":
    # Get current configuration
    current_config = get_config()
    
    # Validate configuration
    is_valid = validate_config(current_config)
    
    # Print summary
    print_config_summary(current_config)
    
    if is_valid:
        print("\n✅ Configuration is valid and ready to use")
    else:
        print("\n❌ Configuration validation failed")
        exit(1) 