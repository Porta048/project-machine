#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for ML Core Project
Contains all hyperparameters and settings for model training
"""

import os
from typing import Dict, List, Any

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for directory in [MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset configuration
DATASETS = {
    'FD001': {
        'description': 'Single fault mode, single operating condition',
        'train_units': 100,
        'test_units': 100,
        'fault_modes': 1,
        'operating_conditions': 1
    },
    'FD002': {
        'description': 'Single fault mode, multiple operating conditions',
        'train_units': 260,
        'test_units': 259,
        'fault_modes': 1,
        'operating_conditions': 6
    },
    'FD003': {
        'description': 'Multiple fault modes, single operating condition',
        'train_units': 100,
        'test_units': 100,
        'fault_modes': 2,
        'operating_conditions': 1
    },
    'FD004': {
        'description': 'Multiple fault modes, multiple operating conditions',
        'train_units': 248,
        'test_units': 249,
        'fault_modes': 2,
        'operating_conditions': 6
    }
}

# Feature configuration
FEATURE_CONFIG = {
    'sensor_columns': [f'sensor_{i}' for i in range(1, 22)],
    'setting_columns': ['setting_1', 'setting_2', 'setting_3'],
    'target_column': 'RUL',
    'sequence_length': 30,
    'max_rul': 125,  # RUL capping value
    'normalization': 'standard'  # 'standard', 'minmax', 'robust'
}

# Model architecture configurations
MODEL_CONFIGS = {
    'basic_lstm': {
        'type': 'lstm',
        'lstm_units': [100, 50],
        'dense_units': [50, 25],
        'dropout_rate': 0.2,
        'batch_normalization': True,
        'activation': 'relu',
        'output_activation': 'linear'
    },
    
    'advanced_lstm': {
        'type': 'lstm',
        'lstm_units': [128, 64, 32],
        'dense_units': [64, 32, 16],
        'dropout_rate': 0.3,
        'batch_normalization': True,
        'activation': 'relu',
        'output_activation': 'linear'
    },
    
    'deep_lstm': {
        'type': 'lstm',
        'lstm_units': [256, 128, 64, 32],
        'dense_units': [128, 64, 32, 16],
        'dropout_rate': 0.4,
        'batch_normalization': True,
        'activation': 'relu',
        'output_activation': 'linear'
    },
    
    'lightweight_lstm': {
        'type': 'lstm',
        'lstm_units': [64, 32],
        'dense_units': [32, 16],
        'dropout_rate': 0.15,
        'batch_normalization': True,
        'activation': 'relu',
        'output_activation': 'linear'
    }
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    
    # Callbacks
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 15,
        'restore_best_weights': True
    },
    
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 8,
        'min_lr': 1e-7
    },
    
    'model_checkpoint': {
        'monitor': 'val_loss',
        'save_best_only': True
    }
}

# Hyperparameter optimization grid
OPTIMIZATION_GRID = {
    'lstm_units': [
        [64, 32],
        [100, 50],
        [128, 64],
        [128, 64, 32]
    ],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
    'learning_rate': [0.0005, 0.001, 0.002, 0.005],
    'batch_size': [16, 32, 64, 128],
    'sequence_length': [20, 30, 40, 50]
}

# Advanced optimization grid (for fine-tuning)
ADVANCED_OPTIMIZATION_GRID = {
    'lstm_units': [
        [100, 50],
        [128, 64],
        [128, 64, 32],
        [256, 128, 64]
    ],
    'dropout_rate': [0.15, 0.2, 0.25, 0.3, 0.35],
    'learning_rate': [0.0003, 0.0005, 0.001, 0.0015, 0.002],
    'batch_size': [24, 32, 48, 64],
    'sequence_length': [25, 30, 35, 40],
    'dense_units': [
        [50, 25],
        [64, 32],
        [64, 32, 16],
        [128, 64, 32]
    ]
}

# Evaluation metrics configuration
EVALUATION_CONFIG = {
    'primary_metric': 'rmse',
    'metrics': {
        'rmse': {'higher_is_better': False, 'weight': 1.0},
        'mae': {'higher_is_better': False, 'weight': 0.8},
        'r2': {'higher_is_better': True, 'weight': 0.6},
        'phm08_score': {'higher_is_better': False, 'weight': 1.2}
    },
    'early_prediction_penalty': 10,  # PHM08 scoring
    'late_prediction_penalty': 13    # PHM08 scoring
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True,
    'log_file': os.path.join(LOGS_DIR, 'training.log')
}

# GPU configuration
GPU_CONFIG = {
    'memory_growth': True,
    'allow_soft_placement': True,
    'mixed_precision': False  # Set to True for RTX cards
}

# Data augmentation configuration
DATA_AUGMENTATION_CONFIG = {
    'enabled': False,
    'noise_factor': 0.01,
    'time_warping': False,
    'magnitude_warping': False
}

# Model ensemble configuration
ENSEMBLE_CONFIG = {
    'enabled': False,
    'n_models': 5,
    'voting': 'average',  # 'average', 'weighted'
    'diversity_threshold': 0.1
}

# Cross-validation configuration
CV_CONFIG = {
    'enabled': False,
    'n_folds': 5,
    'shuffle': True,
    'random_state': 42
}

# Export configuration
EXPORT_CONFIG = {
    'save_model': True,
    'save_weights': True,
    'save_history': True,
    'save_predictions': True,
    'save_metadata': True,
    'model_format': 'h5',  # 'h5', 'savedmodel', 'tflite'
    'compression': True
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'track_memory': True,
    'track_time': True,
    'profile_training': False,
    'save_tensorboard_logs': True,
    'tensorboard_dir': os.path.join(LOGS_DIR, 'tensorboard')
}

# Reproducibility
REPRODUCIBILITY_CONFIG = {
    'random_seed': 42,
    'numpy_seed': 42,
    'tensorflow_seed': 42,
    'deterministic_ops': True
}

def get_config(config_name: str) -> Dict[str, Any]:
    """
    Get configuration by name
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'datasets': DATASETS,
        'features': FEATURE_CONFIG,
        'models': MODEL_CONFIGS,
        'training': TRAINING_CONFIG,
        'optimization': OPTIMIZATION_GRID,
        'advanced_optimization': ADVANCED_OPTIMIZATION_GRID,
        'evaluation': EVALUATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'gpu': GPU_CONFIG,
        'augmentation': DATA_AUGMENTATION_CONFIG,
        'ensemble': ENSEMBLE_CONFIG,
        'cv': CV_CONFIG,
        'export': EXPORT_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'reproducibility': REPRODUCIBILITY_CONFIG
    }
    
    return configs.get(config_name, {})

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model configuration by name
    
    Args:
        model_name: Name of the model configuration
        
    Returns:
        Model configuration dictionary
    """
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['basic_lstm'])

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset information
    
    Args:
        dataset_name: Name of the dataset (FD001, FD002, FD003, FD004)
        
    Returns:
        Dataset information dictionary
    """
    return DATASETS.get(dataset_name, {})

def update_config(config_name: str, updates: Dict[str, Any]) -> None:
    """
    Update configuration with new values
    
    Args:
        config_name: Name of the configuration to update
        updates: Dictionary of updates to apply
    """
    global TRAINING_CONFIG, MODEL_CONFIGS, FEATURE_CONFIG
    
    if config_name == 'training':
        TRAINING_CONFIG.update(updates)
    elif config_name == 'features':
        FEATURE_CONFIG.update(updates)
    elif config_name in MODEL_CONFIGS:
        MODEL_CONFIGS[config_name].update(updates)
    else:
        print(f"Warning: Unknown configuration '{config_name}'")

# Dataset-specific optimized configurations
DATASET_OPTIMIZED_CONFIGS = {
    'FD001': {
        'model': 'basic_lstm',
        'sequence_length': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.2
    },
    'FD002': {
        'model': 'advanced_lstm',
        'sequence_length': 35,
        'batch_size': 64,
        'learning_rate': 0.0005,
        'dropout_rate': 0.3
    },
    'FD003': {
        'model': 'advanced_lstm',
        'sequence_length': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.25
    },
    'FD004': {
        'model': 'deep_lstm',
        'sequence_length': 40,
        'batch_size': 64,
        'learning_rate': 0.0005,
        'dropout_rate': 0.35
    }
}

def get_optimized_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get optimized configuration for a specific dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Optimized configuration dictionary
    """
    return DATASET_OPTIMIZED_CONFIGS.get(dataset_name, DATASET_OPTIMIZED_CONFIGS['FD001'])