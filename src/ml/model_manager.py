#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Manager Module for Engine Predictive Maintenance

This module handles all model-related functionality including:
- Model loading and caching
- Model prediction
- Model status monitoring
- Model versioning

Separated from main app.py for better code organization and maintainability.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, List, Any, Tuple
import logging
import json
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManagerError(Exception):
    """Custom exception for model management errors"""
    pass

class ModelManager:
    """
    Manages predictive maintenance models with caching and error handling
    """
    
    def __init__(self, models_dir: str, config: Dict[str, Any]):
        """
        Initialize the model manager
        
        Args:
            models_dir: Directory containing model files
            config: Configuration dictionary
        """
        self.models_dir = models_dir
        self.config = config
        self._models_cache = {}
        self._models_loaded = False
        self._models_error = None
        self._load_timestamp = None
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
    
    def load_models(self) -> bool:
        """
        Load all trained models with comprehensive error handling
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            logger.info("Loading predictive maintenance models...")
            
            # Clear existing cache
            self._models_cache.clear()
            self._models_error = None
            
            # Available datasets
            datasets = ['FD001', 'FD002', 'FD003', 'FD004']
            loaded_count = 0
            
            for dataset in datasets:
                model_loaded = self._load_single_model(dataset)
                if model_loaded:
                    loaded_count += 1
                    logger.info(f"✓ Loaded model {dataset}")
                else:
                    logger.warning(f"⚠️ Failed to load model {dataset}")
            
            self._models_loaded = loaded_count > 0
            self._load_timestamp = datetime.now()
            
            if self._models_loaded:
                logger.info(f"✅ Successfully loaded {loaded_count}/{len(datasets)} models")
            else:
                self._models_error = "No models could be loaded"
                logger.error("❌ No models loaded successfully")
            
            return self._models_loaded
            
        except Exception as e:
            self._models_error = str(e)
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def _load_single_model(self, dataset: str) -> bool:
        """
        Load a single model for a specific dataset
        
        Args:
            dataset: Dataset identifier (FD001, FD002, etc.)
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Try different naming conventions
            model_paths = [
                os.path.join(self.models_dir, f'model_{dataset.lower()}.h5'),
                os.path.join(self.models_dir, f'modello_{dataset.lower()}.h5'),
                os.path.join(self.models_dir, f'model_{dataset.lower()}_*.h5'),
                os.path.join(self.models_dir, f'modello_{dataset.lower()}_*.h5')
            ]
            
            model_path = None
            
            # Try exact matches first
            for path in model_paths[:2]:
                if os.path.exists(path):
                    model_path = path
                    break
            
            # If no exact match, try pattern matching
            if model_path is None:
                for pattern in model_paths[2:]:
                    matches = glob.glob(pattern)
                    if matches:
                        # Sort by modification time and take the most recent
                        matches.sort(key=os.path.getmtime, reverse=True)
                        model_path = matches[0]
                        break
            
            if model_path is None:
                logger.warning(f"No model file found for {dataset}")
                return False
            
            # Load the model
            model = tf.keras.models.load_model(model_path)
            self._models_cache[dataset] = {
                'model': model,
                'path': model_path,
                'load_time': datetime.now(),
                'file_size': os.path.getsize(model_path)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {dataset}: {e}")
            return False
    
    def get_models(self) -> Optional[Dict[str, Any]]:
        """
        Get cached models
        
        Returns:
            Dictionary of loaded models or None if not loaded
        """
        if not self._models_loaded:
            return None
        
        # Return only the model objects for compatibility
        return {dataset: cache['model'] for dataset, cache in self._models_cache.items()}
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all models
        
        Returns:
            Dictionary with model status information
        """
        status = {
            'models_loaded': self._models_loaded,
            'load_timestamp': self._load_timestamp.isoformat() if self._load_timestamp else None,
            'total_models': len(self._models_cache),
            'available_datasets': list(self._models_cache.keys()),
            'error': self._models_error
        }
        
        # Add individual model status
        model_details = {}
        for dataset, cache in self._models_cache.items():
            model_details[dataset] = {
                'loaded': True,
                'path': cache['path'],
                'load_time': cache['load_time'].isoformat(),
                'file_size': cache['file_size'],
                'model_params': cache['model'].count_params()
            }
        
        status['model_details'] = model_details
        
        return status
    
    def predict_rul(self, sensor_data: np.ndarray, dataset: str = 'FD001') -> Tuple[float, Dict[str, Any]]:
        """
        Predict RUL using the specified model
        
        Args:
            sensor_data: Input sensor data
            dataset: Dataset identifier
            
        Returns:
            Tuple of (rul_prediction, prediction_metadata)
        """
        try:
            if dataset not in self._models_cache:
                raise ModelManagerError(f"Model for dataset {dataset} not available")
            
            model_info = self._models_cache[dataset]
            model = model_info['model']
            
            # Prepare data
            sequence_length = self._get_sequence_length(dataset)
            feature_count = self._get_feature_count(dataset)
            prepared_data = self._prepare_sensor_data(sensor_data, sequence_length, feature_count)
            
            # Make prediction
            prediction = model.predict(prepared_data, verbose=0)
            rul = float(prediction[0][0])
            
            # Ensure positive prediction
            rul = max(0, rul)
            
            metadata = {
                'dataset': dataset,
                'model_path': model_info['path'],
                'prediction_time': datetime.now().isoformat(),
                'input_shape': sensor_data.shape,
                'prepared_shape': prepared_data.shape,
                'raw_prediction': float(prediction[0][0]),
                'final_prediction': rul
            }
            
            return rul, metadata
            
        except Exception as e:
            logger.error(f"Error in RUL prediction: {e}")
            raise ModelManagerError(f"Prediction failed: {str(e)}")
    
    def _get_sequence_length(self, dataset: str) -> int:
        """
        Get appropriate sequence length for dataset
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            Sequence length
        """
        sequence_lengths = {
            'FD001': 50,
            'FD002': 60,
            'FD003': 50,
            'FD004': 70
        }
        
        return sequence_lengths.get(dataset, 50)
    
    def _get_feature_count(self, dataset: str) -> int:
        """
        Get appropriate number of features for dataset
        
        Args:
            dataset: Dataset identifier
            
        Returns:
            Number of features
        """
        feature_counts = {
            'FD001': 11,
            'FD002': 23,
            'FD003': 13,
            'FD004': 23
        }
        
        return feature_counts.get(dataset, 23)
    
    def _prepare_sensor_data(self, sensor_data: np.ndarray, sequence_length: int, feature_count: int) -> np.ndarray:
        """
        Prepare sensor data for prediction
        
        Args:
            sensor_data: Raw sensor data
            sequence_length: Required sequence length
            feature_count: Required number of features
            
        Returns:
            Prepared sensor data
        """
        if len(sensor_data.shape) == 3:
            # Remove batch dimension if present
            sensor_data = sensor_data[0]
        
        if len(sensor_data.shape) == 2:
            # Adjust sequence length
            if sensor_data.shape[0] < sequence_length:
                # Pad with zeros if sequence is too short
                padding = np.zeros((sequence_length - sensor_data.shape[0], sensor_data.shape[1]))
                sensor_data = np.vstack([padding, sensor_data])
            elif sensor_data.shape[0] > sequence_length:
                # Take last sequence_length samples
                sensor_data = sensor_data[-sequence_length:]
            
            # Adjust feature count
            if sensor_data.shape[1] < feature_count:
                # Pad with zeros if not enough features
                padding = np.zeros((sensor_data.shape[0], feature_count - sensor_data.shape[1]))
                sensor_data = np.hstack([sensor_data, padding])
            elif sensor_data.shape[1] > feature_count:
                # Take first feature_count features
                sensor_data = sensor_data[:, :feature_count]
            
            sensor_data = sensor_data.reshape(1, sequence_length, feature_count)
        
        return sensor_data
    
    def reload_models(self) -> bool:
        """
        Reload all models (useful for updates)
        
        Returns:
            True if reload successful
        """
        logger.info("Reloading models...")
        return self.load_models()
    
    def get_best_model_version(self, dataset: str, metric: str = 'r2') -> Optional[Dict[str, Any]]:
        """
        Get the best model version based on a specific metric
        
        Args:
            dataset: Dataset identifier
            metric: Metric to use for selection ('r2', 'rmse', 'mae')
            
        Returns:
            Model metadata or None if not found
        """
        try:
            # Find all model files for this dataset
            model_pattern = os.path.join(self.models_dir, f"model_{dataset.lower()}_*.h5")
            metadata_pattern = os.path.join(self.models_dir, f"metadata_{dataset.lower()}_*.json")
            
            model_files = glob.glob(model_pattern)
            metadata_files = glob.glob(metadata_pattern)
            
            if not model_files:
                return None
            
            best_model_path = None
            best_metadata = None
            best_metric_value = None
            
            # Evaluate each model version
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    metric_value = metadata.get('performance_metrics', {}).get(metric)
                    
                    if metric_value is not None:
                        # For R2, higher is better; for RMSE/MAE, lower is better
                        is_better = False
                        if metric == 'r2':
                            is_better = best_metric_value is None or metric_value > best_metric_value
                        else:  # rmse, mae
                            is_better = best_metric_value is None or metric_value < best_metric_value
                        
                        if is_better:
                            best_metric_value = metric_value
                            best_metadata = metadata
                            # Find corresponding model file
                            timestamp = metadata['model_info']['timestamp']
                            model_type = metadata['model_info']['model_type']
                            for model_file in model_files:
                                if timestamp in model_file and model_type in model_file:
                                    best_model_path = model_file
                                    break
                
                except Exception as e:
                    logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                    continue
            
            if best_model_path is None:
                return None
            
            return {
                'model_path': best_model_path,
                'metadata': best_metadata,
                'metric_value': best_metric_value,
                'metric_name': metric
            }
            
        except Exception as e:
            logger.error(f"Error finding best model version: {e}")
            return None


class EngineStatusAnalyzer:
    """
    Analyze engine status based on RUL predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize status analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.status_thresholds = config.get('STATUS_THRESHOLDS', {
            'critical': 20,
            'warning': 50,
            'good': 100
        })
    
    def get_engine_status(self, rul_value: float) -> Dict[str, Any]:
        """
        Determine engine status based on RUL prediction
        
        Args:
            rul_value: Predicted RUL value
            
        Returns:
            Status information dictionary
        """
        if not isinstance(rul_value, (int, float)) or np.isnan(rul_value):
            return {
                'status': 'Unknown',
                'confidence': 0.0,
                'recommendation': 'Unable to determine status',
                'urgency': 'unknown'
            }
        
        # Determine status based on thresholds
        if rul_value <= self.status_thresholds['critical']:
            status = 'Critical'
            urgency = 'high'
            recommendation = 'Immediate maintenance required'
            confidence = 0.9
        elif rul_value <= self.status_thresholds['warning']:
            status = 'Warning'
            urgency = 'medium'
            recommendation = 'Schedule maintenance soon'
            confidence = 0.8
        elif rul_value <= self.status_thresholds['good']:
            status = 'Good'
            urgency = 'low'
            recommendation = 'Continue monitoring'
            confidence = 0.7
        else:
            status = 'Excellent'
            urgency = 'none'
            recommendation = 'Normal operation'
            confidence = 0.8
        
        return {
            'status': status,
            'rul_value': float(rul_value),
            'confidence': confidence,
            'recommendation': recommendation,
            'urgency': urgency,
            'thresholds': self.status_thresholds
        }


# Factory functions
def create_model_manager(models_dir: str, config: Dict[str, Any]) -> ModelManager:
    """
    Factory function to create a ModelManager instance
    
    Args:
        models_dir: Directory containing model files
        config: Configuration dictionary
        
    Returns:
        Configured ModelManager instance
    """
    return ModelManager(models_dir, config)


def create_status_analyzer(config: Dict[str, Any]) -> EngineStatusAnalyzer:
    """
    Factory function to create an EngineStatusAnalyzer instance
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured EngineStatusAnalyzer instance
    """
    return EngineStatusAnalyzer(config)