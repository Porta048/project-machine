#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core ML Project - Predictive Maintenance Models
Focused on high-performance model training and optimization

This is a streamlined version containing only essential components:
- Data loading and preprocessing
- Model architecture definition
- Training pipeline
- Model evaluation and optimization
- Model persistence
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Efficient data loading and preprocessing for turbofan engine datasets
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.feature_columns = [
            'unit_id', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'
        ] + [f'sensor_{i}' for i in range(1, 22)]
        
        self.sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        self.setting_columns = ['setting_1', 'setting_2', 'setting_3']
        
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training, test data and RUL values for a specific dataset
        
        Args:
            dataset_name: Dataset identifier (FD001, FD002, FD003, FD004)
            
        Returns:
            Tuple of (train_df, test_df, rul_df)
        """
        try:
            # Load training data
            train_path = os.path.join(self.data_dir, f'train_{dataset_name}.txt')
            train_df = pd.read_csv(train_path, sep=' ', header=None, names=self.feature_columns)
            
            # Load test data
            test_path = os.path.join(self.data_dir, f'test_{dataset_name}.txt')
            test_df = pd.read_csv(test_path, sep=' ', header=None, names=self.feature_columns)
            
            # Load RUL values
            rul_path = os.path.join(self.data_dir, f'RUL_{dataset_name}.txt')
            rul_df = pd.read_csv(rul_path, sep=' ', header=None, names=['RUL'])
            
            logger.info(f"Loaded {dataset_name}: Train={len(train_df)}, Test={len(test_df)}, RUL={len(rul_df)}")
            
            return train_df, test_df, rul_df
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                       sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data for LSTM training
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Calculate RUL for training data
        train_df = self._calculate_rul(train_df)
        
        # Select relevant features
        feature_cols = self.sensor_columns + self.setting_columns
        
        # Normalize features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_df[feature_cols])
        test_features = scaler.transform(test_df[feature_cols])
        
        # Create sequences
        X_train, y_train = self._create_sequences(
            train_df, train_features, sequence_length, 'RUL'
        )
        
        X_test, y_test = self._create_sequences(
            test_df, test_features, sequence_length
        )
        
        logger.info(f"Preprocessed data: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"                   X_test={X_test.shape}, y_test={y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def _calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Remaining Useful Life for each unit"""
        df = df.copy()
        
        # Calculate max cycles for each unit
        max_cycles = df.groupby('unit_id')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycles']
        
        # Merge and calculate RUL
        df = df.merge(max_cycles, on='unit_id')
        df['RUL'] = df['max_cycles'] - df['time_cycles']
        
        # Cap RUL at 125 (common practice in literature)
        df['RUL'] = df['RUL'].clip(upper=125)
        
        return df
    
    def _create_sequences(self, df: pd.DataFrame, features: np.ndarray, 
                         sequence_length: int, target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]
            unit_features = features[df['unit_id'] == unit_id]
            
            if len(unit_data) < sequence_length:
                continue
                
            for i in range(sequence_length, len(unit_data)):
                X.append(unit_features[i-sequence_length:i])
                
                if target_col:
                    y.append(unit_data.iloc[i][target_col])
                else:
                    # For test data, use the last RUL value
                    y.append(0)  # Placeholder
        
        return np.array(X), np.array(y)

class ModelArchitecture:
    """
    Optimized LSTM architecture for RUL prediction
    """
    
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        
    def build_lstm_model(self, lstm_units: List[int] = [100, 50], 
                        dropout_rate: float = 0.2,
                        learning_rate: float = 0.001) -> Model:
        """
        Build optimized LSTM model
        
        Args:
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Input(shape=self.input_shape),
            
            # First LSTM layer
            LSTM(lstm_units[0], return_sequences=True, name='lstm_1'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Second LSTM layer
            LSTM(lstm_units[1], return_sequences=False, name='lstm_2'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Dense layers
            Dense(50, activation='relu', name='dense_1'),
            Dropout(dropout_rate),
            Dense(25, activation='relu', name='dense_2'),
            Dense(1, activation='linear', name='output')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        
        return model
    
    def build_advanced_model(self, lstm_units: List[int] = [128, 64, 32],
                           dropout_rate: float = 0.3,
                           learning_rate: float = 0.0005) -> Model:
        """
        Build advanced multi-layer LSTM model
        """
        model = Sequential([
            Input(shape=self.input_shape),
            
            # First LSTM layer
            LSTM(lstm_units[0], return_sequences=True, name='lstm_1'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Second LSTM layer
            LSTM(lstm_units[1], return_sequences=True, name='lstm_2'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Third LSTM layer
            LSTM(lstm_units[2], return_sequences=False, name='lstm_3'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            # Dense layers with residual connections
            Dense(64, activation='relu', name='dense_1'),
            Dropout(dropout_rate),
            Dense(32, activation='relu', name='dense_2'),
            Dropout(dropout_rate),
            Dense(16, activation='relu', name='dense_3'),
            Dense(1, activation='linear', name='output')
        ])
        
        # Compile with custom optimizer
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model

class ModelTrainer:
    """
    High-performance model training with optimization
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def train_model(self, model: Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   dataset_name: str, epochs: int = 100,
                   batch_size: int = 32) -> Dict:
        """
        Train model with advanced callbacks and monitoring
        
        Args:
            model: Keras model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            dataset_name: Dataset identifier
            epochs: Maximum epochs
            batch_size: Batch size
            
        Returns:
            Training history and metrics
        """
        # Define callbacks
        model_path = os.path.join(self.model_dir, f'best_model_{dataset_name}.h5')
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        logger.info(f"Starting training for {dataset_name}...")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        best_model = tf.keras.models.load_model(model_path)
        
        # Evaluate on validation set
        val_predictions = best_model.predict(X_val)
        val_metrics = self._calculate_metrics(y_val, val_predictions)
        
        # Save training metadata
        metadata = {
            'dataset': dataset_name,
            'training_time': datetime.now().isoformat(),
            'epochs_trained': len(history.history['loss']),
            'best_val_loss': min(history.history['val_loss']),
            'validation_metrics': val_metrics,
            'model_params': model.count_params()
        }
        
        metadata_path = os.path.join(self.model_dir, f'metadata_{dataset_name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training completed for {dataset_name}")
        logger.info(f"Best validation RMSE: {val_metrics['rmse']:.3f}")
        
        return {
            'history': history.history,
            'model': best_model,
            'metrics': val_metrics,
            'metadata': metadata
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        # PHM08 scoring function
        phm08_score = self._calculate_phm08_score(y_true, y_pred)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mse': float(mse),
            'phm08_score': float(phm08_score)
        }
    
    def _calculate_phm08_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate PHM08 challenge scoring function"""
        diff = y_pred - y_true
        score = 0
        
        for d in diff:
            if d < 0:  # Early prediction
                score += np.exp(-d/10) - 1
            else:  # Late prediction
                score += np.exp(d/13) - 1
        
        return score

class ModelOptimizer:
    """
    Hyperparameter optimization and model selection
    """
    
    def __init__(self):
        self.best_configs = {}
        
    def optimize_hyperparameters(self, data_loader: DataLoader, 
                                dataset_name: str,
                                param_grid: Dict) -> Dict:
        """
        Perform grid search for hyperparameter optimization
        
        Args:
            data_loader: DataLoader instance
            dataset_name: Dataset to optimize on
            param_grid: Parameter grid for search
            
        Returns:
            Best configuration and results
        """
        logger.info(f"Starting hyperparameter optimization for {dataset_name}")
        
        # Load and preprocess data
        train_df, test_df, rul_df = data_loader.load_dataset(dataset_name)
        X_train, y_train, X_test, y_test = data_loader.preprocess_data(train_df, test_df)
        
        # Split training data for validation
        split_idx = int(0.8 * len(X_train))
        X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
        
        best_score = float('inf')
        best_config = None
        results = []
        
        # Grid search
        for lstm_units in param_grid['lstm_units']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    for batch_size in param_grid['batch_size']:
                        
                        config = {
                            'lstm_units': lstm_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size
                        }
                        
                        logger.info(f"Testing config: {config}")
                        
                        try:
                            # Build and train model
                            arch = ModelArchitecture(X_train.shape[1:])
                            model = arch.build_lstm_model(
                                lstm_units=lstm_units,
                                dropout_rate=dropout_rate,
                                learning_rate=learning_rate
                            )
                            
                            trainer = ModelTrainer()
                            result = trainer.train_model(
                                model, X_train_split, y_train_split,
                                X_val, y_val, f"{dataset_name}_opt",
                                epochs=50, batch_size=batch_size
                            )
                            
                            score = result['metrics']['rmse']
                            config['score'] = score
                            config['metrics'] = result['metrics']
                            results.append(config)
                            
                            if score < best_score:
                                best_score = score
                                best_config = config.copy()
                                
                            logger.info(f"Config score: {score:.3f}")
                            
                        except Exception as e:
                            logger.error(f"Error with config {config}: {e}")
                            continue
        
        self.best_configs[dataset_name] = best_config
        
        logger.info(f"Optimization completed. Best RMSE: {best_score:.3f}")
        logger.info(f"Best config: {best_config}")
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results
        }

def main():
    """
    Main training pipeline
    """
    logger.info("Starting ML Core Project - Predictive Maintenance")
    
    # Initialize components
    data_loader = DataLoader()
    
    # Datasets to train
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    
    # Training configuration
    config = {
        'sequence_length': 30,
        'lstm_units': [100, 50],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
    
    results = {}
    
    for dataset in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {dataset}")
        logger.info(f"{'='*50}")
        
        try:
            # Load and preprocess data
            train_df, test_df, rul_df = data_loader.load_dataset(dataset)
            X_train, y_train, X_test, y_test = data_loader.preprocess_data(
                train_df, test_df, config['sequence_length']
            )
            
            # Split for validation
            split_idx = int(0.8 * len(X_train))
            X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
            y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
            
            # Build model
            arch = ModelArchitecture(X_train.shape[1:])
            model = arch.build_lstm_model(
                lstm_units=config['lstm_units'],
                dropout_rate=config['dropout_rate'],
                learning_rate=config['learning_rate']
            )
            
            # Train model
            trainer = ModelTrainer()
            result = trainer.train_model(
                model, X_train_split, y_train_split,
                X_val, y_val, dataset,
                epochs=config['epochs'],
                batch_size=config['batch_size']
            )
            
            results[dataset] = result
            
        except Exception as e:
            logger.error(f"Error training {dataset}: {e}")
            continue
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    for dataset, result in results.items():
        metrics = result['metrics']
        logger.info(f"{dataset}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")
    
    logger.info("\nTraining completed successfully!")
    
    return results

if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    
    # Run main training pipeline
    results = main()