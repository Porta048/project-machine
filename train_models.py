#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Training Script
Run this script to immediately start training optimized models

Usage:
    python train_models.py --dataset FD001 --model basic_lstm
    python train_models.py --all-datasets --optimize
    python train_models.py --dataset FD004 --model advanced_lstm --epochs 150
"""

import argparse
import sys
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_core_project import DataLoader, ModelArchitecture, ModelTrainer, ModelOptimizer
from config import (
    get_config, get_model_config, get_dataset_info, get_optimized_config,
    DATASETS, MODEL_CONFIGS, TRAINING_CONFIG, OPTIMIZATION_GRID
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_gpu():
    """
    Configure GPU settings for optimal performance
    """
    try:
        import tensorflow as tf
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        else:
            logger.info("No GPUs found, using CPU")
            
        # Set mixed precision if available
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled")
        except:
            logger.info("Mixed precision not available")
            
    except ImportError:
        logger.warning("TensorFlow not available")

def train_single_dataset(dataset_name: str, model_name: str = 'basic_lstm', 
                        epochs: int = None, optimize: bool = False) -> Dict:
    """
    Train a model on a single dataset
    
    Args:
        dataset_name: Dataset to train on (FD001, FD002, FD003, FD004)
        model_name: Model architecture to use
        epochs: Number of epochs (None for default)
        optimize: Whether to run hyperparameter optimization
        
    Returns:
        Training results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name} on {dataset_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Initialize data loader
        data_loader = DataLoader()
        
        # Get optimized configuration for dataset
        if optimize:
            logger.info("Running hyperparameter optimization...")
            optimizer = ModelOptimizer()
            
            # Use smaller grid for quick optimization
            quick_grid = {
                'lstm_units': [[64, 32], [100, 50], [128, 64]],
                'dropout_rate': [0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'batch_size': [32, 64]
            }
            
            opt_result = optimizer.optimize_hyperparameters(
                data_loader, dataset_name, quick_grid
            )
            
            best_config = opt_result['best_config']
            logger.info(f"Best configuration found: {best_config}")
            
        else:
            # Use pre-optimized configuration
            optimized_config = get_optimized_config(dataset_name)
            model_name = optimized_config.get('model', model_name)
            
            best_config = {
                'lstm_units': get_model_config(model_name)['lstm_units'],
                'dropout_rate': optimized_config.get('dropout_rate', 0.2),
                'learning_rate': optimized_config.get('learning_rate', 0.001),
                'batch_size': optimized_config.get('batch_size', 32)
            }
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        train_df, test_df, rul_df = data_loader.load_dataset(dataset_name)
        
        sequence_length = get_optimized_config(dataset_name).get('sequence_length', 30)
        X_train, y_train, X_test, y_test = data_loader.preprocess_data(
            train_df, test_df, sequence_length
        )
        
        # Split for validation
        split_idx = int(0.8 * len(X_train))
        X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
        
        # Build model
        logger.info("Building model architecture...")
        arch = ModelArchitecture(X_train.shape[1:])
        
        if model_name == 'advanced_lstm':
            model = arch.build_advanced_model(
                lstm_units=best_config['lstm_units'],
                dropout_rate=best_config['dropout_rate'],
                learning_rate=best_config['learning_rate']
            )
        else:
            model = arch.build_lstm_model(
                lstm_units=best_config['lstm_units'],
                dropout_rate=best_config['dropout_rate'],
                learning_rate=best_config['learning_rate']
            )
        
        # Train model
        logger.info("Starting model training...")
        trainer = ModelTrainer()
        
        training_epochs = epochs or TRAINING_CONFIG['epochs']
        
        result = trainer.train_model(
            model, X_train_split, y_train_split,
            X_val, y_val, dataset_name,
            epochs=training_epochs,
            batch_size=best_config['batch_size']
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        result['training_time'] = training_time
        result['config'] = best_config
        
        logger.info(f"\nTraining completed in {training_time:.2f} seconds")
        logger.info(f"Final metrics: RMSE={result['metrics']['rmse']:.3f}, MAE={result['metrics']['mae']:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error training {dataset_name}: {e}")
        raise

def train_all_datasets(model_name: str = 'basic_lstm', epochs: int = None, 
                      optimize: bool = False) -> Dict:
    """
    Train models on all datasets
    
    Args:
        model_name: Model architecture to use
        epochs: Number of epochs (None for default)
        optimize: Whether to run hyperparameter optimization
        
    Returns:
        Dictionary of results for all datasets
    """
    logger.info("\n" + "="*80)
    logger.info("TRAINING MODELS ON ALL DATASETS")
    logger.info("="*80)
    
    results = {}
    total_start_time = time.time()
    
    for dataset_name in DATASETS.keys():
        try:
            result = train_single_dataset(dataset_name, model_name, epochs, optimize)
            results[dataset_name] = result
        except Exception as e:
            logger.error(f"Failed to train {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
    
    total_time = time.time() - total_start_time
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    for dataset_name, result in results.items():
        if 'error' in result:
            logger.error(f"{dataset_name}: FAILED - {result['error']}")
        else:
            metrics = result['metrics']
            time_taken = result.get('training_time', 0)
            logger.info(
                f"{dataset_name}: RMSE={metrics['rmse']:.3f}, "
                f"MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}, "
                f"Time={time_taken:.1f}s"
            )
    
    logger.info(f"\nTotal training time: {total_time:.2f} seconds")
    
    return results

def quick_benchmark() -> Dict:
    """
    Run a quick benchmark on all datasets with lightweight models
    """
    logger.info("\n" + "="*60)
    logger.info("QUICK BENCHMARK - LIGHTWEIGHT MODELS")
    logger.info("="*60)
    
    results = {}
    
    for dataset_name in DATASETS.keys():
        logger.info(f"\nBenchmarking {dataset_name}...")
        
        try:
            result = train_single_dataset(
                dataset_name, 
                model_name='lightweight_lstm',
                epochs=20,  # Quick training
                optimize=False
            )
            results[dataset_name] = result
            
        except Exception as e:
            logger.error(f"Benchmark failed for {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
    
    return results

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(
        description='Train ML models for predictive maintenance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py --dataset FD001
  python train_models.py --all-datasets --model advanced_lstm
  python train_models.py --dataset FD004 --optimize --epochs 150
  python train_models.py --benchmark
        """
    )
    
    parser.add_argument(
        '--dataset', 
        choices=['FD001', 'FD002', 'FD003', 'FD004'],
        help='Dataset to train on'
    )
    
    parser.add_argument(
        '--all-datasets', 
        action='store_true',
        help='Train on all datasets'
    )
    
    parser.add_argument(
        '--model', 
        choices=list(MODEL_CONFIGS.keys()),
        default='basic_lstm',
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--optimize', 
        action='store_true',
        help='Run hyperparameter optimization'
    )
    
    parser.add_argument(
        '--benchmark', 
        action='store_true',
        help='Run quick benchmark on all datasets'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        default=True,
        help='Enable GPU configuration'
    )
    
    args = parser.parse_args()
    
    # Setup GPU if requested
    if args.gpu:
        setup_gpu()
    
    # Validate arguments
    if not any([args.dataset, args.all_datasets, args.benchmark]):
        parser.error("Must specify --dataset, --all-datasets, or --benchmark")
    
    if args.dataset and args.all_datasets:
        parser.error("Cannot specify both --dataset and --all-datasets")
    
    # Log configuration
    logger.info(f"Starting training at {datetime.now()}")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        if args.benchmark:
            results = quick_benchmark()
        elif args.all_datasets:
            results = train_all_datasets(args.model, args.epochs, args.optimize)
        else:
            results = train_single_dataset(args.dataset, args.model, args.epochs, args.optimize)
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        # Save results
        import json
        results_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()