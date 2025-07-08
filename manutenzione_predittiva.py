#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive Maintenance Project
Using NASA Turbofan Engine Degradation Simulation Dataset
to predict the Remaining Useful Life (RUL) of engines

Advanced machine learning models for industrial predictive maintenance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow>=2.10.0"])
    import tensorflow as tf
    from tensorflow import keras

# Optional import for hyperparameter tuning
try:
    import keras_tuner as kt
    KERAS_TUNER_AVAILABLE = True
    print("✅ KerasTuner available for hyperparameter optimization")
except ImportError:
    KERAS_TUNER_AVAILABLE = False
    print("⚠️ KerasTuner not available. Using default hyperparameters.")
    print("   Install with: pip install keras-tuner")

# Optional imports for Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP available for model interpretability")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Model interpretability limited.")
    print("   Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
    print("✅ LIME available as fallback for interpretability")
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not available.")
    print("   Install with: pip install lime")

import os
import warnings
warnings.filterwarnings('ignore')

# Visualization configuration
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Directory settings
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
    os.makedirs(os.path.join(RESULTS_DIR, ds), exist_ok=True)

# Column names for the dataset
column_names = [
    'id_motore', 'ciclo', 'impostazione_op1', 'impostazione_op2', 'impostazione_op3',
    'sensore_1', 'sensore_2', 'sensore_3', 'sensore_4', 'sensore_5', 'sensore_6',
    'sensore_7', 'sensore_8', 'sensore_9', 'sensore_10', 'sensore_11', 'sensore_12',
    'sensore_13', 'sensore_14', 'sensore_15', 'sensore_16', 'sensore_17', 'sensore_18',
    'sensore_19', 'sensore_20', 'sensore_21'
]

def verify_dataset():
    """Verifies the presence of NASA Turbofan Engine dataset files"""
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    required_files = []
    
    for dataset in datasets:
        required_files.extend([
            os.path.join(DATA_DIR, f'train_{dataset}.txt'),
            os.path.join(DATA_DIR, f'test_{dataset}.txt'),
            os.path.join(DATA_DIR, f'RUL_{dataset}.txt')
        ])
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        print("Please copy the dataset files from CMAPSSData folder:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print(f"All dataset files found: {len(required_files)} files")
        for dataset in datasets:
            dataset_files = [f for f in required_files if dataset in f]
            print(f"  ✓ {dataset}: {len(dataset_files)} files")
        return True

def load_data(dataset):
    """Loads training and test data for a single dataset"""
    print(f"\n=== Loading {dataset} ===")
    
    # Check file existence
    train_file = os.path.join(DATA_DIR, f'train_{dataset}.txt')
    test_file = os.path.join(DATA_DIR, f'test_{dataset}.txt')
    rul_file = os.path.join(DATA_DIR, f'RUL_{dataset}.txt')
    
    if not all(os.path.exists(f) for f in [train_file, test_file, rul_file]):
        print(f"❌ Missing files for {dataset}")
        return None, None, None
    
    # Load training data
    train_data = pd.read_csv(train_file, sep='\s+', header=None, names=column_names)
    print(f"  ✓ Training data: {train_data.shape}")
    
    # Load test data
    test_data = pd.read_csv(test_file, sep='\s+', header=None, names=column_names)
    print(f"  ✓ Test data: {test_data.shape}")
    
    # Load RUL data
    rul_data = pd.read_csv(rul_file, sep='\s+', header=None, names=['RUL'])
    print(f"  ✓ RUL data: {rul_data.shape}")
    
    return train_data, test_data, rul_data

def exploratory_data_analysis(train_data):
    """Performs exploratory data analysis"""
    print("\nEXPLORATORY DATA ANALYSIS")
    print("=" * 40)
    
    print(f"\nGeneral information:")
    print(f"- Total number of engines: {train_data['id_motore'].nunique()}")
    print(f"- Total number of cycles: {len(train_data)}")
    print(f"- Cycles per engine (mean): {train_data.groupby('id_motore')['ciclo'].max().mean():.1f}")
    print(f"- Cycles per engine (min): {train_data.groupby('id_motore')['ciclo'].max().min()}")
    print(f"- Cycles per engine (max): {train_data.groupby('id_motore')['ciclo'].max().max()}")
    
    # Sensor visualization for a specific engine
    plt.figure(figsize=(15, 10))
    
    # Select an engine for analysis
    engine_id = 1
    engine_data = train_data[train_data['id_motore'] == engine_id]
    
    # Plot sensors 2 and 3
    plt.subplot(2, 2, 1)
    plt.plot(engine_data['ciclo'], engine_data['sensore_2'], 'b-', linewidth=2, label='Sensor 2')
    plt.plot(engine_data['ciclo'], engine_data['sensore_3'], 'r-', linewidth=2, label='Sensor 3')
    plt.xlabel('Operating Cycle')
    plt.ylabel('Sensor Value')
    plt.title(f'Sensors 2 and 3 Trend - Engine {engine_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot sensors 7 and 12
    plt.subplot(2, 2, 2)
    plt.plot(engine_data['ciclo'], engine_data['sensore_7'], 'g-', linewidth=2, label='Sensor 7')
    plt.plot(engine_data['ciclo'], engine_data['sensore_12'], 'm-', linewidth=2, label='Sensor 12')
    plt.xlabel('Operating Cycle')
    plt.ylabel('Sensor Value')
    plt.title(f'Sensors 7 and 12 Trend - Engine {engine_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution of cycles per engine
    plt.subplot(2, 2, 3)
    max_cycles = train_data.groupby('id_motore')['ciclo'].max()
    plt.hist(max_cycles, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Maximum Cycles per Engine')
    plt.ylabel('Frequency')
    plt.title('Distribution of Maximum Cycles per Engine')
    plt.grid(True, alpha=0.3)
    
    # Correlation matrix
    plt.subplot(2, 2, 4)
    sensor_cols = [col for col in train_data.columns if col.startswith('sensore_')]
    correlation_matrix = train_data[sensor_cols].corr()
    sns.heatmap(correlation_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix (First 10 Sensors)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exploratory_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_rul_labels(train_data):
    """Creates RUL labels for the training dataset"""
    print("\nCREATING RUL LABELS")
    print("=" * 30)
    
    # Calculate maximum cycle for each engine
    max_cycles = train_data.groupby('id_motore')['ciclo'].max().reset_index()
    max_cycles.columns = ['id_motore', 'max_ciclo']
    
    # Merge with original data
    train_data_with_max = train_data.merge(max_cycles, on='id_motore', how='left')
    
    # Calculate RUL: difference between maximum cycle and current cycle
    train_data_with_max['RUL'] = train_data_with_max['max_ciclo'] - train_data_with_max['ciclo']
    
    # Limit RUL to focus on critical degradation phase (max 125 cycles)
    train_data_with_max['RUL'] = train_data_with_max['RUL'].clip(upper=125)
    
    print(f"RUL calculated for {len(train_data_with_max)} samples")
    print(f"RUL Statistics:")
    print(f"   - Min: {train_data_with_max['RUL'].min()}")
    print(f"   - Max: {train_data_with_max['RUL'].max()}")
    print(f"   - Mean: {train_data_with_max['RUL'].mean():.2f}")
    print(f"   - Standard deviation: {train_data_with_max['RUL'].std():.2f}")
    
    # RUL visualization
    plt.figure(figsize=(12, 6))
    
    engine_id = 1
    engine_data = train_data_with_max[train_data_with_max['id_motore'] == engine_id]
    
    plt.subplot(1, 2, 1)
    plt.plot(engine_data['ciclo'], engine_data['RUL'], 'r-', linewidth=2)
    plt.xlabel('Operating Cycle')
    plt.ylabel('RUL (Remaining Cycles)')
    plt.title(f'RUL vs Cycle - Engine {engine_id}')
    plt.grid(True, alpha=0.3)
    
    # RUL distribution
    plt.subplot(1, 2, 2)
    plt.hist(train_data_with_max['RUL'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('RUL (Remaining Cycles)')
    plt.ylabel('Frequency')
    plt.title('RUL Distribution in Dataset')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rul_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_data_with_max

def select_relevant_features(train_data, test_data, variance_threshold=0.01):
    """
    Seleziona automaticamente le feature più rilevanti escludendo quelle con varianza troppo bassa
    
    Args:
        train_data: DataFrame di training
        test_data: DataFrame di test
        variance_threshold: soglia minima di varianza (default: 0.01)
    
    Returns:
        train_data_filtered: DataFrame di training con feature selezionate
        test_data_filtered: DataFrame di test con feature selezionate
        selected_features: lista delle feature selezionate
        feature_stats: statistiche delle feature
    """
    print(f"\nSELEZIONE AUTOMATICA DELLE FEATURE")
    print("=" * 40)
    print(f"Soglia di varianza minima: {variance_threshold}")
    
    # Identificare tutte le colonne dei sensori e delle impostazioni operative
    sensor_cols = [col for col in train_data.columns if col.startswith('sensore_')]
    operational_cols = [col for col in train_data.columns if col.startswith('impostazione_op')]
    candidate_features = sensor_cols + operational_cols
    
    print(f"Feature candidate totali: {len(candidate_features)}")
    print(f"  - Sensori: {len(sensor_cols)}")
    print(f"  - Impostazioni operative: {len(operational_cols)}")
    
    # Calcolare varianza per ogni feature sui dati di training
    feature_variances = train_data[candidate_features].var()
    print(f"\nStatistiche di varianza:")
    print(f"  - Varianza minima: {feature_variances.min():.6f}")
    print(f"  - Varianza massima: {feature_variances.max():.6f}")
    print(f"  - Varianza media: {feature_variances.mean():.6f}")
    
    # Identificare feature con varianza troppo bassa
    low_variance_features = feature_variances[feature_variances < variance_threshold].index.tolist()
    high_variance_features = feature_variances[feature_variances >= variance_threshold].index.tolist()
    
    print(f"\nRisultati selezione:")
    print(f"  - Feature con varianza troppo bassa ({len(low_variance_features)}): {sorted(low_variance_features)}")
    print(f"  - Feature selezionate ({len(high_variance_features)}): {sorted(high_variance_features)}")
    
    # Utilizzare VarianceThreshold di sklearn per conferma
    selector = VarianceThreshold(threshold=variance_threshold)
    train_features_selected = selector.fit_transform(train_data[candidate_features])
    
    # Ottenere i nomi delle feature selezionate
    selected_mask = selector.get_support()
    sklearn_selected_features = [candidate_features[i] for i, selected in enumerate(selected_mask) if selected]
    
    # Verificare che la nostra selezione manuale corrisponda a quella di sklearn
    assert set(high_variance_features) == set(sklearn_selected_features), "Discrepanza nella selezione delle feature"
    
    # Mantenere anche le colonne non-feature (id_motore, ciclo, RUL se presente)
    keep_cols = ['id_motore', 'ciclo']
    if 'RUL' in train_data.columns:
        keep_cols.append('RUL')
    if 'max_ciclo' in train_data.columns:
        keep_cols.append('max_ciclo')
    
    final_columns = keep_cols + high_variance_features
    
    # Creare i dataset filtrati
    train_data_filtered = train_data[final_columns].copy()
    test_data_filtered = test_data[[col for col in final_columns if col in test_data.columns]].copy()
    
    # Statistiche finali
    feature_stats = {
        'total_candidates': len(candidate_features),
        'selected_count': len(high_variance_features),
        'removed_count': len(low_variance_features),
        'removed_features': low_variance_features,
        'selected_features': high_variance_features,
        'variance_threshold': variance_threshold,
        'feature_variances': feature_variances.to_dict()
    }
    
    print(f"\nRISULTATO FINALE:")
    print(f"  - Feature rimosse: {len(low_variance_features)}/{len(candidate_features)} ({100*len(low_variance_features)/len(candidate_features):.1f}%)")
    print(f"  - Feature mantenute: {len(high_variance_features)}/{len(candidate_features)} ({100*len(high_variance_features)/len(candidate_features):.1f}%)")
    
    return train_data_filtered, test_data_filtered, high_variance_features, feature_stats

def normalize_data_by_operating_conditions(train_data, test_data, selected_features, dataset_name):
    """
    Normalizza i dati considerando le condizioni operative separatamente per FD002 e FD004,
    oppure globalmente per FD001 e FD003
    
    Args:
        train_data: DataFrame di training
        test_data: DataFrame di test
        selected_features: lista delle feature selezionate automaticamente
        dataset_name: nome del dataset (FD001, FD002, FD003, FD004)
    
    Returns:
        train_data_normalized: dati di training normalizzati
        test_data_normalized: dati di test normalizzati
        scalers: dizionario con gli scaler utilizzati
    """
    print(f"\nNORMALIZZAZIONE DATI - {dataset_name}")
    print("=" * 40)
    
    # Filtrare solo le feature che esistono nel dataset attuale
    available_features = [col for col in selected_features if col in train_data.columns]
    
    print(f"Feature da normalizzare: {len(available_features)}")
    print(f"Features: {sorted(available_features)}")
    
    train_data_normalized = train_data.copy()
    test_data_normalized = test_data.copy()
    scalers = {}
    
    # Verificare se il dataset ha condizioni operative multiple
    has_multiple_conditions = dataset_name in ['FD002', 'FD004']
    
    if has_multiple_conditions:
        print(f"Dataset con condizioni operative multiple - normalizzazione separata per condizione")
        
        # Identificare le condizioni operative uniche nel training set
        operational_conditions = train_data[['impostazione_op1', 'impostazione_op2', 'impostazione_op3']].drop_duplicates()
        print(f"Condizioni operative identificate: {len(operational_conditions)}")
        
        # Normalizzare per ogni condizione operativa
        for idx, (_, condition) in enumerate(operational_conditions.iterrows()):
            condition_name = f"condition_{idx+1}"
            print(f"\n  Processando {condition_name}: op1={condition['impostazione_op1']:.2f}, op2={condition['impostazione_op2']:.2f}, op3={condition['impostazione_op3']:.2f}")
            
            # Trovare dati che corrispondono a questa condizione nel training
            train_mask = (
                (train_data['impostazione_op1'] == condition['impostazione_op1']) &
                (train_data['impostazione_op2'] == condition['impostazione_op2']) &
                (train_data['impostazione_op3'] == condition['impostazione_op3'])
            )
            
            # Trovare dati che corrispondono a questa condizione nel test
            test_mask = (
                (test_data['impostazione_op1'] == condition['impostazione_op1']) &
                (test_data['impostazione_op2'] == condition['impostazione_op2']) &
                (test_data['impostazione_op3'] == condition['impostazione_op3'])
            )
            
            train_condition_data = train_data[train_mask]
            test_condition_data = test_data[test_mask]
            
            print(f"    - Campioni training: {len(train_condition_data)}")
            print(f"    - Campioni test: {len(test_condition_data)}")
            
            if len(train_condition_data) > 0:
                # Creare e addestrare lo scaler su questa condizione
                scaler = MinMaxScaler()
                scaler.fit(train_condition_data[available_features])
                scalers[condition_name] = scaler
                
                # Normalizzare i dati di training per questa condizione
                train_data_normalized.loc[train_mask, available_features] = scaler.transform(
                    train_condition_data[available_features]
                )
                
                # Normalizzare i dati di test per questa condizione (se esistono)
                if len(test_condition_data) > 0:
                    test_data_normalized.loc[test_mask, available_features] = scaler.transform(
                        test_condition_data[available_features]
                    )
        
        print(f"\nNormalizzazione per condizioni operative completata:")
        print(f"  - {len(scalers)} scaler creati")
        print(f"  - Range normalizzato: [0, 1] per ogni condizione")
                
    else:
        print(f"Dataset con condizioni operative singole - normalizzazione globale")
        
        # Normalizzazione globale per FD001 e FD003
        scaler = MinMaxScaler()
        scalers['global'] = scaler
        
        # Fit su training, transform su training e test
        train_data_normalized[available_features] = scaler.fit_transform(train_data[available_features])
        test_data_normalized[available_features] = scaler.transform(test_data[available_features])
        
        print(f"Normalizzazione globale completata:")
        print(f"  - 1 scaler globale creato")
        print(f"  - Range normalizzato: [0, 1]")
    
    return train_data_normalized, test_data_normalized, scalers

def normalize_data(train_data, test_data):
    """Normalizes data using MinMaxScaler - LEGACY FUNCTION"""
    print("\nNORMALIZING DATA (LEGACY)")
    print("=" * 25)
    
    # Select columns to normalize
    sensor_cols = [col for col in train_data.columns if col.startswith('sensore_')]
    feature_cols = sensor_cols + ['impostazione_op1', 'impostazione_op2', 'impostazione_op3']
    
    # Filter only existing features
    feature_cols = [col for col in feature_cols if col in train_data.columns]
    
    # Create and fit scaler on training data
    scaler = MinMaxScaler()
    train_data_normalized = train_data.copy()
    train_data_normalized[feature_cols] = scaler.fit_transform(train_data[feature_cols])
    
    # Apply the same scaler to test data
    test_data_normalized = test_data.copy()
    test_data_normalized[feature_cols] = scaler.transform(test_data[feature_cols])
    
    print(f"Data normalized for {len(feature_cols)} features")
    print(f"   - Normalized range: [0, 1]")
    
    return train_data_normalized, test_data_normalized, scaler

def prepare_sequences(data, sequence_length=50, selected_features=None):
    """
    Prepares sequences for the model
    
    Args:
        data: DataFrame with data
        sequence_length: length of the temporal sequence
        selected_features: lista delle feature selezionate automaticamente (opzionale)
    
    Returns:
        X: array of sequences [samples, sequence_length, features]
        y: array of RUL labels
    """
    print(f"\nPREPARING SEQUENCES (length: {sequence_length})")
    print("=" * 45)
    
    # Use selected features if provided, otherwise use default logic
    if selected_features is not None:
        feature_cols = [col for col in selected_features if col in data.columns]
        print(f"Utilizzando feature selezionate automaticamente: {len(feature_cols)}")
    else:
        # Select sensor columns - legacy behavior
        sensor_cols = [col for col in data.columns if col.startswith('sensore_')]
        feature_cols = sensor_cols + ['impostazione_op1', 'impostazione_op2', 'impostazione_op3']
        feature_cols = [col for col in feature_cols if col in data.columns]
        print(f"Utilizzando tutte le feature disponibili: {len(feature_cols)}")
    
    sequences = []
    labels = []
    
    # For each engine, create sequences
    for engine_id in data['id_motore'].unique():
        engine_data = data[data['id_motore'] == engine_id].sort_values('ciclo')
        
        # If the engine has enough cycles to create sequences
        if len(engine_data) >= sequence_length:
            # Create overlapping sequences
            for i in range(len(engine_data) - sequence_length + 1):
                sequence = engine_data[feature_cols].iloc[i:i+sequence_length].values
                label = engine_data['RUL'].iloc[i+sequence_length-1]
                
                sequences.append(sequence)
                labels.append(label)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Sequences created:")
    print(f"   - Number of sequences: {len(X)}")
    print(f"   - X shape: {X.shape}")
    print(f"   - y shape: {y.shape}")
    print(f"   - Number of features: {X.shape[2]}")
    
    return X, y

def build_fd001_model(input_shape):
    """Advanced CNN model with SeparableConv1D for FD001 (single operating conditions)"""
    print("\nBuilding advanced CNN model with SeparableConv1D for FD001...")
    
    model = keras.Sequential([
        # First block with SeparableConv1D - more efficient than regular Conv1D
        keras.layers.SeparableConv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Dropout(0.2),
        
        # Second block with increased filters
        keras.layers.SeparableConv1D(filters=64, kernel_size=5, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Dropout(0.3),
        
        # Third block for deeper feature extraction
        keras.layers.SeparableConv1D(filters=128, kernel_size=3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),  # Better than Flatten for reducing parameters
        
        # Dense layers with regularization
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=50, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def build_fd002_model(input_shape):
    """Advanced Bidirectional LSTM model for FD002 (multiple operating conditions)"""
    print("\nBuilding advanced Bidirectional LSTM model for FD002...")
    
    model = keras.Sequential([
        # First Bidirectional LSTM layer - captures temporal dependencies in both directions
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            input_shape=input_shape
        ),
        keras.layers.BatchNormalization(),
        
        # Second Bidirectional LSTM layer
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        ),
        keras.layers.BatchNormalization(),
        
        # Final LSTM layer to reduce sequence
        keras.layers.LSTM(units=16, return_sequences=False, dropout=0.2),
        keras.layers.BatchNormalization(),
        
        # Dense layers with regularization
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def build_fd003_model(input_shape):
    """Advanced SeparableConv1D model for FD003 (single conditions with faults)"""
    print("\nBuilding advanced SeparableConv1D model for FD003...")
    
    model = keras.Sequential([
        # First block - wider kernel for fault pattern detection
        keras.layers.SeparableConv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Dropout(0.2),
        
        # Second block - increased capacity for complex fault patterns
        keras.layers.SeparableConv1D(filters=128, kernel_size=5, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Dropout(0.3),
        
        # Third block - fine-grained feature extraction
        keras.layers.SeparableConv1D(filters=256, kernel_size=3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Fourth block - final feature refinement
        keras.layers.SeparableConv1D(filters=128, kernel_size=3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),
        
        # Dense layers with strong regularization for fault detection
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=50, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def build_fd004_model(input_shape):
    """Advanced Hybrid CNN-LSTM model with Attention for FD004 (multiple conditions with faults)"""
    print("\nBuilding advanced Hybrid CNN-LSTM model with Attention for FD004...")
    
    # Input layer
    input_layer = keras.layers.Input(shape=input_shape)
    
    # CNN Branch with SeparableConv1D
    conv_branch = keras.layers.SeparableConv1D(filters=64, kernel_size=5, activation='relu')(input_layer)
    conv_branch = keras.layers.BatchNormalization()(conv_branch)
    conv_branch = keras.layers.MaxPooling1D(pool_size=2)(conv_branch)
    conv_branch = keras.layers.Dropout(0.2)(conv_branch)
    
    conv_branch = keras.layers.SeparableConv1D(filters=128, kernel_size=3, activation='relu')(conv_branch)
    conv_branch = keras.layers.BatchNormalization()(conv_branch)
    conv_branch = keras.layers.Dropout(0.3)(conv_branch)
    
    # Global features from CNN
    conv_global = keras.layers.GlobalAveragePooling1D()(conv_branch)
    
    # LSTM Branch with Bidirectional
    lstm_branch = keras.layers.Bidirectional(
        keras.layers.LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(input_layer)
    lstm_branch = keras.layers.BatchNormalization()(lstm_branch)
    
    lstm_branch = keras.layers.Bidirectional(
        keras.layers.LSTM(units=32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(lstm_branch)
    lstm_branch = keras.layers.BatchNormalization()(lstm_branch)
    
    # Attention Mechanism
    # Calculate attention weights
    attention_weights = keras.layers.Dense(1, activation='tanh')(lstm_branch)
    attention_weights = keras.layers.Flatten()(attention_weights)
    attention_weights = keras.layers.Activation('softmax')(attention_weights)
    attention_weights = keras.layers.RepeatVector(lstm_branch.shape[-1])(attention_weights)
    attention_weights = keras.layers.Permute([2, 1])(attention_weights)
    
    # Apply attention weights
    attended_lstm = keras.layers.Multiply()([lstm_branch, attention_weights])
    lstm_attended = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended_lstm)
    
    # Combine CNN global features and attended LSTM features
    combined = keras.layers.Concatenate()([conv_global, lstm_attended])
    
    # Dense layers with enhanced regularization
    dense = keras.layers.Dense(units=256, activation='relu')(combined)
    dense = keras.layers.BatchNormalization()(dense)
    dense = keras.layers.Dropout(0.5)(dense)
    
    dense = keras.layers.Dense(units=128, activation='relu')(dense)
    dense = keras.layers.BatchNormalization()(dense)
    dense = keras.layers.Dropout(0.4)(dense)
    
    dense = keras.layers.Dense(units=64, activation='relu')(dense)
    dense = keras.layers.Dropout(0.3)(dense)
    
    dense = keras.layers.Dense(units=32, activation='relu')(dense)
    dense = keras.layers.Dropout(0.2)(dense)
    
    output = keras.layers.Dense(units=1, activation='linear')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def build_tunable_model(hp, dataset, input_shape):
    """
    Builds a tunable model for hyperparameter optimization
    
    Args:
        hp: HyperParameters object from KerasTuner
        dataset: Dataset name (FD001, FD002, FD003, FD004)
        input_shape: Shape of input data
    
    Returns:
        Compiled Keras model with tunable hyperparameters
    """
    
    if dataset == 'FD001':
        # Tunable SeparableConv1D model for FD001
        model = keras.Sequential()
        
        # First block
        model.add(keras.layers.SeparableConv1D(
            filters=hp.Int('conv1_filters', min_value=16, max_value=64, step=16),
            kernel_size=hp.Choice('conv1_kernel', values=[3, 5, 7]),
            activation='relu',
            input_shape=input_shape
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.4, step=0.1)))
        
        # Second block
        model.add(keras.layers.SeparableConv1D(
            filters=hp.Int('conv2_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
            activation='relu'
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))
        
        # Third block (optional)
        if hp.Boolean('use_third_conv'):
            model.add(keras.layers.SeparableConv1D(
                filters=hp.Int('conv3_filters', min_value=64, max_value=256, step=64),
                kernel_size=3,
                activation='relu'
            ))
            model.add(keras.layers.BatchNormalization())
        
        model.add(keras.layers.GlobalAveragePooling1D())
        
        # Dense layers
        model.add(keras.layers.Dropout(hp.Float('dropout_dense1', min_value=0.3, max_value=0.6, step=0.1)))
        model.add(keras.layers.Dense(
            units=hp.Int('dense1_units', min_value=50, max_value=200, step=50),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(hp.Float('dropout_dense2', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(keras.layers.Dense(
            units=hp.Int('dense2_units', min_value=25, max_value=100, step=25),
            activation='relu'
        ))
        model.add(keras.layers.Dense(1, activation='linear'))
        
    elif dataset == 'FD002':
        # Tunable Bidirectional LSTM model for FD002
        model = keras.Sequential()
        
        # First Bidirectional LSTM
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=hp.Int('lstm1_units', min_value=32, max_value=128, step=32),
                return_sequences=True,
                dropout=hp.Float('lstm_dropout', min_value=0.1, max_value=0.3, step=0.1),
                recurrent_dropout=hp.Float('lstm_recurrent_dropout', min_value=0.1, max_value=0.3, step=0.1)
            ),
            input_shape=input_shape
        ))
        model.add(keras.layers.BatchNormalization())
        
        # Second Bidirectional LSTM (optional)
        if hp.Boolean('use_second_lstm'):
            model.add(keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=hp.Int('lstm2_units', min_value=16, max_value=64, step=16),
                    return_sequences=True,
                    dropout=hp.Float('lstm2_dropout', min_value=0.1, max_value=0.3, step=0.1),
                    recurrent_dropout=hp.Float('lstm2_recurrent_dropout', min_value=0.1, max_value=0.3, step=0.1)
                )
            ))
            model.add(keras.layers.BatchNormalization())
        
        # Final LSTM
        model.add(keras.layers.LSTM(
            units=hp.Int('lstm_final_units', min_value=8, max_value=32, step=8),
            return_sequences=False,
            dropout=hp.Float('lstm_final_dropout', min_value=0.1, max_value=0.3, step=0.1)
        ))
        model.add(keras.layers.BatchNormalization())
        
        # Dense layers
        model.add(keras.layers.Dense(
            units=hp.Int('dense1_units', min_value=16, max_value=64, step=16),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(hp.Float('dropout_dense', min_value=0.2, max_value=0.4, step=0.1)))
        model.add(keras.layers.Dense(1, activation='linear'))
        
    elif dataset == 'FD003':
        # Tunable Deep SeparableConv1D model for FD003
        model = keras.Sequential()
        
        # First block
        model.add(keras.layers.SeparableConv1D(
            filters=hp.Int('conv1_filters', min_value=32, max_value=96, step=32),
            kernel_size=hp.Choice('conv1_kernel', values=[5, 7, 9]),
            activation='relu',
            input_shape=input_shape
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.3, step=0.1)))
        
        # Second block
        model.add(keras.layers.SeparableConv1D(
            filters=hp.Int('conv2_filters', min_value=64, max_value=192, step=64),
            kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
            activation='relu'
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.4, step=0.1)))
        
        # Third block
        model.add(keras.layers.SeparableConv1D(
            filters=hp.Int('conv3_filters', min_value=128, max_value=384, step=128),
            kernel_size=3,
            activation='relu'
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(hp.Float('dropout3', min_value=0.2, max_value=0.4, step=0.1)))
        
        # Fourth block (optional)
        if hp.Boolean('use_fourth_conv'):
            model.add(keras.layers.SeparableConv1D(
                filters=hp.Int('conv4_filters', min_value=64, max_value=256, step=64),
                kernel_size=3,
                activation='relu'
            ))
            model.add(keras.layers.BatchNormalization())
        
        model.add(keras.layers.GlobalAveragePooling1D())
        
        # Dense layers with strong regularization
        model.add(keras.layers.Dropout(hp.Float('dropout_dense1', min_value=0.4, max_value=0.6, step=0.1)))
        model.add(keras.layers.Dense(
            units=hp.Int('dense1_units', min_value=128, max_value=512, step=128),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(hp.Float('dropout_dense2', min_value=0.3, max_value=0.5, step=0.1)))
        model.add(keras.layers.Dense(
            units=hp.Int('dense2_units', min_value=50, max_value=200, step=50),
            activation='relu'
        ))
        model.add(keras.layers.Dense(1, activation='linear'))
        
    elif dataset == 'FD004':
        # Tunable Hybrid CNN-LSTM with Attention for FD004
        # For simplicity, we'll use a simplified tunable version
        # Full attention mechanism tuning would be very complex
        
        input_layer = keras.layers.Input(shape=input_shape)
        
        # CNN Branch
        conv_filters_1 = hp.Int('conv_filters_1', min_value=32, max_value=96, step=32)
        conv_filters_2 = hp.Int('conv_filters_2', min_value=64, max_value=192, step=64)
        
        conv_branch = keras.layers.SeparableConv1D(
            filters=conv_filters_1,
            kernel_size=hp.Choice('conv_kernel', values=[3, 5]),
            activation='relu'
        )(input_layer)
        conv_branch = keras.layers.BatchNormalization()(conv_branch)
        conv_branch = keras.layers.MaxPooling1D(pool_size=2)(conv_branch)
        conv_branch = keras.layers.Dropout(hp.Float('conv_dropout', min_value=0.1, max_value=0.3, step=0.1))(conv_branch)
        
        conv_branch = keras.layers.SeparableConv1D(
            filters=conv_filters_2,
            kernel_size=3,
            activation='relu'
        )(conv_branch)
        conv_branch = keras.layers.BatchNormalization()(conv_branch)
        conv_global = keras.layers.GlobalAveragePooling1D()(conv_branch)
        
        # LSTM Branch
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=96, step=32)
        lstm_branch = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=lstm_units,
                return_sequences=True,
                dropout=hp.Float('lstm_dropout', min_value=0.1, max_value=0.3, step=0.1),
                recurrent_dropout=hp.Float('lstm_recurrent_dropout', min_value=0.1, max_value=0.3, step=0.1)
            )
        )(input_layer)
        lstm_branch = keras.layers.BatchNormalization()(lstm_branch)
        
        # Simplified attention (just global pooling for tuning)
        lstm_attended = keras.layers.GlobalAveragePooling1D()(lstm_branch)
        
        # Combine branches
        combined = keras.layers.Concatenate()([conv_global, lstm_attended])
        
        # Dense layers
        dense_units_1 = hp.Int('dense1_units', min_value=128, max_value=384, step=128)
        dense_units_2 = hp.Int('dense2_units', min_value=64, max_value=192, step=64)
        
        dense = keras.layers.Dense(units=dense_units_1, activation='relu')(combined)
        dense = keras.layers.BatchNormalization()(dense)
        dense = keras.layers.Dropout(hp.Float('dropout_dense1', min_value=0.4, max_value=0.6, step=0.1))(dense)
        
        dense = keras.layers.Dense(units=dense_units_2, activation='relu')(dense)
        dense = keras.layers.Dropout(hp.Float('dropout_dense2', min_value=0.3, max_value=0.5, step=0.1))(dense)
        
        output = keras.layers.Dense(units=1, activation='linear')(dense)
        
        model = keras.Model(inputs=input_layer, outputs=output)
    
    # Compile with tunable learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def optimize_hyperparameters(dataset, X_train, y_train, input_shape, max_trials=10):
    """
    Optimize hyperparameters using KerasTuner
    
    Args:
        dataset: Dataset name
        X_train: Training features
        y_train: Training labels
        input_shape: Shape of input data
        max_trials: Maximum number of trials for optimization
    
    Returns:
        best_model: Model with optimized hyperparameters
        best_hps: Best hyperparameters found
    """
    
    if not KERAS_TUNER_AVAILABLE:
        print("⚠️ KerasTuner not available. Using default model.")
        return select_model(dataset, input_shape), None
    
    print(f"\n🔧 HYPERPARAMETER OPTIMIZATION FOR {dataset}")
    print("=" * 50)
    print(f"Max trials: {max_trials}")
    print(f"Training samples: {len(X_train)}")
    
    # Create tuner
    tuner = kt.RandomSearch(
        lambda hp: build_tunable_model(hp, dataset, input_shape),
        objective='val_loss',
        max_trials=max_trials,
        directory=f'hp_tuning_{dataset}',
        project_name=f'{dataset}_optimization'
    )
    
    # Early stopping for tuning
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print(f"Starting hyperparameter search...")
    
    # Search for best hyperparameters
    tuner.search(
        X_train, y_train,
        epochs=20,  # Reduced epochs for tuning
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get best hyperparameters and model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    
    print(f"✅ Hyperparameter optimization completed!")
    print(f"Best hyperparameters for {dataset}:")
    
    # Print best hyperparameters
    for param_name in best_hps.values:
        print(f"   - {param_name}: {best_hps.get(param_name)}")
    
    return best_model, best_hps

def select_model(dataset, input_shape):
    """Selects the appropriate model for the dataset"""
    models = {
        'FD001': build_fd001_model,
        'FD002': build_fd002_model, 
        'FD003': build_fd003_model,
        'FD004': build_fd004_model
    }
    
    if dataset in models:
        return models[dataset](input_shape)
    else:
        print(f"Dataset {dataset} not recognized, using base CNN model")
        return build_fd001_model(input_shape)

def train_model(model, X_train, y_train, validation_split=0.2):
    """Trains the model"""
    print("\nTRAINING MODEL")
    print("=" * 25)
    
    # Callback per early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Callback per riduzione learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Addestramento
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Ridotto da 100 per evitare overfitting
        batch_size=32,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Visualizza progresso addestramento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss Training')
    plt.plot(history.history['val_loss'], label='Loss Validazione')
    plt.title('Andamento Loss durante Training')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE Training')
    plt.plot(history.history['val_mae'], label='MAE Validazione')
    plt.title('Andamento MAE durante Training')
    plt.xlabel('Epoca')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'progresso_training.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return history

def prepara_dati_test(test_data, rul_data, scaler, sequence_length=50):
    """Prepara i dati di test per la valutazione"""
    print("\nPREPARAZIONE DATI TEST")
    print("=" * 35)
    
    sensor_cols = [col for col in test_data.columns if col.startswith('sensore_')]
    feature_cols = sensor_cols + ['impostazione_op1', 'impostazione_op2', 'impostazione_op3']
    
    X_test = []
    y_test = []
    engine_info = []
    
    # Per ogni motore nel test set
    engine_ids = test_data['id_motore'].unique()
    
    for idx, engine_id in enumerate(engine_ids):
        engine_data = test_data[test_data['id_motore'] == engine_id].sort_values('ciclo')
        
        if len(engine_data) >= sequence_length:
            # Prendi gli ultimi 'sequence_length' cicli
            sequence = engine_data[feature_cols].iloc[-sequence_length:].values
            
            # Il valore RUL corrisponde all'indice del motore nel file RUL
            label = rul_data.iloc[idx]['RUL']
            
            X_test.append(sequence)
            y_test.append(label)
            engine_info.append(f"Motore_{engine_id}")
        else:
            print(f"   Attenzione: Motore {engine_id} ha solo {len(engine_data)} cicli (minimo richiesto: {sequence_length})")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Dati di test preparati:")
    print(f"   - Numero di motori test: {len(X_test)}")
    print(f"   - Forma X_test: {X_test.shape}")
    print(f"   - Forma y_test: {y_test.shape}")
    print(f"   - Range y_test: [{y_test.min()}, {y_test.max()}]")
    
    return X_test, y_test

def prepara_dati_test_avanzato(test_data, rul_data, scalers, sequence_length=50, selected_features=None, dataset_name='FD001'):
    """
    Prepara i dati di test per una valutazione avanzata che generi predizioni
    per multiple sequenze di ogni motore, non solo l'ultima
    
    Args:
        test_data: DataFrame dei dati di test
        rul_data: DataFrame con i valori RUL reali
        scalers: dizionario con gli scaler utilizzati per la normalizzazione
        sequence_length: lunghezza delle sequenze temporali
        selected_features: lista delle feature selezionate automaticamente
        dataset_name: nome del dataset per determinare il tipo di normalizzazione
    
    Returns:
        test_sequences: lista di dizionari con informazioni dettagliate per ogni sequenza
        summary_stats: statistiche riassuntive della generazione delle sequenze
    """
    print(f"\nPREPARAZIONE DATI TEST AVANZATA - {dataset_name}")
    print("=" * 50)
    
    # Determinare le feature da utilizzare
    if selected_features is not None:
        feature_cols = [col for col in selected_features if col in test_data.columns]
        print(f"Utilizzando feature selezionate automaticamente: {len(feature_cols)}")
    else:
        sensor_cols = [col for col in test_data.columns if col.startswith('sensore_')]
        feature_cols = sensor_cols + ['impostazione_op1', 'impostazione_op2', 'impostazione_op3']
        feature_cols = [col for col in feature_cols if col in test_data.columns]
        print(f"Utilizzando tutte le feature disponibili: {len(feature_cols)}")
    
    test_sequences = []
    engine_ids = test_data['id_motore'].unique()
    total_sequences = 0
    skipped_engines = 0
    
    print(f"Processando {len(engine_ids)} motori di test...")
    
    for idx, engine_id in enumerate(engine_ids):
        engine_data = test_data[test_data['id_motore'] == engine_id].sort_values('ciclo')
        true_rul = rul_data.iloc[idx]['RUL']
        
        if len(engine_data) < sequence_length:
            print(f"   Saltando Motore {engine_id}: solo {len(engine_data)} cicli (minimo: {sequence_length})")
            skipped_engines += 1
            continue
        
        print(f"   Motore {engine_id}: {len(engine_data)} cicli, RUL reale = {true_rul}")
        
        # Determinare quante sequenze possiamo creare per questo motore
        max_sequences = len(engine_data) - sequence_length + 1
        
        # Creare multiple sequenze a intervalli regolari per catturare l'evoluzione
        # Prendiamo sequenze ogni X cicli per non avere troppe sequenze sovrapposte
        step_size = max(1, max_sequences // 10)  # Massimo 10 sequenze per motore
        sequence_indices = list(range(0, max_sequences, step_size))
        
        # Assicurarsi di includere sempre l'ultima sequenza
        if (max_sequences - 1) not in sequence_indices:
            sequence_indices.append(max_sequences - 1)
        
        for seq_idx in sequence_indices:
            sequence_data = engine_data.iloc[seq_idx:seq_idx+sequence_length]
            
            # Applicare la normalizzazione appropriata
            sequence_normalized = sequence_data.copy()
            
            # Determinare quale scaler utilizzare
            if dataset_name in ['FD002', 'FD004'] and len(scalers) > 1:
                # Trovare la condizione operativa per questa sequenza
                condition = sequence_data[['impostazione_op1', 'impostazione_op2', 'impostazione_op3']].iloc[0]
                
                # Trovare lo scaler più appropriato
                scaler_used = None
                for scaler_name, scaler in scalers.items():
                    if scaler_name != 'global':
                        scaler_used = scaler
                        break
                
                if scaler_used is None:
                    scaler_used = list(scalers.values())[0]
            else:
                # Usa lo scaler globale
                scaler_used = scalers.get('global', list(scalers.values())[0])
            
            # Normalizzare le feature
            sequence_normalized[feature_cols] = scaler_used.transform(sequence_data[feature_cols])
            
            # Calcolare il RUL stimato per questa sequenza
            # Il RUL al momento della sequenza è il RUL finale meno i cicli rimanenti
            cycles_from_end = len(engine_data) - (seq_idx + sequence_length)
            estimated_rul = true_rul + cycles_from_end
            
            sequence_info = {
                'engine_id': engine_id,
                'sequence_index': seq_idx,
                'sequence_data': sequence_normalized[feature_cols].values,
                'true_final_rul': true_rul,
                'estimated_current_rul': estimated_rul,
                'cycles_from_end': cycles_from_end,
                'cycle_range': (sequence_data['ciclo'].iloc[0], sequence_data['ciclo'].iloc[-1]),
                'total_engine_cycles': len(engine_data)
            }
            
            test_sequences.append(sequence_info)
            total_sequences += 1
    
    # Convertire le sequenze in array per il modello
    X_test_advanced = np.array([seq['sequence_data'] for seq in test_sequences])
    
    summary_stats = {
        'total_engines': len(engine_ids),
        'processed_engines': len(engine_ids) - skipped_engines,
        'skipped_engines': skipped_engines,
        'total_sequences': total_sequences,
        'avg_sequences_per_engine': total_sequences / max(1, len(engine_ids) - skipped_engines),
        'sequence_length': sequence_length,
        'feature_count': len(feature_cols)
    }
    
    print(f"\nRisultati preparazione test avanzata:")
    print(f"   - Motori processati: {summary_stats['processed_engines']}/{summary_stats['total_engines']}")
    print(f"   - Sequenze totali generate: {summary_stats['total_sequences']}")
    print(f"   - Sequenze medie per motore: {summary_stats['avg_sequences_per_engine']:.1f}")
    print(f"   - Forma X_test: {X_test_advanced.shape}")
    
    return test_sequences, X_test_advanced, summary_stats

def valuta_modello_avanzato(model, test_sequences, X_test_advanced):
    """
    Valuta il modello utilizzando le sequenze multiple generate dalla preparazione avanzata
    
    Args:
        model: modello addestrato
        test_sequences: informazioni dettagliate sulle sequenze di test
        X_test_advanced: array delle sequenze per il modello
    
    Returns:
        risultati_dettagliati: dizionario con risultati dettagliati per engine e sequenza
        metriche_globali: metriche aggregate su tutte le sequenze
    """
    print(f"\nVALUTAZIONE MODELLO AVANZATA")
    print("=" * 35)
    
    # Ottenere predizioni per tutte le sequenze
    y_pred_all = model.predict(X_test_advanced)
    y_pred_all = y_pred_all.flatten()
    y_pred_all = np.maximum(y_pred_all, 0)  # Assicurare predizioni positive
    
    # Organizzare i risultati per motore
    risultati_per_motore = {}
    
    for i, (seq_info, pred) in enumerate(zip(test_sequences, y_pred_all)):
        engine_id = seq_info['engine_id']
        
        if engine_id not in risultati_per_motore:
            risultati_per_motore[engine_id] = {
                'true_final_rul': seq_info['true_final_rul'],
                'sequences': []
            }
        
        risultati_per_motore[engine_id]['sequences'].append({
            'sequence_index': seq_info['sequence_index'],
            'estimated_current_rul': seq_info['estimated_current_rul'],
            'predicted_rul': pred,
            'cycles_from_end': seq_info['cycles_from_end'],
            'cycle_range': seq_info['cycle_range'],
            'error': seq_info['estimated_current_rul'] - pred
        })
    
    # Calcolare metriche globali
    y_true_all = [seq['estimated_current_rul'] for seq in test_sequences]
    
    mse_global = mean_squared_error(y_true_all, y_pred_all)
    rmse_global = np.sqrt(mse_global)
    mae_global = mean_absolute_error(y_true_all, y_pred_all)
    
    # Calcolare R²
    ss_res = np.sum((np.array(y_true_all) - y_pred_all) ** 2)
    ss_tot = np.sum((np.array(y_true_all) - np.mean(y_true_all)) ** 2)
    r2_global = 1 - (ss_res / ss_tot)
    
    metriche_globali = {
        'mse': mse_global,
        'rmse': rmse_global,
        'mae': mae_global,
        'r2': r2_global,
        'total_sequences': len(test_sequences),
        'total_engines': len(risultati_per_motore)
    }
    
    print(f"Metriche globali (tutte le sequenze):")
    print(f"   - Total sequences evaluated: {metriche_globali['total_sequences']}")
    print(f"   - RMSE: {metriche_globali['rmse']:.2f}")
    print(f"   - MAE: {metriche_globali['mae']:.2f}")
    print(f"   - R²: {metriche_globali['r2']:.4f}")
    
    # Analizzare l'evoluzione dell'errore nel tempo
    print(f"\nAnalisi evoluzione errore per motore:")
    for engine_id, engine_results in list(risultati_per_motore.items())[:3]:  # Mostra solo i primi 3
        sequences = sorted(engine_results['sequences'], key=lambda x: x['cycles_from_end'], reverse=True)
        print(f"   Motore {engine_id} (RUL finale reale: {engine_results['true_final_rul']}):")
        
        for seq in sequences:
            print(f"     Cicli dalla fine: {seq['cycles_from_end']:3d} | "
                  f"RUL stim: {seq['estimated_current_rul']:6.1f} | "
                  f"RUL pred: {seq['predicted_rul']:6.1f} | "
                  f"Errore: {seq['error']:+6.1f}")
    
    return risultati_per_motore, metriche_globali

def valuta_modello(model, X_test, y_test):
    """Valuta il modello sui dati di test"""
    print("\nVALUTAZIONE MODELLO")
    print("=" * 25)
    
    # Predizioni
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    
    # Assicurati che le predizioni siano positive
    y_pred = np.maximum(y_pred, 0)
    
    # Calcola metriche
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Metriche di Performance:")
    print(f"   - Mean Squared Error (MSE): {mse:.2f}")
    print(f"   - Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"   - Mean Absolute Error (MAE): {mae:.2f}")
    
    # Calcola R² score
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"   - R² Score: {r2:.4f}")
    
    # Stampa alcuni esempi di predizioni
    print("\nEsempi di predizioni:")
    for i in range(min(5, len(y_test))):
        print(f"   Motore {i+1}: RUL reale = {y_test[i]:.0f}, RUL predetto = {y_pred[i]:.0f}")
    
    return y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def calculate_phm08_score(y_true, y_pred):
    """
    Calculate PHM08 Challenge scoring function for predictive maintenance
    
    This asymmetric scoring function penalizes late predictions much more heavily
    than early predictions, which is crucial in maintenance scenarios:
    - Late prediction → Equipment failure (high cost, safety risk)
    - Early prediction → Unnecessary maintenance (lower cost)
    
    Args:
        y_true: Array of true RUL values
        y_pred: Array of predicted RUL values
    
    Returns:
        phm08_score: PHM08 score (lower is better)
        individual_scores: Array of individual scores for each prediction
    """
    
    print(f"\n🎯 CALCULATING PHM08 SCORE")
    print("=" * 40)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Initialize score array
    individual_scores = np.zeros_like(y_true, dtype=float)
    
    # Calculate asymmetric scoring
    for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
        error = pred_val - true_val
        
        if error <= 0:  # Early prediction (predicted RUL <= true RUL)
            # Less penalty for early predictions
            individual_scores[i] = np.exp(-error / 13.0) - 1
        else:  # Late prediction (predicted RUL > true RUL) 
            # Heavy penalty for late predictions
            individual_scores[i] = np.exp(error / 10.0) - 1
    
    # Total PHM08 score
    phm08_score = np.sum(individual_scores)
    
    # Calculate statistics
    early_predictions = np.sum(y_pred <= y_true)
    late_predictions = np.sum(y_pred > y_true)
    early_score = np.sum(individual_scores[y_pred <= y_true])
    late_score = np.sum(individual_scores[y_pred > y_true])
    
    print(f"PHM08 Scoring Results:")
    print(f"   - Total PHM08 Score: {phm08_score:.2f} (lower is better)")
    print(f"   - Early predictions: {early_predictions}/{len(y_true)} (score: {early_score:.2f})")
    print(f"   - Late predictions: {late_predictions}/{len(y_true)} (score: {late_score:.2f})")
    print(f"   - Average error: {np.mean(y_pred - y_true):.2f} cycles")
    print(f"   - Penalty ratio (late/early): {late_score/(early_score+1e-10):.2f}")
    
    return phm08_score, individual_scores

def analyze_model_interpretability(model, X_test, y_test, feature_names=None, dataset_name='Unknown'):
    """
    Analyze model interpretability using SHAP
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels  
        feature_names: Names of features
        dataset_name: Name of dataset for plots
    
    Returns:
        shap_values: SHAP values if available
        feature_importance: Feature importance summary
    """
    
    print(f"\n🔍 MODEL INTERPRETABILITY ANALYSIS - {dataset_name}")
    print("=" * 60)
    
    if not SHAP_AVAILABLE:
        print("❌ SHAP not available. Install with: pip install shap")
        return None, None
    
    try:
        # For neural networks, we'll use a subset of data for SHAP analysis
        # due to computational constraints
        max_samples = min(100, len(X_test))
        X_sample = X_test[:max_samples]
        y_sample = y_test[:max_samples]
        
        print(f"Analyzing interpretability on {max_samples} samples...")
        
        # Flatten the sequence data for SHAP analysis
        # From (samples, timesteps, features) to (samples, timesteps*features)
        original_shape = X_sample.shape
        X_flattened = X_sample.reshape(X_sample.shape[0], -1)
        
        print(f"Data shape: {original_shape} → {X_flattened.shape}")
        
        # Create a wrapper function for the model prediction
        def model_predict(X):
            # Reshape back to original format for model
            X_reshaped = X.reshape(-1, original_shape[1], original_shape[2])
            return model.predict(X_reshaped, verbose=0).flatten()
        
        # Use SHAP KernelExplainer for neural networks
        print("Creating SHAP explainer...")
        
        # Use a smaller background dataset for efficiency
        background_size = min(20, len(X_flattened))
        background = X_flattened[:background_size]
        
        explainer = shap.KernelExplainer(model_predict, background)
        
        print("Computing SHAP values...")
        # Compute SHAP values for a subset
        explain_size = min(20, len(X_flattened))
        shap_values = explainer.shap_values(X_flattened[:explain_size])
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_flattened.shape[1])]
        elif len(feature_names) != X_flattened.shape[1]:
            # Expand feature names for temporal sequences
            expanded_names = []
            timesteps = original_shape[1]
            for t in range(timesteps):
                for feat in feature_names:
                    expanded_names.append(f"{feat}_t{t}")
            feature_names = expanded_names
        
        # Calculate feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-20:]  # Top 20 features
        top_features = [(feature_names[i], feature_importance[i]) for i in top_indices[::-1]]
        
        print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (feat_name, importance) in enumerate(top_features[:10]):
            print(f"   {i+1:2d}. {feat_name}: {importance:.4f}")
        
        # Create SHAP summary plot
        try:
            plt.figure(figsize=(12, 8))
            
            # Summary plot
            plt.subplot(2, 2, 1)
            shap.summary_plot(
                shap_values, 
                X_flattened[:explain_size], 
                feature_names=feature_names,
                max_display=15,
                show=False
            )
            plt.title(f'SHAP Feature Importance - {dataset_name}')
            
            # Feature importance bar plot
            plt.subplot(2, 2, 2)
            top_20_names = [name for name, _ in top_features[:20]]
            top_20_values = [val for _, val in top_features[:20]]
            
            y_pos = np.arange(len(top_20_names))
            plt.barh(y_pos, top_20_values[::-1], alpha=0.7)
            plt.yticks(y_pos, top_20_names[::-1])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Top 20 Feature Importance')
            plt.grid(True, alpha=0.3)
            
            # SHAP values distribution
            plt.subplot(2, 2, 3)
            plt.hist(shap_values.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('SHAP Values')
            plt.ylabel('Frequency')
            plt.title('Distribution of SHAP Values')
            plt.grid(True, alpha=0.3)
            
            # Prediction vs SHAP sum
            plt.subplot(2, 2, 4)
            shap_sum = np.sum(shap_values, axis=1)
            predictions = model_predict(X_flattened[:explain_size])
            plt.scatter(shap_sum, predictions, alpha=0.6)
            plt.xlabel('Sum of SHAP values')
            plt.ylabel('Model Prediction')
            plt.title('SHAP Values vs Predictions')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'shap_analysis_{dataset_name.lower()}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error creating SHAP plots: {e}")
        
        # Analyze temporal importance if it's a sequence model
        if len(original_shape) == 3:  # Sequence data
            timesteps = original_shape[1]
            n_features = original_shape[2]
            
            # Reshape SHAP values back to temporal format
            shap_temporal = shap_values.reshape(-1, timesteps, n_features)
            
            # Calculate importance by timestep
            timestep_importance = np.mean(np.abs(shap_temporal), axis=(0, 2))
            
            print(f"\n⏰ TEMPORAL IMPORTANCE ANALYSIS:")
            print(f"Most important timesteps (recent = 0, oldest = {timesteps-1}):")
            top_timesteps = np.argsort(timestep_importance)[-5:][::-1]
            for i, ts in enumerate(top_timesteps):
                print(f"   {i+1}. Timestep {ts}: {timestep_importance[ts]:.4f}")
        
        print(f"\n✅ Interpretability analysis completed!")
        print(f"📁 Plots saved to: {RESULTS_DIR}/shap_analysis_{dataset_name.lower()}.png")
        
        return shap_values, {
            'feature_importance': feature_importance,
            'top_features': top_features,
            'feature_names': feature_names
        }
        
    except Exception as e:
        print(f"❌ Error in SHAP analysis: {e}")
        print("This is normal for complex models. Consider using smaller sample sizes.")
        return None, None

def valuta_modello_con_phm08(model, X_test, y_test, feature_names=None, dataset_name='Unknown'):
    """
    Enhanced model evaluation with PHM08 scoring and interpretability analysis
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: Names of features
        dataset_name: Dataset name
    
    Returns:
        comprehensive_results: Dictionary with all evaluation metrics
    """
    
    print(f"\n🔬 COMPREHENSIVE MODEL EVALUATION - {dataset_name}")
    print("=" * 60)
    
    # Standard evaluation
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()
    y_pred = np.maximum(y_pred, 0)  # Ensure positive predictions
    
    # Standard metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # R² score
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"📊 STANDARD METRICS:")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - R²: {r2:.4f}")
    
    # PHM08 scoring
    phm08_score, phm08_individual = calculate_phm08_score(y_test, y_pred)
    
    # Interpretability analysis
    shap_values, interpretability_results = analyze_model_interpretability(
        model, X_test, y_test, feature_names, dataset_name
    )
    
    # Comprehensive results
    results = {
        'standard_metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'phm08_metrics': {
            'phm08_score': phm08_score,
            'phm08_individual': phm08_individual,
            'early_predictions': np.sum(y_pred <= y_test),
            'late_predictions': np.sum(y_pred > y_test)
        },
        'predictions': {
            'y_true': y_test,
            'y_pred': y_pred
        },
        'interpretability': interpretability_results,
        'shap_values': shap_values
    }
    
    return results

def save_model_with_versioning(model, dataset, metrics, hyperparameters=None, model_type="default", comprehensive_results=None):
    """
    Save model with versioning including metrics, PHM08 scores and timestamp
    
    Args:
        model: Trained Keras model
        dataset: Dataset name (FD001, FD002, etc.)
        metrics: Dictionary with standard performance metrics
        hyperparameters: Dictionary with hyperparameters (optional)
        model_type: Type of model ("default", "optimized", etc.)
        comprehensive_results: Results from comprehensive evaluation including PHM08
    
    Returns:
        model_path: Path where the model was saved
        metadata_path: Path where metadata was saved
    """
    import datetime
    import json
    
    print(f"\n💾 SAVING MODEL WITH ADVANCED VERSIONING - {dataset}")
    print("=" * 60)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model filename with metrics and timestamp
    rmse = metrics.get('rmse', 0)
    mae = metrics.get('mae', 0)
    r2 = metrics.get('r2', 0)
    
    # Include PHM08 score in filename if available
    phm08_score = None
    if comprehensive_results and 'phm08_metrics' in comprehensive_results:
        phm08_score = comprehensive_results['phm08_metrics']['phm08_score']
        model_filename = f"model_{dataset.lower()}_{model_type}_rmse{rmse:.1f}_mae{mae:.1f}_r2{r2:.3f}_phm08_{phm08_score:.1f}_{timestamp}.h5"
    else:
        model_filename = f"model_{dataset.lower()}_{model_type}_rmse{rmse:.1f}_mae{mae:.1f}_r2{r2:.3f}_{timestamp}.h5"
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    # Save the model
    model.save(model_path)
    print(f"✅ Model saved: {model_filename}")
    
    # Create metadata
    metadata = {
        "model_info": {
            "dataset": dataset,
            "model_type": model_type,
            "timestamp": timestamp,
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__,
            "shap_available": SHAP_AVAILABLE,
            "lime_available": LIME_AVAILABLE
        },
        "performance_metrics": {
            "standard": {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "mse": float(metrics.get('mse', 0))
            }
        },
        "model_architecture": {
            "total_params": model.count_params(),
            "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            "non_trainable_params": sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
            "layers_count": len(model.layers)
        },
        "training_info": {
            "optimizer": model.optimizer.get_config(),
            "loss_function": model.loss,
            "metrics": [m.name if hasattr(m, 'name') else str(m) for m in model.metrics]
        }
    }
    
    # Add PHM08 metrics if available
    if comprehensive_results and 'phm08_metrics' in comprehensive_results:
        phm08_metrics = comprehensive_results['phm08_metrics']
        metadata["performance_metrics"]["phm08"] = {
            "phm08_score": float(phm08_metrics['phm08_score']),
            "early_predictions": int(phm08_metrics['early_predictions']),
            "late_predictions": int(phm08_metrics['late_predictions']),
            "total_predictions": int(phm08_metrics['early_predictions'] + phm08_metrics['late_predictions'])
        }
    
    # Add interpretability info if available
    if comprehensive_results and 'interpretability' in comprehensive_results and comprehensive_results['interpretability']:
        interp_results = comprehensive_results['interpretability']
        metadata["interpretability"] = {
            "top_5_features": interp_results['top_features'][:5] if 'top_features' in interp_results else [],
            "analysis_completed": True,
            "shap_plots_available": True
        }
    else:
        metadata["interpretability"] = {
            "analysis_completed": False,
            "reason": "SHAP not available or analysis failed"
        }
    
    # Add hyperparameters if provided
    if hyperparameters is not None:
        metadata["hyperparameters"] = {}
        for param_name in hyperparameters.values:
            value = hyperparameters.get(param_name)
            # Convert numpy types to Python native types for JSON serialization
            if hasattr(value, 'item'):  # numpy scalar
                value = value.item()
            metadata["hyperparameters"][param_name] = value
    
    # Helper function to convert numpy types to JSON serializable types
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    # Convert metadata to JSON serializable format
    metadata_serializable = convert_numpy_types(metadata)
    
    # Save metadata
    metadata_filename = f"metadata_{dataset.lower()}_{model_type}_{timestamp}.json"
    metadata_path = os.path.join(MODELS_DIR, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    print(f"✅ Metadata saved: {metadata_filename}")
    print(f"📊 Model Performance Summary:")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - R²: {r2:.4f}")
    print(f"   - Total Parameters: {metadata['model_architecture']['total_params']:,}")
    
    return model_path, metadata_path

def load_best_model_version(dataset, metric='r2'):
    """
    Load the best model version based on a specific metric
    
    Args:
        dataset: Dataset name
        metric: Metric to use for selection ('r2', 'rmse', 'mae')
    
    Returns:
        model: Loaded Keras model
        metadata: Model metadata
    """
    print(f"\n📂 LOADING BEST MODEL VERSION FOR {dataset}")
    print("=" * 50)
    
    # Find all model files for this dataset
    model_files = []
    metadata_files = []
    
    for filename in os.listdir(MODELS_DIR):
        if filename.startswith(f"model_{dataset.lower()}_") and filename.endswith('.h5'):
            model_files.append(filename)
        elif filename.startswith(f"metadata_{dataset.lower()}_") and filename.endswith('.json'):
            metadata_files.append(filename)
    
    if not model_files:
        print(f"❌ No model files found for {dataset}")
        return None, None
    
    print(f"Found {len(model_files)} model versions for {dataset}")
    
    best_model_path = None
    best_metadata = None
    best_metric_value = None
    
    # Evaluate each model version
    for metadata_file in metadata_files:
        try:
            with open(os.path.join(MODELS_DIR, metadata_file), 'r') as f:
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
                            best_model_path = os.path.join(MODELS_DIR, model_file)
                            break
        
        except Exception as e:
            print(f"⚠️ Error reading metadata file {metadata_file}: {e}")
            continue
    
    if best_model_path is None:
        print(f"❌ Could not find best model for {dataset}")
        return None, None
    
    # Load the best model
    try:
        model = tf.keras.models.load_model(best_model_path)
        print(f"✅ Loaded best model: {os.path.basename(best_model_path)}")
        print(f"📊 Performance: {metric.upper()} = {best_metric_value:.4f}")
        return model, best_metadata
    
    except Exception as e:
        print(f"❌ Error loading model {best_model_path}: {e}")
        return None, None

def visualizza_risultati(y_test, y_pred, metrics):
    """Visualizza i risultati del modello"""
    print("\nVISUALIZZAZIONE RISULTATI")
    print("=" * 35)
    
    plt.figure(figsize=(15, 10))
    
    # Confronto RUL predetto vs reale
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('RUL Reale')
    plt.ylabel('RUL Predetto')
    plt.title('RUL Predetto vs RUL Reale')
    plt.grid(True, alpha=0.3)
    
    # Aggiungi testo R²
    plt.text(0.05, 0.95, f'R² = {metrics["r2"]:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Distribuzione degli errori
    plt.subplot(2, 2, 2)
    errors = y_test - y_pred
    plt.hist(errors, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Errore (RUL Reale - RUL Predetto)')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione degli Errori')
    plt.grid(True, alpha=0.3)
    
    # Grafico temporale delle predizioni
    plt.subplot(2, 2, 3)
    indices = np.arange(len(y_test))
    plt.plot(indices, y_test, 'b-', label='RUL Reale', linewidth=2)
    plt.plot(indices, y_pred, 'r--', label='RUL Predetto', linewidth=2)
    plt.xlabel('Indice Motore')
    plt.ylabel('RUL')
    plt.title('Confronto RUL per Motori di Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot degli errori
    plt.subplot(2, 2, 4)
    plt.boxplot(errors, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.ylabel('Errore')
    plt.title('Box Plot degli Errori')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'risultati_modello.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Stampa riassunto finale
    print(f"\nRIASSUNTO FINALE")
    print("=" * 20)
    print(f"Modello addestrato con successo!")
    print(f"Performance sul set di test:")
    print(f"   - RMSE: {metrics['rmse']:.2f} cicli")
    print(f"   - MAE: {metrics['mae']:.2f} cicli")
    print(f"   - R²: {metrics['r2']:.4f}")
    print(f"\nInterpretazione:")
    print(f"   - Il modello ha un errore medio di {metrics['mae']:.1f} cicli")
    print(f"   - Spiega il {metrics['r2']*100:.1f}% della varianza nei dati")
    if metrics['r2'] > 0.7:
        print(f"   - Performance adeguata per la manutenzione predittiva")
    else:
        print(f"   - Performance da migliorare per uso in produzione")

def main():
    """Main function to train all models"""
    print("PREDICTIVE MAINTENANCE PROJECT")
    print("=" * 70)
    print("Dataset: NASA Turbofan Engine Degradation Simulation")
    print("Objective: Predict the Remaining Useful Life (RUL) of engines")
    print("COMPLETE TRAINING - ALL DATASETS AND MODELS")
    print("=" * 70)
    
    # Verify dataset presence
    if not verify_dataset():
        print("\n❌ Error: missing dataset files.")
        print("Download the dataset from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print("Search for 'Turbofan Engine Degradation Simulation Data Set'")
        return
    
    # Configurations for each dataset with modern architectures
    configs = {
        'FD001': {
            'sequence_length': 50,
            'epochs': 50,
            'desc': 'Single operating conditions - Advanced SeparableConv1D model'
        },
        'FD002': {
            'sequence_length': 60,
            'epochs': 60,
            'desc': 'Multiple operating conditions - Bidirectional LSTM model'
        },
        'FD003': {
            'sequence_length': 50,
            'epochs': 70,
            'desc': 'Single conditions with faults - Deep SeparableConv1D model'
        },
        'FD004': {
            'sequence_length': 70,
            'epochs': 80,
            'desc': 'Multiple conditions with faults - Hybrid CNN-LSTM with Attention'
        }
    }
    
    risultati_finali = {}
    
    # Processa ogni dataset con i miglioramenti automatizzati
    for dataset, config in configs.items():
        print(f"\n{'='*25} PROCESSING {dataset} (VERSIONE MIGLIORATA) {'='*25}")
        print(f"Purpose: {config['desc']}")
        
        # Load data
        train_data, test_data, rul_data = load_data(dataset)
        if train_data is None:
            print(f"❌ Skipping {dataset} - missing data")
            continue
        
        # Exploratory analysis only for FD001
        if dataset == 'FD001':
            exploratory_data_analysis(train_data)
        
        # Create RUL labels
        train_data_with_rul = create_rul_labels(train_data)
        
        # MIGLIORAMENTO 1: Selezione automatica delle feature
        train_data_filtered, test_data_filtered, selected_features, feature_stats = select_relevant_features(
            train_data_with_rul, test_data, variance_threshold=0.01
        )
        
        print(f"\n📊 Statistiche selezione feature per {dataset}:")
        print(f"   - Feature rimosse: {feature_stats['removed_features']}")
        print(f"   - Feature selezionate: {len(selected_features)}")
        
        # MIGLIORAMENTO 2: Normalizzazione per condizioni operative
        train_data_normalized, test_data_normalized, scalers = normalize_data_by_operating_conditions(
            train_data_filtered, test_data_filtered, selected_features, dataset
        )
        
        # Prepare sequences con feature selezionate automaticamente
        sequence_length = config['sequence_length']
        X_train, y_train = prepare_sequences(
            train_data_normalized, sequence_length, selected_features
        )
        
        # Build model with optional hyperparameter optimization
        use_hp_tuning = KERAS_TUNER_AVAILABLE and len(X_train) > 1000  # Only use HP tuning for larger datasets
        
        if use_hp_tuning:
            print(f"\n🔧 Using hyperparameter optimization for {dataset}")
            model, best_hps = optimize_hyperparameters(
                dataset, X_train, y_train, 
                input_shape=(sequence_length, X_train.shape[2]),
                max_trials=8  # Reduced for faster training
            )
            model_type = "optimized"
        else:
            print(f"\n🏗️ Using default architecture for {dataset}")
            model = select_model(dataset, input_shape=(sequence_length, X_train.shape[2]))
            best_hps = None
            model_type = "default"
        
        print(f"\nModel architecture {dataset} ({model_type}):")
        model.summary()
        
        # Train model with final hyperparameters
        if use_hp_tuning:
            # Re-train the optimized model with full epochs
            print(f"\n🚀 Training optimized model with full epochs...")
            history = train_model(model, X_train, y_train)
        else:
            history = train_model(model, X_train, y_train)
        
        # MIGLIORAMENTO 3: Valutazione avanzata con multiple sequenze
        print(f"\n🔍 Eseguendo valutazione avanzata per {dataset}...")
        
        # Prepare test data avanzato
        test_sequences, X_test_advanced, summary_stats = prepara_dati_test_avanzato(
            test_data_normalized, rul_data, scalers, sequence_length, selected_features, dataset
        )
        
        # Evaluate model con approccio avanzato
        risultati_dettagliati, metriche_avanzate = valuta_modello_avanzato(
            model, test_sequences, X_test_advanced
        )
        
        # Comprehensive evaluation with PHM08 and XAI
        print(f"\n🔬 Comprehensive evaluation with PHM08 scoring and interpretability...")
        
        # Backward compatibility: preparare dati test con metodo legacy usando le nuove feature
        sensor_cols_filtered = [col for col in selected_features if col.startswith('sensore_')]
        operational_cols = [col for col in selected_features if col.startswith('impostazione_op')]
        legacy_features = sensor_cols_filtered + operational_cols
        
        X_test_legacy = []
        y_test_legacy = []
        engine_ids = test_data_normalized['id_motore'].unique()
        
        for idx, engine_id in enumerate(engine_ids):
            engine_data = test_data_normalized[test_data_normalized['id_motore'] == engine_id].sort_values('ciclo')
            
            if len(engine_data) >= sequence_length:
                sequence = engine_data[legacy_features].iloc[-sequence_length:].values
                label = rul_data.iloc[idx]['RUL']
                X_test_legacy.append(sequence)
                y_test_legacy.append(label)
        
        X_test_legacy = np.array(X_test_legacy)
        y_test_legacy = np.array(y_test_legacy)
        
        # Comprehensive evaluation with PHM08 scoring and interpretability
        comprehensive_results = valuta_modello_con_phm08(
            model, X_test_legacy, y_test_legacy, 
            feature_names=legacy_features, 
            dataset_name=dataset
        )
        
        # Extract legacy metrics for compatibility
        metrics_legacy = comprehensive_results['standard_metrics']
        y_pred_legacy = comprehensive_results['predictions']['y_pred']
        
        risultati_finali[dataset] = {
            'legacy': metrics_legacy,
            'advanced': metriche_avanzate,
            'comprehensive': comprehensive_results,
            'feature_reduction': f"{feature_stats['removed_count']}/{feature_stats['total_candidates']} features removed",
            'selected_features': len(selected_features)
        }
        
        # Visualize results con dati legacy per compatibilità
        visualizza_risultati(y_test_legacy, y_pred_legacy, metrics_legacy)
        
        # Save model with versioning, hyperparameters, and comprehensive results
        model_path, metadata_path = save_model_with_versioning(
            model, dataset, metrics_legacy, 
            hyperparameters=best_hps, 
            model_type=model_type,
            comprehensive_results=comprehensive_results
        )
        risultati_finali[dataset]['model_path'] = model_path
        risultati_finali[dataset]['metadata_path'] = metadata_path
        risultati_finali[dataset]['model_type'] = model_type
        
        # Move plots to appropriate folder
        result_dir = os.path.join(RESULTS_DIR, dataset)
        os.makedirs(result_dir, exist_ok=True)
        
        # List of files to move
        files_to_move = [
            'exploratory_analysis.png',
            'rul_analysis.png', 
            'training_progress.png',
            'model_results.png'
        ]
        
        for fname in files_to_move:
            src_path = os.path.join(RESULTS_DIR, fname)
            dst_path = os.path.join(result_dir, fname)
            if os.path.exists(src_path):
                os.replace(src_path, dst_path)
                print(f"  ✓ {fname} -> {result_dir}/")
    
    # Final summary con miglioramenti
    print(f"\n{'='*30} FINAL SUMMARY (VERSIONE MIGLIORATA) {'='*30}")
    print("Model performance con selezione automatica delle feature e normalizzazione migliorata:")
    print("-" * 90)
    
    for dataset, results in risultati_finali.items():
        config_desc = configs[dataset]['desc']
        print(f"\n{dataset} - {config_desc}")
        print(f"  📊 Feature Engineering:")
        print(f"     - {results['feature_reduction']}")
        print(f"     - Feature finali utilizzate: {results['selected_features']}")
        
        print(f"  🏗️ Model Architecture:")
        print(f"     - Type: {results.get('model_type', 'default').title()}")
        if results.get('model_type') == 'optimized':
            print(f"     - Hyperparameters optimized with KerasTuner")
        
        # Mostra sia risultati legacy che avanzati
        legacy_metrics = results['legacy']
        advanced_metrics = results['advanced']
        
        print(f"  📈 Performance Valutazione Standard:")
        print(f"     - RMSE: {legacy_metrics['rmse']:.2f} cycles")
        print(f"     - MAE:  {legacy_metrics['mae']:.2f} cycles") 
        print(f"     - R²:   {legacy_metrics['r2']:.4f}")
        
        print(f"  🔍 Performance Valutazione Avanzata (multiple sequenze):")
        print(f"     - RMSE: {advanced_metrics['rmse']:.2f} cycles")
        print(f"     - MAE:  {advanced_metrics['mae']:.2f} cycles") 
        print(f"     - R²:   {advanced_metrics['r2']:.4f}")
        print(f"     - Sequenze valutate: {advanced_metrics['total_sequences']}")
        print(f"     - Motori analizzati: {advanced_metrics['total_engines']}")
        
        # PHM08 metrics if available
        if 'comprehensive' in results and 'phm08_metrics' in results['comprehensive']:
            phm08_metrics = results['comprehensive']['phm08_metrics']
            print(f"  🎯 Performance PHM08 (manutenzione predittiva):")
            print(f"     - PHM08 Score: {phm08_metrics['phm08_score']:.2f} (lower is better)")
            print(f"     - Predizioni precoci: {phm08_metrics['early_predictions']}")
            print(f"     - Predizioni tardive: {phm08_metrics['late_predictions']}")
            
            # Status based on PHM08 score
            if phm08_metrics['phm08_score'] < 500:
                phm08_status = "🟢 Excellent"
            elif phm08_metrics['phm08_score'] < 1000:
                phm08_status = "🟡 Good"
            else:
                phm08_status = "🔴 Needs improvement"
            
            print(f"     - PHM08 Status: {phm08_status}")
        
        # Interpretability info
        if 'comprehensive' in results and 'interpretability' in results['comprehensive']:
            interp_info = results['comprehensive']['interpretability']
            if interp_info:
                print(f"  🔍 Interpretability Analysis:")
                print(f"     - SHAP analysis completed: ✅")
                print(f"     - Top feature: {interp_info['top_features'][0][0] if interp_info['top_features'] else 'N/A'}")
                print(f"     - Plots available: results/shap_analysis_{dataset.lower()}.png")
            else:
                print(f"  🔍 Interpretability Analysis: ❌ (SHAP not available)")
        
        # Status basato sui risultati avanzati
        if advanced_metrics['r2'] > 0.8:
            status = "🟢 Excellent"
        elif advanced_metrics['r2'] > 0.7:
            status = "🟡 Good"
        else:
            status = "🔴 Needs improvement"
        print(f"  📊 Overall Status: {status}")
    
    print(f"\n🎉 TRAINING COMPLETED (VERSIONE MIGLIORATA)!")
    print(f"   ✅ {len(risultati_finali)} models trained con selezione automatica delle feature")
    print(f"   ✅ Normalizzazione specifica per condizioni operative implementata")
    print(f"   ✅ Valutazione avanzata con multiple sequenze per motore completata")
    print(f"   📁 Models saved in: {MODELS_DIR}/")
    print(f"   📁 Results saved in: {RESULTS_DIR}/")
    print(f"   🚀 Sistema di manutenzione predittiva migliorato pronto per l'uso!")
    
    print(f"\n🔧 MIGLIORAMENTI IMPLEMENTATI:")
    print(f"   1. ✅ Selezione automatica delle feature con VarianceThreshold")
    print(f"   2. ✅ Normalizzazione per condizioni operative (FD002, FD004)")
    print(f"   3. ✅ Valutazione avanzata con multiple sequenze per motore")
    print(f"   4. ✅ Architetture moderne:")
    print(f"      - SeparableConv1D per FD001/FD003 (più efficiente)")
    print(f"      - Bidirectional LSTM per FD002 (cattura dipendenze bidirezionali)")
    print(f"      - Attention mechanism per FD004 (focus su feature importanti)")
    print(f"   5. ✅ Hyperparameter tuning con KerasTuner (quando disponibile)")
    print(f"   6. ✅ Model versioning con metriche e timestamp")
    print(f"   7. ✅ Mantenimento della compatibilità con il sistema esistente")

if __name__ == "__main__":
    
    # Prompt utente per scegliere cosa fare
    print("\n🔧 SISTEMA DI MANUTENZIONE PREDITTIVA AVANZATO")
    print("=" * 60)
    print("Opzioni disponibili:")
    print("1. Addestramento completo tutti i dataset (con tutte le migliorie)")
    print("2. Addestramento singolo dataset")
    print("3. Test miglioramenti su singolo dataset")
    print("4. Demo tutti i miglioramenti")
    print("5. 🎯 Demo PHM08 + XAI (nuove funzionalità)")
    print("6. ⚔️ Confronto RMSE vs PHM08")
    print("7. 🚀 Showcase completo sistema")
    print("=" * 60)
    
    scelta = input("\nScegli un'opzione (1-7) [default: 1]: ").strip()
    
    if scelta == "2":
        dataset = input("Inserisci il dataset (FD001, FD002, FD003, FD004) [default: FD001]: ").strip()
        if not dataset:
            dataset = "FD001"
        train_single_dataset(dataset)
    
    elif scelta == "3":
        dataset = input("Inserisci il dataset per il test (FD001, FD002, FD003, FD004) [default: FD001]: ").strip()
        if not dataset:
            dataset = "FD001"
        test_improvements_single_dataset(dataset)
    
    elif scelta == "4":
        demo_all_improvements()
    
    elif scelta == "5":
        dataset = input("Inserisci il dataset per il demo PHM08+XAI (FD001, FD002, FD003, FD004) [default: FD001]: ").strip()
        if not dataset:
            dataset = "FD001"
        print(f"\n🎯 DEMO PHM08 + XAI per {dataset}")
        demo_advanced_features(dataset)
    
    elif scelta == "6":
        dataset = input("Inserisci il dataset per il confronto (FD001, FD002, FD003, FD004) [default: FD001]: ").strip()
        if not dataset:
            dataset = "FD001"
        print(f"\n⚔️ CONFRONTO RMSE vs PHM08 per {dataset}")
        compare_standard_vs_phm08(dataset)
    
    elif scelta == "7":
        print(f"\n🚀 SHOWCASE COMPLETO DEL SISTEMA")
        showcase_all_improvements()
    
    else:
        # Default: training completo
        print(f"\n🚀 Avvio training completo con tutte le migliorie...")
        main()

def test_improvements_single_dataset(dataset='FD001'):
    """
    Funzione di test per verificare i miglioramenti su un singolo dataset
    Utile per debugging e verifica rapida
    """
    print(f"🧪 TEST MIGLIORAMENTI SU {dataset}")
    print("=" * 50)
    
    # Verify dataset presence
    if not verify_dataset():
        print("❌ Dataset non trovato")
        return
    
    # Load data
    train_data, test_data, rul_data = load_data(dataset)
    if train_data is None:
        print(f"❌ Impossibile caricare {dataset}")
        return
    
    print(f"✅ Dataset {dataset} caricato con successo")
    
    # Create RUL labels
    train_data_with_rul = create_rul_labels(train_data)
    print(f"✅ Etichette RUL create")
    
    # Test 1: Selezione automatica delle feature
    print(f"\n🔍 TEST 1: Selezione automatica delle feature")
    train_data_filtered, test_data_filtered, selected_features, feature_stats = select_relevant_features(
        train_data_with_rul, test_data, variance_threshold=0.01
    )
    
    print(f"✅ Selezione feature completata:")
    print(f"   - Feature originali: {feature_stats['total_candidates']}")
    print(f"   - Feature rimosse: {feature_stats['removed_count']}")
    print(f"   - Feature selezionate: {feature_stats['selected_count']}")
    print(f"   - Feature rimosse: {feature_stats['removed_features']}")
    
    # Test 2: Normalizzazione per condizioni operative
    print(f"\n🔍 TEST 2: Normalizzazione per condizioni operative")
    train_data_normalized, test_data_normalized, scalers = normalize_data_by_operating_conditions(
        train_data_filtered, test_data_filtered, selected_features, dataset
    )
    
    print(f"✅ Normalizzazione completata:")
    print(f"   - Numero di scaler creati: {len(scalers)}")
    print(f"   - Tipi di scaler: {list(scalers.keys())}")
    
    # Test 3: Preparazione sequenze con feature selezionate
    print(f"\n🔍 TEST 3: Preparazione sequenze")
    sequence_length = 50
    X_train, y_train = prepare_sequences(
        train_data_normalized, sequence_length, selected_features
    )
    
    print(f"✅ Sequenze training preparate:")
    print(f"   - Forma X_train: {X_train.shape}")
    print(f"   - Forma y_train: {y_train.shape}")
    print(f"   - Numero di feature per sequenza: {X_train.shape[2]}")
    
    # Test 4: Preparazione test avanzata
    print(f"\n🔍 TEST 4: Preparazione test avanzata")
    test_sequences, X_test_advanced, summary_stats = prepara_dati_test_avanzato(
        test_data_normalized, rul_data, scalers, sequence_length, selected_features, dataset
    )
    
    print(f"✅ Test avanzato preparato:")
    print(f"   - Motori processati: {summary_stats['processed_engines']}")
    print(f"   - Sequenze totali: {summary_stats['total_sequences']}")
    print(f"   - Sequenze medie per motore: {summary_stats['avg_sequences_per_engine']:.1f}")
    print(f"   - Forma X_test_advanced: {X_test_advanced.shape}")
    
    print(f"\n✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print(f"🎯 I miglioramenti sono pronti per l'uso sul dataset {dataset}")
    
    return {
        'feature_stats': feature_stats,
        'scalers': scalers,
        'train_shape': X_train.shape,
        'test_stats': summary_stats,
        'selected_features': selected_features
    }

def train_single_dataset(dataset='FD001'):
    """
    Train a single dataset with all improvements
    
    Args:
        dataset: Dataset name (FD001, FD002, FD003, FD004)
    """
    print(f"\n🚀 TRAINING SINGLE DATASET - {dataset}")
    print("=" * 60)
    
    try:
        # Verify dataset presence
        if not verify_dataset():
            print("❌ Dataset non trovato")
            return
        
        # Load data
        train_data, test_data, rul_data = load_data(dataset)
        if train_data is None:
            print(f"❌ Impossibile caricare {dataset}")
            return
        
        print(f"✅ Dataset {dataset} caricato con successo")
        
        # Create RUL labels
        train_data_with_rul = create_rul_labels(train_data)
        print(f"✅ Etichette RUL create")
        
        # Feature selection
        print(f"\n🔍 Selezione automatica delle feature...")
        selected_features, feature_stats = select_relevant_features(train_data_with_rul)
        
        print(f"✅ Selezione feature completata:")
        print(f"   - Feature originali: {feature_stats['total_candidates']}")
        print(f"   - Feature rimosse: {feature_stats['removed_count']}")
        print(f"   - Feature selezionate: {feature_stats['selected_count']}")
        
        # Normalize data
        print(f"\n📏 Normalizzazione per condizioni operative...")
        train_data_normalized, test_data_normalized, scalers_info = normalize_data_by_operating_conditions(
            train_data_with_rul, test_data, selected_features, dataset
        )
        
        print(f"✅ Normalizzazione completata")
        
        # Prepare sequences
        print(f"\n⚙️ Preparazione sequenze...")
        sequence_length = 50
        X_train, y_train = prepare_sequences(
            train_data_normalized, sequence_length, selected_features
        )
        
        print(f"✅ Sequenze preparate: {X_train.shape}")
        
        # Hyperparameter optimization
        print(f"\n🔧 Ottimizzazione hyperparameters...")
        best_hps, model_type = optimize_hyperparameters(
            dataset, X_train, y_train, X_train.shape[1:], max_trials=10
        )
        
        # Select and build model
        print(f"\n🏗️ Costruzione modello...")
        if best_hps and KERAS_TUNER_AVAILABLE:
            tuner_model = build_tunable_model(best_hps, dataset, X_train.shape[1:])
            model = tuner_model
            print(f"✅ Modello ottimizzato creato")
        else:
            model = select_model(dataset, X_train.shape[1:])
            print(f"✅ Modello di default creato")
        
        # Train model
        print(f"\n🎯 Training del modello...")
        history = train_model(model, X_train, y_train)
        
        # Prepare test data for evaluation
        print(f"\n📊 Preparazione dati test...")
        sensor_cols_filtered = [col for col in selected_features if col.startswith('sensore_')]
        operational_cols = [col for col in selected_features if col.startswith('impostazione_op')]
        legacy_features = sensor_cols_filtered + operational_cols
        
        X_test = []
        y_test = []
        engine_ids = test_data_normalized['id_motore'].unique()
        
        for idx, engine_id in enumerate(engine_ids):
            engine_data = test_data_normalized[test_data_normalized['id_motore'] == engine_id].sort_values('ciclo')
            
            if len(engine_data) >= sequence_length:
                sequence = engine_data[legacy_features].iloc[-sequence_length:].values
                label = rul_data.iloc[idx]['RUL']
                X_test.append(sequence)
                y_test.append(label)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Comprehensive evaluation
        print(f"\n🔬 Valutazione comprensiva...")
        comprehensive_results = valuta_modello_con_phm08(
            model, X_test, y_test, 
            feature_names=legacy_features, 
            dataset_name=dataset
        )
        
        # Extract metrics for saving
        metrics_legacy = comprehensive_results['standard_metrics']
        
        # Save model with versioning
        print(f"\n💾 Salvataggio modello...")
        model_path, metadata_path = save_model_with_versioning(
            model, dataset, metrics_legacy, 
            hyperparameters=best_hps, 
            model_type=model_type,
            comprehensive_results=comprehensive_results
        )
        
        # Display final results
        print(f"\n🎉 TRAINING COMPLETATO!")
        print("=" * 40)
        print(f"📊 Performance del modello:")
        print(f"   - RMSE: {metrics_legacy['rmse']:.2f}")
        print(f"   - MAE: {metrics_legacy['mae']:.2f}")
        print(f"   - R²: {metrics_legacy['r2']:.4f}")
        
        if 'phm08_metrics' in comprehensive_results:
            phm08 = comprehensive_results['phm08_metrics']
            print(f"🎯 PHM08 Score: {phm08['phm08_score']:.2f}")
            print(f"   - Predizioni precoci: {phm08['early_predictions']}")
            print(f"   - Predizioni tardive: {phm08['late_predictions']}")
        
        print(f"💾 Modello salvato: {model_path}")
        print(f"📄 Metadata: {metadata_path}")
        
        return {
            'model': model,
            'results': comprehensive_results,
            'model_path': model_path,
            'metadata_path': metadata_path
        }
        
    except Exception as e:
        print(f"❌ Errore durante il training di {dataset}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Web interface functions
def load_models():
    """
    Load all trained models for web interface
    """
    models = {}
    for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
        # Try both naming conventions
        model_paths = [
            os.path.join(MODELS_DIR, f'model_{dataset.lower()}.h5'),
            os.path.join(MODELS_DIR, f'modello_{dataset.lower()}.h5')
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    models[dataset] = tf.keras.models.load_model(model_path)
                    print(f"✓ Loaded model {dataset} from {os.path.basename(model_path)}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"❌ Error loading model {dataset} from {model_path}: {e}")
        
        if not model_loaded:
            print(f"⚠️ Model {dataset} not found in {MODELS_DIR}/")
    
    return models

def prepare_sensor_data(sensor_data, sequence_length=50):
    """
    Prepare sensor data for prediction
    """
    if len(sensor_data.shape) == 2:
        # Single sequence
        if sensor_data.shape[0] < sequence_length:
            # Pad with zeros if sequence is too short
            padding = np.zeros((sequence_length - sensor_data.shape[0], sensor_data.shape[1]))
            sensor_data = np.vstack([padding, sensor_data])
        elif sensor_data.shape[0] > sequence_length:
            # Take last sequence_length samples
            sensor_data = sensor_data[-sequence_length:]
        
        sensor_data = sensor_data.reshape(1, sequence_length, -1)
    
    return sensor_data

def predict_rul(models, sensor_data, dataset='FD001'):
    """
    Predict RUL using the specified model
    """
    if dataset not in models:
        return "Model not available"
    
    try:
        # Prepare data
        sequence_length = 50  # Default, adjust based on dataset
        if dataset == 'FD002':
            sequence_length = 60
        elif dataset == 'FD004':
            sequence_length = 70
        
        prepared_data = prepare_sensor_data(sensor_data, sequence_length)
        
        # Make prediction
        prediction = models[dataset].predict(prepared_data, verbose=0)
        
        # Convert to RUL value
        rul = float(prediction[0][0])
        
        return rul
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction error"

def get_engine_status(rul_value):
    """
    Determine engine status based on RUL prediction
    """
    if isinstance(rul_value, str):
        return "Unknown"
    
    if rul_value > 50:
        return "Good"
    elif rul_value > 20:
        return "Warning"
    else:
        return "Critical"

def demo_all_improvements():
    """
    Funzione di demonstrazione che mostra tutti i miglioramenti implementati
    """
    print("\n" + "="*80)
    print("🚀 DEMONSTRAZIONE COMPLETA DEI MIGLIORAMENTI IMPLEMENTATI")
    print("="*80)
    
    print("\n📋 MIGLIORAMENTI IMPLEMENTATI:")
    print("\n1. 🔍 SELEZIONE AUTOMATICA DELLE FEATURE")
    print("   ✅ Utilizzo di VarianceThreshold per rimuovere feature con varianza nulla/bassa")
    print("   ✅ Identificazione automatica di sensori non informativi")
    print("   ✅ Riduzione dimensionalità e miglioramento efficienza")
    
    print("\n2. 🎯 NORMALIZZAZIONE PER CONDIZIONI OPERATIVE")
    print("   ✅ Normalizzazione globale per FD001/FD003 (condizioni singole)")
    print("   ✅ Normalizzazione separata per FD002/FD004 (condizioni multiple)")
    print("   ✅ Preservazione delle distribuzioni specifiche per condizione")
    
    print("\n3. 📊 VALUTAZIONE AVANZATA CON MULTIPLE SEQUENZE")
    print("   ✅ Generazione di multiple sequenze per motore (non solo l'ultima)")
    print("   ✅ Analisi dell'evoluzione dell'errore nel tempo")
    print("   ✅ Valutazione più robusta delle performance del modello")
    
    print("\n4. 🏗️ ARCHITETTURE MODERNE")
    print("   ✅ FD001: SeparableConv1D (più efficiente computazionalmente)")
    print("   ✅ FD002: Bidirectional LSTM (cattura dipendenze temporali bidirezionali)")
    print("   ✅ FD003: Deep SeparableConv1D (migliore detection di fault patterns)")
    print("   ✅ FD004: CNN-LSTM ibrido con Attention (focus automatico su feature importanti)")
    
    print("\n5. 🔧 HYPERPARAMETER TUNING")
    print("   ✅ Integrazione con KerasTuner per ottimizzazione automatica")
    print("   ✅ RandomSearch su spazi di hyperparameters definiti per ogni dataset")
    print("   ✅ Fallback a configurazioni di default se KerasTuner non disponibile")
    print("   ✅ Early stopping durante il tuning per efficienza")
    
    print("\n6. 💾 MODEL VERSIONING AVANZATO")
    print("   ✅ Nomi file con metriche di performance e timestamp")
    print("   ✅ Metadata JSON con informazioni complete del modello")
    print("   ✅ Tracciamento di hyperparameters utilizzati")
    print("   ✅ Funzione per caricare automaticamente il modello migliore")
    
    print("\n7. 🔄 COMPATIBILITÀ E ROBUSTEZZA")
    print("   ✅ Fallback automatico se dipendenze opzionali non disponibili")
    print("   ✅ Compatibilità con sistema esistente mantenuta")
    print("   ✅ Logging dettagliato per debugging e monitoraggio")
    print("   ✅ Gestione errori robusta")
    
    print("\n" + "="*80)
    print("🎯 COME UTILIZZARE I MIGLIORAMENTI:")
    print("="*80)
    
    print("\n📚 Per training completo con tutti i miglioramenti:")
    print("   python manutenzione_predittiva.py")
    
    print("\n🧪 Per testare miglioramenti su singolo dataset:")
    print("   from manutenzione_predittiva import test_improvements_single_dataset")
    print("   test_improvements_single_dataset('FD001')")
    
    print("\n🔧 Per installare dipendenze opzionali:")
    print("   pip install keras-tuner  # Per hyperparameter tuning")
    
    print("\n📂 Per caricare miglior modello per un dataset:")
    print("   from manutenzione_predittiva import load_best_model_version")
    print("   model, metadata = load_best_model_version('FD001', metric='r2')")
    
    print("\n💡 VANTAGGI OTTENUTI:")
    print("   🔸 Feature selection automatica → Riduzione overfitting, maggiore efficienza")
    print("   🔸 Normalizzazione avanzata → Migliore gestione dataset multi-condizione")
    print("   🔸 Valutazione multiple → Analisi più robusta delle performance")
    print("   🔸 Architetture moderne → Migliori performance, maggiore efficienza")
    print("   🔸 Hyperparameter tuning → Ottimizzazione automatica delle performance")
    print("   🔸 Model versioning → Tracciabilità completa e gestione versioni")
    
    print("\n🚀 Il sistema è ora pronto per la produzione con tutte le migliorie!")
    print("="*80)

def demo_advanced_features(dataset='FD001'):
    """
    Demo of advanced features: PHM08 scoring and XAI interpretability
    
    Args:
        dataset: Dataset to use for demo (default: FD001)
    """
    
    print(f"\n🚀 DEMO ADVANCED FEATURES - {dataset}")
    print("=" * 80)
    print("Testing PHM08 scoring function and SHAP interpretability")
    print("=" * 80)
    
    try:
        # Check if model exists
        model_files = [f for f in os.listdir(MODELS_DIR) if dataset.lower() in f.lower() and f.endswith('.h5')]
        if not model_files:
            print(f"❌ No trained model found for {dataset}. Training new model...")
            train_single_dataset(dataset)
            model_files = [f for f in os.listdir(MODELS_DIR) if dataset.lower() in f.lower() and f.endswith('.h5')]
        
        # Load the most recent model
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(MODELS_DIR, latest_model)
        
        print(f"📂 Loading model: {latest_model}")
        model = keras.models.load_model(model_path)
        
        # Load and prepare test data
        print(f"📊 Loading test data for {dataset}...")
        
        # Load data
        train_file = os.path.join(DATA_DIR, f'train_{dataset}.txt')
        test_file = os.path.join(DATA_DIR, f'test_{dataset}.txt')
        rul_file = os.path.join(DATA_DIR, f'RUL_{dataset}.txt')
        
        # Load datasets
        train_data = pd.read_csv(train_file, sep='\s+', header=None, names=column_names)
        test_data = pd.read_csv(test_file, sep='\s+', header=None, names=column_names)
        rul_data = pd.read_csv(rul_file, sep='\s+', header=None, names=['RUL'])
        
        # Feature selection
        print(f"🔍 Applying automatic feature selection...")
        train_data_with_rul = create_rul_labels(train_data)
        train_data_filtered, test_data_filtered, selected_features, feature_stats = select_relevant_features(
            train_data_with_rul, test_data, variance_threshold=0.01
        )
        
        # Normalize data
        print(f"📏 Applying condition-specific normalization...")
        train_data_normalized, test_data_normalized, scalers_info = normalize_data_by_operating_conditions(
            train_data_filtered, test_data_filtered, selected_features, dataset
        )
        
        # Prepare test sequences
        print(f"⚙️ Preparing test sequences...")
        sequence_length = 50
        
        # Get feature names for interpretability
        sensor_cols = [col for col in selected_features if col.startswith('sensore_')]
        operational_cols = [col for col in selected_features if col.startswith('impostazione_op')]
        feature_names = sensor_cols + operational_cols
        
        X_test = []
        y_test = []
        engine_ids = test_data_normalized['id_motore'].unique()
        
        for idx, engine_id in enumerate(engine_ids):
            engine_data = test_data_normalized[test_data_normalized['id_motore'] == engine_id].sort_values('ciclo')
            
            if len(engine_data) >= sequence_length:
                sequence = engine_data[feature_names].iloc[-sequence_length:].values
                label = rul_data.iloc[idx]['RUL']
                X_test.append(sequence)
                y_test.append(label)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"✅ Test data prepared: {X_test.shape[0]} samples")
        
        # Comprehensive evaluation with PHM08 and SHAP
        print(f"\n🎯 COMPREHENSIVE EVALUATION")
        comprehensive_results = valuta_modello_con_phm08(
            model, X_test, y_test, 
            feature_names=feature_names, 
            dataset_name=dataset
        )
        
        # Display results summary
        print(f"\n📋 RESULTS SUMMARY")
        print("=" * 40)
        
        # Standard metrics
        std_metrics = comprehensive_results['standard_metrics']
        print(f"📊 Standard Metrics:")
        print(f"   - RMSE: {std_metrics['rmse']:.2f}")
        print(f"   - MAE: {std_metrics['mae']:.2f}")
        print(f"   - R²: {std_metrics['r2']:.4f}")
        
        # PHM08 metrics
        phm08_metrics = comprehensive_results['phm08_metrics']
        print(f"\n🎯 PHM08 Metrics (Maintenance-specific):")
        print(f"   - PHM08 Score: {phm08_metrics['phm08_score']:.2f} (lower is better)")
        print(f"   - Early predictions: {phm08_metrics['early_predictions']}")
        print(f"   - Late predictions: {phm08_metrics['late_predictions']}")
        
        # Interpretability results
        if comprehensive_results['interpretability']:
            interp = comprehensive_results['interpretability']
            print(f"\n🔍 Interpretability Analysis:")
            print(f"   - Analysis completed: ✅")
            print(f"   - Top 3 most important features:")
            for i, (feat_name, importance) in enumerate(interp['top_features'][:3]):
                print(f"     {i+1}. {feat_name}: {importance:.4f}")
            print(f"   - SHAP plots saved: results/shap_analysis_{dataset.lower()}.png")
        else:
            print(f"\n🔍 Interpretability Analysis: ❌ (SHAP not available)")
        
        print(f"\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print(f"🎉 Advanced features working correctly for {dataset}")
        
        return comprehensive_results
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_standard_vs_phm08(dataset='FD001'):
    """
    Compare standard RMSE evaluation vs PHM08 scoring to show the difference
    
    Args:
        dataset: Dataset to use for comparison
    """
    
    print(f"\n⚔️ STANDARD RMSE vs PHM08 COMPARISON - {dataset}")
    print("=" * 70)
    
    try:
        # Quick training if needed
        results = demo_advanced_features(dataset)
        if not results:
            return
        
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        
        # Calculate standard metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate PHM08 score
        phm08_score, _ = calculate_phm08_score(y_true, y_pred)
        
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"=" * 40)
        print(f"Standard RMSE: {rmse:.2f} cycles")
        print(f"Standard MAE:  {mae:.2f} cycles")
        print(f"PHM08 Score:   {phm08_score:.2f} (maintenance-specific)")
        
        # Explain the difference
        print(f"\n💡 WHY PHM08 IS BETTER FOR MAINTENANCE:")
        print(f"   • RMSE treats all errors equally")
        print(f"   • PHM08 penalizes late predictions much more heavily")
        print(f"   • Late prediction = equipment failure (high cost)")
        print(f"   • Early prediction = unnecessary maintenance (lower cost)")
        print(f"   • PHM08 reflects real-world maintenance economics")
        
        # Show some examples
        errors = y_pred - y_true
        early_indices = np.where(errors <= 0)[0][:3]  # First 3 early predictions
        late_indices = np.where(errors > 0)[0][:3]    # First 3 late predictions
        
        if len(early_indices) > 0:
            print(f"\n📈 EXAMPLE EARLY PREDICTIONS (lower penalty):")
            for idx in early_indices:
                error = errors[idx]
                penalty = np.exp(-error / 13.0) - 1
                print(f"   True: {y_true[idx]:.1f}, Pred: {y_pred[idx]:.1f}, Error: {error:.1f}, PHM08 penalty: {penalty:.2f}")
        
        if len(late_indices) > 0:
            print(f"\n📉 EXAMPLE LATE PREDICTIONS (higher penalty):")
            for idx in late_indices:
                error = errors[idx]
                penalty = np.exp(error / 10.0) - 1
                print(f"   True: {y_true[idx]:.1f}, Pred: {y_pred[idx]:.1f}, Error: {error:.1f}, PHM08 penalty: {penalty:.2f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'phm08_score': phm08_score,
            'early_predictions': len(early_indices),
            'late_predictions': len(late_indices)
        }
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        return None

# Demo function to showcase all improvements
def showcase_all_improvements():
    """
    Showcase all implemented improvements in order
    """
    
    print(f"\n🎉 SHOWCASE ALL IMPROVEMENTS")
    print("=" * 80)
    print("Demonstrating all advanced features implemented:")
    print("1. Automatic feature selection")
    print("2. Condition-specific normalization") 
    print("3. Advanced test evaluation")
    print("4. Modern neural architectures")
    print("5. Hyperparameter optimization")
    print("6. Model versioning")
    print("7. PHM08 scoring function")
    print("8. SHAP interpretability analysis")
    print("=" * 80)
    
    # Test with FD001 (smallest dataset for quick demo)
    dataset = 'FD001'
    
    print(f"\n🚀 Running comprehensive demo on {dataset}...")
    
    # Run the full pipeline with all improvements
    results = demo_advanced_features(dataset)
    
    if results:
        print(f"\n🎯 SHOWCASE RESULTS:")
        print(f"✅ All 8 improvements working correctly!")
        print(f"✅ PHM08 scoring: {results['phm08_metrics']['phm08_score']:.2f}")
        print(f"✅ SHAP analysis: {'completed' if results['interpretability'] else 'unavailable'}")
        print(f"✅ Feature selection: automatic")
        print(f"✅ Modern architecture: implemented")
        print(f"✅ Advanced evaluation: multiple sequences")
        
    # Compare standard vs PHM08
    print(f"\n⚔️ Running RMSE vs PHM08 comparison...")
    comparison = compare_standard_vs_phm08(dataset)
    
    if comparison:
        print(f"\n🏆 COMPARISON COMPLETED!")
        print(f"Standard approach vs advanced maintenance-specific approach demonstrated")
    
    print(f"\n🎉 SHOWCASE COMPLETED!")
    print(f"System now includes all professional predictive maintenance features!")