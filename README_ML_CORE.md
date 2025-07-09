# ML Core Project - Predictive Maintenance

Progetto semplificato e ottimizzato per l'addestramento di modelli ML performanti per la manutenzione predittiva di motori turbofan.

## 🎯 Obiettivo

Sviluppare modelli LSTM ad alte prestazioni per la predizione del Remaining Useful Life (RUL) utilizzando i dataset NASA Turbofan Engine Degradation.

## 📁 Struttura del Progetto

```
project-machine/
├── ml_core_project.py      # Modulo principale con classi per training
├── config.py               # Configurazioni e iperparametri
├── train_models.py         # Script di avvio rapido
├── requirements_ml_core.txt # Dipendenze essenziali
├── data/                   # Dataset NASA Turbofan
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── ... (FD002, FD003, FD004)
├── models/                 # Modelli addestrati
├── logs/                   # Log di training
└── results/                # Risultati e metriche
```

## 🚀 Avvio Rapido

### 1. Installazione Dipendenze

```bash
pip install -r requirements_ml_core.txt
```

### 2. Training Singolo Dataset

```bash
# Training base su FD001
python train_models.py --dataset FD001

# Training avanzato su FD004
python train_models.py --dataset FD004 --model advanced_lstm --epochs 150

# Training con ottimizzazione iperparametri
python train_models.py --dataset FD002 --optimize
```

### 3. Training su Tutti i Dataset

```bash
# Training base su tutti i dataset
python train_models.py --all-datasets

# Training avanzato su tutti i dataset
python train_models.py --all-datasets --model advanced_lstm --epochs 200
```

### 4. Benchmark Rapido

```bash
# Test rapido su tutti i dataset (20 epochs)
python train_models.py --benchmark
```

### 5. Interfaccia Web Interattiva 🌐

**Avvio rapido con controllo automatico:**
```bash
python start_interface.py
```

**Avvio manuale:**
```bash
python app.py
```

**Accesso interfaccia:**
- Locale: http://localhost:5000
- Rete: http://[IP_COMPUTER]:5000

**Funzionalità interfaccia:**
- 📤 Upload drag-and-drop di immagini motore
- 🎯 Predizioni RUL in tempo reale
- 🎨 Visualizzazione 3D interattiva
- 📊 Metriche dettagliate dei modelli
- 🔄 Selezione dinamica dataset/modello

## 📊 Dataset

| Dataset | Descrizione | Unità Train | Unità Test | Modalità Guasto | Condizioni Operative |
|---------|-------------|-------------|------------|-----------------|----------------------|
| FD001   | Singola modalità, singola condizione | 100 | 100 | 1 | 1 |
| FD002   | Singola modalità, multiple condizioni | 260 | 259 | 1 | 6 |
| FD003   | Multiple modalità, singola condizione | 100 | 100 | 2 | 1 |
| FD004   | Multiple modalità, multiple condizioni | 248 | 249 | 2 | 6 |

## 🏗️ Architetture Modelli

### Basic LSTM
- 2 layer LSTM (100, 50 unità)
- Dropout 0.2
- Ideale per FD001

### Advanced LSTM
- 3 layer LSTM (128, 64, 32 unità)
- Dropout 0.3
- Batch Normalization
- Ideale per FD002, FD003

### Deep LSTM
- 4 layer LSTM (256, 128, 64, 32 unità)
- Dropout 0.4
- Ideale per FD004 (dataset più complesso)

### Lightweight LSTM
- 2 layer LSTM (64, 32 unità)
- Dropout 0.15
- Per test rapidi

## ⚙️ Configurazioni Ottimizzate

Il sistema include configurazioni pre-ottimizzate per ogni dataset:

```python
# FD001: Configurazione semplice
{
    'model': 'basic_lstm',
    'sequence_length': 30,
    'batch_size': 32,
    'learning_rate': 0.001
}

# FD004: Configurazione avanzata
{
    'model': 'deep_lstm',
    'sequence_length': 40,
    'batch_size': 64,
    'learning_rate': 0.0005
}
```

## 📈 Metriche di Valutazione

- **RMSE**: Root Mean Square Error (metrica principale)
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **PHM08 Score**: Scoring function del PHM08 Challenge

## 🔧 Utilizzo Programmatico

```python
from ml_core_project import DataLoader, ModelArchitecture, ModelTrainer
from config import get_optimized_config

# Carica dati
data_loader = DataLoader()
train_df, test_df, rul_df = data_loader.load_dataset('FD001')
X_train, y_train, X_test, y_test = data_loader.preprocess_data(train_df, test_df)

# Costruisci modello
arch = ModelArchitecture(X_train.shape[1:])
model = arch.build_lstm_model()

# Addestra
trainer = ModelTrainer()
result = trainer.train_model(model, X_train, y_train, X_test, y_test, 'FD001')

print(f"RMSE: {result['metrics']['rmse']:.3f}")
```

## 🎛️ Ottimizzazione Iperparametri

```python
from ml_core_project import ModelOptimizer

optimizer = ModelOptimizer()

# Griglia di ricerca
param_grid = {
    'lstm_units': [[64, 32], [100, 50], [128, 64]],
    'dropout_rate': [0.2, 0.3],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64]
}

result = optimizer.optimize_hyperparameters(data_loader, 'FD001', param_grid)
print(f"Best config: {result['best_config']}")
```

## 📋 Risultati Attesi

### Performance Target (RMSE)
- **FD001**: < 15.0
- **FD002**: < 20.0
- **FD003**: < 15.0
- **FD004**: < 25.0

### Tempi di Training (GPU)
- **FD001**: ~5-10 minuti
- **FD002**: ~15-25 minuti
- **FD003**: ~5-10 minuti
- **FD004**: ~20-30 minuti

## 🔍 Monitoraggio Training

```bash
# Visualizza log in tempo reale
tail -f training.log

# Controlla modelli salvati
ls -la models/

# Visualizza risultati
cat results_*.json
```

## 🚀 Ottimizzazioni Performance

### GPU
- Memory growth automatico
- Mixed precision (se supportata)
- Batch size ottimizzato per GPU

### CPU
- Parallelizzazione automatica
- Ottimizzazione memoria
- Early stopping intelligente

### Callbacks Avanzati
- **EarlyStopping**: Patience 15 epochs
- **ReduceLROnPlateau**: Riduzione LR automatica
- **ModelCheckpoint**: Salvataggio best model

## 📊 Analisi Risultati

```python
import json
import matplotlib.pyplot as plt

# Carica risultati
with open('results_20240101_120000.json', 'r') as f:
    results = json.load(f)

# Plot metriche
for dataset, result in results.items():
    if 'metrics' in result:
        print(f"{dataset}: RMSE={result['metrics']['rmse']:.3f}")
```

## 🔧 Troubleshooting

### Errori Comuni

1. **Out of Memory**
   ```bash
   # Riduci batch size
   python train_models.py --dataset FD001 --model lightweight_lstm
   ```

2. **Dataset Non Trovato**
   ```bash
   # Verifica struttura directory data/
   ls -la data/
   ```

3. **GPU Non Rilevata**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

### Performance Lente

1. **Usa modello lightweight per test**
2. **Riduci sequence_length**
3. **Aumenta batch_size (se memoria sufficiente)**
4. **Abilita mixed precision**

## 📝 Note Tecniche

- **Preprocessing**: StandardScaler per normalizzazione
- **Sequence Length**: 30 timesteps (ottimale per la maggior parte dei dataset)
- **RUL Capping**: 125 cicli (standard in letteratura)
- **Validation Split**: 80/20
- **Seed**: 42 (per riproducibilità)

## 🎯 Prossimi Passi

1. **Ensemble Methods**: Combinazione di più modelli
2. **Attention Mechanisms**: LSTM con attention
3. **Transfer Learning**: Pre-training su dataset multipli
4. **Real-time Inference**: Ottimizzazione per produzione

---

**Avvia subito il training:**
```bash
python train_models.py --all-datasets --model advanced_lstm
```