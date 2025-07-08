# Predictive Maintenance Project - Fincantieri Inspired

## Description

This project implements a **predictive maintenance** system using **Machine Learning** and **Deep Learning** techniques to predict the **Remaining Useful Life (RUL)** of industrial engines. The project is inspired by the needs of companies like Fincantieri in the shipbuilding industry.

## Objective

Predict how many operating cycles remain before an engine fails, allowing to:
- **Optimize scheduled maintenance**
- **Reduce unplanned downtime**
- **Minimize maintenance costs**
- **Increase system reliability**

## Dataset

We use the **NASA Turbofan Engine Degradation Simulation Data Set**, a public dataset that perfectly simulates industrial sensor data:

- **Training data**: 100 engines with complete readings until failure
- **Test data**: 100 engines to evaluate performance
- **Sensors**: 21 different sensors (temperature, pressure, vibrations, etc.)
- **Operating cycles**: From 128 to 362 cycles per engine

## System Architecture

### 1. **Data Preprocessing**
- Min-Max normalization of sensors
- Creation of temporal sequences for LSTM
- Automatic RUL calculation

### 2. **LSTM Model**
- **Input**: Sequences of 50 operating cycles
- **Architecture**: 2 LSTM layers + Dropout + Dense
- **Output**: RUL prediction (remaining cycles)

### 3. **Evaluation**
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination

## Installation and Usage

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Execution
```bash
python manutenzione_predittiva.py
```

## Expected Results

The model should achieve:
- **RMSE**: ~20-30 cycles
- **MAE**: ~15-25 cycles
- **R²**: >0.85

## Project Structure

```
project-machine/
├── manutenzione_predittiva.py    # Main script
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── train_FD001.txt              # Training data (automatically downloaded)
├── test_FD001.txt               # Test data (automatically downloaded)
├── RUL_FD001.txt                # Real RUL for testing (automatically downloaded)
├── predictive_maintenance_model.h5  # Trained model (generated)
├── exploratory_analysis.png      # EDA plots (generated)
├── rul_analysis.png              # RUL analysis (generated)
├── training_progress.png         # Training curves (generated)
└── model_results.png             # Final results (generated)
```

## Technical Features

### Technologies Used
- **TensorFlow/Keras**: Deep Learning
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Preprocessing
- **Matplotlib/Seaborn**: Visualization

### Implemented Algorithms
- **LSTM (Long Short-Term Memory)**: Recurrent neural network
- **MinMaxScaler**: Data normalization
- **Early Stopping**: Overfitting prevention
- **Learning Rate Scheduling**: Convergence optimization

## System Output

The system automatically generates:

1. **Exploratory Analysis**: Sensor plots and correlations
2. **Training Curves**: Loss and MAE during training
3. **Final Results**: Predictions vs real values comparison
4. **Saved Model**: .h5 file for deployment

## Industrial Applications

This system can be adapted for:
- **Marine engines** (
- **Wind turbines**
- **Aircraft engines**
- **Pumping systems**
- **Industrial compressors**

## Future Extensions

- **Real-time monitoring** with data streaming
- **Multi-class classification** for failure types
- **Ensemble methods** to improve accuracy
- **Explainable AI** for interpretability
- **Edge computing** for IoT device deployment

## Support

For questions or issues:
- Verify all dependencies are installed
- Check internet connection for dataset download
- Ensure at least 4GB of RAM is available

---

**Note**: This project is for educational and demonstration purposes. For real industrial applications, the model must be validated on domain-specific data. 