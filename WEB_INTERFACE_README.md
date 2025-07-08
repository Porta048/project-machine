# Engine Predictive Maintenance - Web Interface

## Overview

This web interface allows users to upload engine images and get comprehensive analysis combining:
- **Visual Analysis**: AI-powered image analysis for detecting defects, rust, oil leaks, and structural anomalies
- **Sensor Prediction**: Integration with trained predictive maintenance models to estimate Remaining Useful Life (RUL)

## Features

### Image Analysis
- **Drag & Drop Interface**: Easy file upload with visual feedback
- **Multiple Format Support**: JPG, PNG, GIF, BMP (up to 16MB)
- **Real-time Processing**: Instant analysis with progress indicators
- **Visual Defect Detection**: 
  - Rust detection using color analysis
  - Oil leak detection through dark area analysis
  - Structural anomalies via edge detection and blob analysis
  - Brightness and contrast analysis

### Predictive Models
- **Multi-Dataset Support**: FD001, FD002, FD003, FD004 models
- **RUL Prediction**: Remaining Useful Life estimation
- **Status Assessment**: Good/Warning/Critical status based on predictions
- **Confidence Scoring**: Reliability metrics for predictions

### Comprehensive Results
- **Visual Analysis Report**: Detailed metrics and anomaly detection
- **Sensor Prediction Report**: RUL values and engine status
- **Interactive Plots**: Generated analysis visualizations
- **Responsive Design**: Works on desktop and mobile devices

## Installation

### Prerequisites
1. Python 3.8+ installed
2. Trained models from the main predictive maintenance system

### Setup Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models** (if not already done):
   ```bash
   python manutenzione_predittiva.py
   ```

3. **Start Web Interface**:
   ```bash
   python start_web_interface.py
   ```

4. **Access Interface**:
   - Open browser to: `http://localhost:5000`
   - The interface will open automatically

## Usage

### Uploading Images
1. **Drag & Drop**: Simply drag engine images onto the upload area
2. **Click to Browse**: Click the upload area to select files manually
3. **Supported Formats**: JPG, PNG, GIF, BMP
4. **File Size**: Maximum 16MB per file

### Understanding Results

#### Visual Analysis
- **Image Size**: Dimensions of uploaded image
- **Brightness**: Average brightness level (0-255)
- **Contrast**: Standard deviation of pixel values
- **Edge Density**: Measure of structural complexity
- **Visual Assessment**: Good/Moderate/Poor based on detected anomalies
- **Confidence**: Reliability score of the visual analysis

#### Sensor Prediction
- **Dataset**: Which predictive model was used (FD001-FD004)
- **RUL Prediction**: Estimated Remaining Useful Life in cycles
- **Status**: 
  - Good: RUL > 50 cycles
  - Warning: RUL 20-50 cycles  
  - Critical: RUL < 20 cycles

#### Anomalies Detected
- **Rust Detection**: Percentage of image showing rust-like colors
- **Oil Leaks**: Percentage of very dark areas (potential leaks)
- **Structural Defects**: Number of detected anomalies via blob detection

## Technical Details

### Image Analysis Pipeline
1. **Preprocessing**: Convert to RGB, resize if needed
2. **Grayscale Analysis**: Edge detection and structural analysis
3. **Color Analysis**: HSV color space for rust detection
4. **Blob Detection**: Identify potential defects
5. **Statistical Analysis**: Brightness, contrast, edge density

### Model Integration
- **Multi-Model Loading**: Automatic loading of all available models
- **Data Preparation**: Sensor data formatting and normalization
- **Prediction Pipeline**: RUL estimation with error handling
- **Status Classification**: Automatic status determination

### Architecture
```
Frontend (HTML/CSS/JS)
    ↓
Flask Web Server (app.py)
    ↓
Image Analysis (OpenCV)
    ↓
Predictive Models (TensorFlow)
    ↓
Results Display
```

## File Structure
```
project-machine/
├── app.py                      # Main Flask application
├── start_web_interface.py      # Web interface launcher
├── templates/
│   └── index.html             # Web interface template
├── uploads/                   # Uploaded images storage
├── models/                    # Trained model files
├── manutenzione_predittiva.py # Core predictive maintenance system
└── requirements.txt           # Python dependencies
```

## Troubleshooting

### Common Issues

1. **"Model not available" Error**:
   - Ensure models are trained: `python manutenzione_predittiva.py`
   - Check models directory exists and contains `.h5` files

2. **Image Upload Fails**:
   - Verify file format is supported (JPG, PNG, GIF, BMP)
   - Check file size is under 16MB
   - Ensure uploads directory has write permissions

3. **Server Won't Start**:
   - Check if port 5000 is available
   - Install missing dependencies: `pip install -r requirements.txt`
   - Verify Python version is 3.8+

4. **Analysis Takes Too Long**:
   - Large images may take longer to process
   - Check system resources (CPU, memory)
   - Consider resizing very large images

### Performance Tips
- **Image Optimization**: Use images under 5MB for faster processing
- **Model Loading**: First run may be slower as models load into memory
- **Browser Compatibility**: Use modern browsers (Chrome, Firefox, Safari, Edge)

## Security Notes
- Uploaded images are stored locally in the `uploads/` directory
- No data is sent to external services
- File validation prevents malicious uploads
- Maximum file size limits prevent DoS attacks

## Future Enhancements
- **Batch Processing**: Upload multiple images at once
- **Historical Analysis**: Track engine condition over time
- **Advanced AI Models**: Integration with pre-trained vision models
- **Real-time Monitoring**: Live sensor data integration
- **Export Reports**: PDF/Excel report generation
- **User Authentication**: Multi-user support with login system

## Support
For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure models are properly trained
4. Check console output for error messages

---

**Note**: This interface is designed for educational and research purposes. For production use, additional security measures and error handling should be implemented.