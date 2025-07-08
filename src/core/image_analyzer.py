#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Analysis Module for Engine Predictive Maintenance

This module handles all image analysis functionality including:
- Visual anomaly detection
- Engine defect identification  
- Image quality assessment
- Sensor data generation from images

Separated from main app.py for better code organization and maintainability.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
from typing import Dict, Tuple, Optional, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalysisError(Exception):
    """Custom exception for image analysis errors"""
    pass

class ImageAnalyzer:
    """
    Main class for engine image analysis
    
    Handles all image processing, anomaly detection, and quality assessment
    for engine predictive maintenance applications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image analyzer with configuration
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        required_keys = [
            'BLOB_MIN_AREA', 'BLOB_MAX_AREA', 'RUST_COLOR_LOWER', 
            'RUST_COLOR_UPPER', 'RUST_THRESHOLD', 'DARK_THRESHOLD', 
            'OIL_LEAK_THRESHOLD', 'MAX_IMAGE_SIZE', 'MIN_IMAGE_SIZE'
        ]
        
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    def analyze_engine_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze engine image for visual anomalies and defects with robust error handling
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results or error information
            
        Raises:
            ImageAnalysisError: If analysis fails critically
        """
        
        # Validate input
        if not image_path or not isinstance(image_path, str):
            return self._create_error_response(
                "Invalid image path provided",
                "validation_error",
                "Image path must be a non-empty string"
            )
        
        if not os.path.exists(image_path):
            return self._create_error_response(
                "Image file not found",
                "file_not_found", 
                f"File does not exist: {image_path}"
            )
        
        try:
            # Load and validate image
            image = self._load_and_validate_image(image_path)
            if 'error' in image:
                return image
            
            # Perform comprehensive analysis
            analysis = self._perform_image_analysis(image)
            
            # Calculate confidence and assessment
            analysis = self._calculate_assessment(analysis)
            
            return analysis
            
        except cv2.error as e:
            return self._create_error_response(
                "OpenCV processing error",
                "opencv_error",
                str(e),
                extra_info={'opencv_version': cv2.__version__}
            )
        except MemoryError:
            return self._create_error_response(
                "Insufficient memory for image processing",
                "memory_error",
                "Image is too large or system memory is insufficient"
            )
        except Exception as e:
            logger.error(f"Unexpected error during image analysis: {e}")
            return self._create_error_response(
                "Unexpected error during image analysis",
                "unexpected_error",
                str(e),
                extra_info={'error_class': type(e).__name__}
            )
    
    def _load_and_validate_image(self, image_path: str) -> Dict[str, Any]:
        """Load and validate image file"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return self._create_error_response(
                "Failed to load image",
                "image_load_error",
                "cv2.imread returned None. Possible causes: corrupted file, unsupported format, or insufficient memory",
                extra_info={
                    'file_path': image_path,
                    'file_size': os.path.getsize(image_path) if os.path.exists(image_path) else "unknown"
                }
            )
        
        # Validate image dimensions
        if image.size == 0:
            return self._create_error_response(
                "Empty image file",
                "empty_image",
                "Image has zero dimensions"
            )
        
        # Check image dimensions against limits
        height, width = image.shape[:2]
        min_width, min_height = self.config['MIN_IMAGE_SIZE']
        max_width, max_height = self.config['MAX_IMAGE_SIZE']
        
        if height < min_height or width < min_width:
            return self._create_error_response(
                "Image too small for analysis",
                "image_too_small",
                f"Image dimensions ({width}x{height}) are below minimum required ({min_width}x{min_height})"
            )
        
        if height > max_height or width > max_width:
            return self._create_error_response(
                "Image too large for analysis",
                "image_too_large",
                f"Image dimensions ({width}x{height}) exceed maximum allowed ({max_width}x{max_height})"
            )
        
        return {'image': image, 'dimensions': (width, height)}
    
    def _perform_image_analysis(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive image analysis"""
        
        image = image_data['image']
        width, height = image_data['dimensions']
        
        # Initialize analysis results
        analysis = {
            "image_size": image.shape,
            "brightness": 0.0,
            "contrast": 0.0,
            "edge_density": 0.0,
            "anomalies_detected": [],
            "analysis_quality": "unknown"
        }
        
        # Calculate basic metrics
        try:
            analysis["brightness"] = float(np.mean(image))
            analysis["contrast"] = float(np.std(image))
        except Exception as e:
            raise ImageAnalysisError(f"Failed to calculate basic image metrics: {e}")
        
        # Convert to grayscale for advanced analysis
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            raise ImageAnalysisError(f"Failed to convert image to grayscale: {e}")
        
        # Edge detection
        analysis = self._perform_edge_analysis(gray, analysis)
        
        # Blob detection for potential defects
        analysis = self._perform_blob_detection(gray, analysis)
        
        # Color analysis for rust, oil leaks, etc.
        analysis = self._perform_color_analysis(image, analysis)
        
        return analysis
    
    def _perform_edge_analysis(self, gray: np.ndarray, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform edge detection analysis"""
        try:
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            analysis["edge_density"] = float(edge_density)
        except Exception as e:
            analysis["edge_density"] = 0.0
            analysis["anomalies_detected"].append("Edge detection failed - analysis limited")
            logger.warning(f"Edge detection failed: {e}")
        
        return analysis
    
    def _perform_blob_detection(self, gray: np.ndarray, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform blob detection for defect identification"""
        try:
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = 10
            params.maxThreshold = 200
            params.filterByArea = True
            params.minArea = self.config['BLOB_MIN_AREA']
            params.maxArea = self.config['BLOB_MAX_AREA']
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)
            
            if len(keypoints) > 0:
                analysis["anomalies_detected"].append(f"Detected {len(keypoints)} potential defects")
        except Exception as e:
            analysis["anomalies_detected"].append("Blob detection failed - analysis limited")
            logger.warning(f"Blob detection failed: {e}")
        
        return analysis
    
    def _perform_color_analysis(self, image: np.ndarray, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform color-based analysis for rust and oil leaks"""
        
        # Rust detection
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            lower_rust = np.array(self.config['RUST_COLOR_LOWER'])
            upper_rust = np.array(self.config['RUST_COLOR_UPPER'])
            rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
            rust_pixels = np.sum(rust_mask > 0)
            rust_percentage = rust_pixels / (image.shape[0] * image.shape[1]) * 100
            
            if rust_percentage > self.config['RUST_THRESHOLD']:
                analysis["anomalies_detected"].append(f"Rust detected: {rust_percentage:.2f}% of image")
        except Exception as e:
            analysis["anomalies_detected"].append("Rust detection failed - analysis limited")
            logger.warning(f"Rust detection failed: {e}")
        
        # Oil leak detection
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dark_mask = cv2.inRange(gray, 0, self.config['DARK_THRESHOLD'])
            dark_pixels = np.sum(dark_mask > 0)
            dark_percentage = dark_pixels / (image.shape[0] * image.shape[1]) * 100
            
            if dark_percentage > self.config['OIL_LEAK_THRESHOLD']:
                analysis["anomalies_detected"].append(f"Potential oil leaks: {dark_percentage:.2f}% dark areas")
        except Exception as e:
            analysis["anomalies_detected"].append("Oil leak detection failed - analysis limited")
            logger.warning(f"Oil leak detection failed: {e}")
        
        return analysis
    
    def _calculate_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence and visual assessment"""
        
        try:
            anomaly_count = len(analysis["anomalies_detected"])
            
            # Calculate confidence based on multiple factors
            confidence_factors = []
            
            # Factor 1: Image quality
            if analysis["brightness"] > 50 and analysis["brightness"] < 200:
                confidence_factors.append(0.9)  # Good brightness
            else:
                confidence_factors.append(0.6)  # Poor brightness
                
            # Factor 2: Contrast quality
            if analysis["contrast"] > 30 and analysis["contrast"] < 100:
                confidence_factors.append(0.9)  # Good contrast
            else:
                confidence_factors.append(0.7)  # Poor contrast
                
            # Factor 3: Edge density (complexity)
            if analysis["edge_density"] > 0.05 and analysis["edge_density"] < 0.20:
                confidence_factors.append(0.85)  # Normal complexity
            else:
                confidence_factors.append(0.75)  # Unusual complexity
            
            # Calculate final confidence
            base_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            analysis["confidence"] = float(base_confidence)
            
            # Determine visual assessment
            if anomaly_count == 0 and analysis["brightness"] > 80 and analysis["brightness"] < 180:
                analysis["visual_assessment"] = "Good"
                analysis["confidence"] = min(base_confidence + 0.1, 0.95)
            elif anomaly_count <= 1 and analysis["edge_density"] < 0.15:
                analysis["visual_assessment"] = "Good" 
                analysis["confidence"] = base_confidence
            elif anomaly_count <= 2:
                analysis["visual_assessment"] = "Moderate"
                analysis["confidence"] = base_confidence - 0.1
            else:
                analysis["visual_assessment"] = "Poor"
                analysis["confidence"] = base_confidence - 0.2
            
            # Determine analysis quality
            failed_analyses = sum(1 for anomaly in analysis["anomalies_detected"] if "failed" in anomaly.lower())
            if failed_analyses == 0:
                analysis["analysis_quality"] = "complete"
            elif failed_analyses <= 2:
                analysis["analysis_quality"] = "partial"
            else:
                analysis["analysis_quality"] = "limited"
                
        except Exception as e:
            analysis["visual_assessment"] = "Unknown"
            analysis["confidence"] = 0.0
            analysis["analysis_quality"] = "error"
            analysis["anomalies_detected"].append(f"Assessment calculation failed: {str(e)}")
            logger.error(f"Assessment calculation failed: {e}")
        
        return analysis
    
    def generate_analysis_plot(self, image_path: str, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate a visualization of the analysis results
        
        Args:
            image_path: Path to the original image
            analysis: Analysis results dictionary
            
        Returns:
            Base64 encoded plot image or None if generation fails
        """
        try:
            # Load original image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Engine Image Analysis Results', fontsize=16, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            axes[0, 1].imshow(gray, cmap='gray')
            axes[0, 1].set_title('Grayscale Analysis')
            axes[0, 1].axis('off')
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            axes[1, 0].imshow(edges, cmap='gray')
            axes[1, 0].set_title('Edge Detection')
            axes[1, 0].axis('off')
            
            # Analysis metrics
            axes[1, 1].axis('off')
            metrics_text = f"""
            Image Size: {analysis['image_size'][1]}x{analysis['image_size'][0]}
            Brightness: {analysis['brightness']:.1f}
            Contrast: {analysis['contrast']:.1f}
            Edge Density: {analysis['edge_density']:.4f}
            Visual Assessment: {analysis['visual_assessment']}
            Confidence: {analysis['confidence']:.2f}
            
            Anomalies Detected:
            """
            for anomaly in analysis['anomalies_detected']:
                metrics_text += f"• {anomaly}\n"
            
            axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot to bytes
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Convert to base64
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            return img_str
            
        except Exception as e:
            logger.error(f"Failed to generate analysis plot: {e}")
            return None
    
    def _create_error_response(self, error_msg: str, error_type: str, details: str, 
                             extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized error response"""
        response = {
            "error": error_msg,
            "error_type": error_type,
            "details": details
        }
        
        if extra_info:
            response.update(extra_info)
        
        return response


class SensorDataGenerator:
    """
    Generate simulated sensor data based on image analysis
    
    ⚠️  IMPORTANT: This is a DEMONSTRATION function for workflow testing.
    The correlations between image characteristics and sensor values are 
    simplified heuristics and do not represent real physical relationships.
    
    For production applications, this should be replaced with:
    1. Real sensor data from the actual engine
    2. A trained model correlating visual features to sensor patterns
    3. Physics-based models of engine degradation
    """
    
    def __init__(self):
        """Initialize sensor data generator"""
        self.simulation_warning = {
            'type': 'simulation',
            'warning': 'This is simulated data for demonstration purposes only',
            'limitations': [
                'Image-sensor correlations are heuristic approximations',
                'Not based on real physical engine models',
                'Should not be used for actual maintenance decisions',
                'For production: use real sensor data or trained correlation models'
            ],
            'recommendations': [
                'Collect real sensor data from the engine',
                'Train models on actual image-sensor correlations',
                'Validate with domain experts and physical models',
                'Implement proper sensor data acquisition systems'
            ]
        }
    
    def generate_sensor_data_from_image(self, image_analysis: Dict[str, Any], dataset: str = 'FD001') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate simulated sensor data based on image analysis
        
        Args:
            image_analysis: Results from image analysis
            dataset: Dataset identifier to determine correct dimensions
            
        Returns:
            Tuple of (sensor_data_array, generation_metadata)
        """
        
        try:
            # Base sensor values (typical turbofan engine ranges)
            base_values = {
                'temperature': 350.0,      # °C
                'pressure': 14.5,          # bar
                'vibration': 0.1,          # g
                'fuel_flow': 2.5,          # kg/s
                'rpm': 12000,              # RPM
                'oil_pressure': 3.2        # bar
            }
            
            # Extract image characteristics
            num_anomalies = len(image_analysis.get('anomalies_detected', []))
            brightness = image_analysis.get('brightness', 128)
            contrast = image_analysis.get('contrast', 50)
            edge_density = image_analysis.get('edge_density', 0.1)
            
            # Simplified degradation simulation
            # Note: These are DEMONSTRATION correlations only
            degradation_factors = {
                'temperature': 1 + (num_anomalies * 0.1) + ((128 - brightness) * 0.002),
                'pressure': 1 - (num_anomalies * 0.05),
                'vibration': 1 + (num_anomalies * 0.2) + (contrast * 0.001),
                'fuel_flow': 1 + (num_anomalies * 0.08),
                'rpm': 1 - (num_anomalies * 0.03),
                'oil_pressure': 1 - (num_anomalies * 0.07)
            }
            
            # Generate sensor data array with correct dimensions for each dataset
            # Based on hyperparameter tuning results:
            dataset_dimensions = {
                'FD001': (50, 11),
                'FD002': (60, 23), 
                'FD003': (50, 13),
                'FD004': (70, 23)
            }
            
            sequence_length, num_features = dataset_dimensions.get(dataset, (50, 23))
            sensor_data = np.zeros((1, sequence_length, num_features))
            
            # Fill with simulated patterns
            cycles = sensor_data.shape[1]  # Use dynamic cycle count
            for cycle in range(cycles):
                # Progressive degradation over cycles
                cycle_factor = 1 + (cycle * 0.005)  # Gradual degradation
                
                # Core sensors (first 6)
                sensor_data[0, cycle, 0] = base_values['temperature'] * degradation_factors['temperature'] * cycle_factor + np.random.normal(0, 3)
                sensor_data[0, cycle, 1] = base_values['pressure'] * degradation_factors['pressure'] / cycle_factor + np.random.normal(0, 0.2)
                sensor_data[0, cycle, 2] = base_values['vibration'] * degradation_factors['vibration'] * cycle_factor + np.random.normal(0, 0.005)
                sensor_data[0, cycle, 3] = base_values['fuel_flow'] * degradation_factors['fuel_flow'] * cycle_factor + np.random.normal(0, 0.1)
                sensor_data[0, cycle, 4] = base_values['rpm'] * degradation_factors['rpm'] / cycle_factor + np.random.normal(0, 50)
                sensor_data[0, cycle, 5] = base_values['oil_pressure'] * degradation_factors['oil_pressure'] / cycle_factor + np.random.normal(0, 0.1)
                
                # Additional sensors (7 to max) with correlated patterns
                max_sensors = sensor_data.shape[2]
                for sensor in range(6, max_sensors):
                    # Create correlated patterns based on core sensors
                    correlation_factor = 0.3 + (sensor * 0.02)
                    base_value = 100 + (sensor * 10)
                    
                    # Correlate with temperature and vibration
                    correlated_value = (
                        base_value * cycle_factor * 
                        (1 + correlation_factor * (sensor_data[0, cycle, 0] / base_values['temperature'] - 1)) +
                        correlation_factor * sensor_data[0, cycle, 2] * 100
                    )
                    
                    sensor_data[0, cycle, sensor] = correlated_value + np.random.normal(0, 2)
            
            # Ensure realistic bounds
            sensor_data = np.clip(sensor_data, 0, 1000)
            
            return sensor_data, {
                'simulation_info': self.simulation_warning,
                'generation_params': {
                    'base_values': base_values,
                    'degradation_factors': degradation_factors,
                    'image_factors': {
                        'anomalies': num_anomalies,
                        'brightness': brightness,
                        'contrast': contrast,
                        'edge_density': edge_density
                    }
                },
                'data_shape': sensor_data.shape,
                'data_range': {
                    'min': float(np.min(sensor_data)),
                    'max': float(np.max(sensor_data)),
                    'mean': float(np.mean(sensor_data))
                }
            }
            
        except Exception as e:
            # Return safe fallback data with correct dimensions
            logger.error(f"Error generating sensor data: {e}, using fallback values")
            dataset_dimensions = {
                'FD001': (50, 11),
                'FD002': (60, 23), 
                'FD003': (50, 13),
                'FD004': (70, 23)
            }
            sequence_length, num_features = dataset_dimensions.get(dataset, (50, 23))
            fallback_data = np.random.normal(100, 10, (1, sequence_length, num_features))
            fallback_data = np.clip(fallback_data, 0, 200)
            
            return fallback_data, {
                'simulation_info': self.simulation_warning,
                'error': str(e),
                'fallback_used': True,
                'data_shape': fallback_data.shape
            }


class ModelSelector:
    """
    Choose the best predictive model based on image analysis and operational context
    """
    
    def __init__(self):
        """Initialize model selector"""
        self.context_mapping = {
            'stable': 'FD001',
            'variable': 'FD002', 
            'fault_conditions': 'FD003',
            'complex_faults': 'FD004'
        }
    
    def choose_best_model(self, image_analysis: Dict[str, Any], 
                         user_context: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Choose the best predictive model based on image analysis and operational context
        
        Args:
            image_analysis: Results from image analysis
            user_context: User-specified operational context
            
        Returns:
            Tuple of (selected_dataset, selection_reasoning)
        """
        
        # If user provides context, use it directly (more reliable)
        if user_context:
            if user_context in self.context_mapping:
                return self.context_mapping[user_context], {
                    'reason': f'User-specified context: {user_context}',
                    'confidence': 0.95,
                    'method': 'user_context'
                }
            else:
                # Fallback to auto-selection if invalid context
                logger.warning(f"Invalid user context '{user_context}', using auto-selection")
        
        # Auto-selection based on image analysis (fallback)
        try:
            num_anomalies = len(image_analysis.get('anomalies_detected', []))
            brightness = image_analysis.get('brightness', 128)
            edge_density = image_analysis.get('edge_density', 0.1)
            
            # Simplified, more conservative logic
            if num_anomalies >= 3:
                selected_model = 'FD004'
                reason = f'High anomaly count ({num_anomalies}) suggests complex fault conditions'
                confidence = 0.7
            elif num_anomalies >= 2:
                selected_model = 'FD003'
                reason = f'Moderate anomaly count ({num_anomalies}) suggests fault conditions'
                confidence = 0.6
            elif edge_density > 0.15:
                selected_model = 'FD002'
                reason = f'High edge density ({edge_density:.3f}) suggests variable conditions'
                confidence = 0.5
            else:
                selected_model = 'FD001'
                reason = 'Standard conditions detected, using baseline model'
                confidence = 0.8
            
            return selected_model, {
                'reason': reason,
                'confidence': confidence,
                'method': 'image_analysis',
                'factors': {
                    'anomalies': num_anomalies,
                    'brightness': brightness,
                    'edge_density': edge_density
                }
            }
            
        except Exception as e:
            # Fallback to safest option
            logger.error(f"Error in model selection: {e}, using FD001 as fallback")
            return 'FD001', {
                'reason': 'Fallback due to analysis error',
                'confidence': 0.3,
                'method': 'fallback',
                'error': str(e)
            }


# Factory function for creating analyzer instances
def create_image_analyzer(config: Dict[str, Any]) -> ImageAnalyzer:
    """
    Factory function to create an ImageAnalyzer instance
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ImageAnalyzer instance
    """
    return ImageAnalyzer(config)


def create_sensor_generator() -> SensorDataGenerator:
    """
    Factory function to create a SensorDataGenerator instance
    
    Returns:
        SensorDataGenerator instance
    """
    return SensorDataGenerator()


def create_model_selector() -> ModelSelector:
    """
    Factory function to create a ModelSelector instance
    
    Returns:
        ModelSelector instance
    """
    return ModelSelector()