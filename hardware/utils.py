"""
Shared Utilities for Hardware AI Models
Provides common functionality for model loading, logging, and error handling
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Any, Callable
from functools import wraps

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ModelLoader:
    """Centralized TFLite model loading with fallback logic"""
    
    def __init__(self, model_search_paths: List[str] = None):
        """
        Initialize model loader with search paths
        
        Args:
            model_search_paths: List of directories to search for models
        """
        self.logger = logging.getLogger(__name__)
        
        # Default search paths for Raspberry Pi deployment
        if model_search_paths is None:
            model_search_paths = [
                "hardware/models/",
                "models/",
                "/home/pi/models/",
                str(Path.cwd() / "models"),
                str(Path.cwd() / "hardware" / "models")
            ]
        
        self.search_paths = [Path(p) for p in model_search_paths]
    
    def find_model(self, model_filename: str) -> Optional[Path]:
        """
        Search for model file in configured paths
        
        Args:
            model_filename: Name of the model file to find
            
        Returns:
            Path to model file if found, None otherwise
        """
        for search_path in self.search_paths:
            model_path = search_path / model_filename
            if model_path.exists():
                self.logger.info(f"Found model: {model_path}")
                return model_path
        
        self.logger.warning(f"Model not found: {model_filename}")
        return None
    
    def load_tflite_model(self, model_filename: str) -> Optional[Any]:
        """
        Load TFLite model with automatic path resolution
        
        Args:
            model_filename: Name of the TFLite model file
            
        Returns:
            TFLite Interpreter if successful, None otherwise
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow not available")
            return None
        
        model_path = self.find_model(model_filename)
        if model_path is None:
            return None
        
        try:
            interpreter = tf.lite.Interpreter(str(model_path))
            interpreter.allocate_tensors()
            self.logger.info(f"Loaded TFLite model: {model_filename}")
            return interpreter
        except Exception as e:
            self.logger.error(f"Failed to load TFLite model {model_filename}: {e}")
            return None
    
    def load_keras_model(self, model_filename: str) -> Optional[Any]:
        """
        Load Keras .h5 model with automatic path resolution
        
        Args:
            model_filename: Name of the Keras model file
            
        Returns:
            Keras model if successful, None otherwise
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow not available")
            return None
        
        model_path = self.find_model(model_filename)
        if model_path is None:
            return None
        
        try:
            from tensorflow.keras.models import load_model
            model = load_model(str(model_path))
            self.logger.info(f"Loaded Keras model: {model_filename}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load Keras model {model_filename}: {e}")
            return None


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup consistent logger configuration
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Format: [2024-01-01 12:00:00] MODULE - LEVEL - Message
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


def safe_execute(default_return: Any = None, log_errors: bool = True):
    """
    Decorator for safe execution with error handling
    
    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator


def normalize_image(image: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize image for model input
    
    Args:
        image: Input image array
        method: Normalization method ('standard', 'mobilenet', 'zero_one')
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    
    if method == 'standard':
        # Standard normalization: (x - 127.5) / 128.0
        return (image - 127.5) / 128.0
    elif method == 'mobilenet':
        # MobileNet normalization: (x - 127.5) / 127.5
        return (image - 127.5) / 127.5
    elif method == 'zero_one':
        # Simple 0-1 normalization
        return image / 255.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    L2 normalize a vector (important for face embeddings)
    
    Args:
        vector: Input vector
        
    Returns:
        L2 normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    vec1_norm = l2_normalize(vec1)
    vec2_norm = l2_normalize(vec2)
    return float(np.dot(vec1_norm, vec2_norm))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(vec1 - vec2))


class FrameSkipper:
    """Helper class to skip frames for performance optimization"""
    
    def __init__(self, skip_interval: int = 5):
        """
        Initialize frame skipper
        
        Args:
            skip_interval: Process every Nth frame
        """
        self.skip_interval = skip_interval
        self.frame_count = 0
    
    def should_process(self) -> bool:
        """
        Check if current frame should be processed
        
        Returns:
            True if frame should be processed
        """
        self.frame_count += 1
        return self.frame_count % self.skip_interval == 0
    
    def reset(self):
        """Reset frame counter"""
        self.frame_count = 0


class PerformanceTracker:
    """Track performance metrics for optimization"""
    
    def __init__(self, name: str):
        """
        Initialize performance tracker
        
        Args:
            name: Name of the tracked component
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.execution_times = []
        self.max_samples = 100
    
    def record(self, execution_time: float):
        """
        Record execution time
        
        Args:
            execution_time: Time in seconds
        """
        self.execution_times.append(execution_time)
        if len(self.execution_times) > self.max_samples:
            self.execution_times.pop(0)
    
    def get_stats(self) -> dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary with min, max, avg, and latest execution times
        """
        if not self.execution_times:
            return {}
        
        return {
            'name': self.name,
            'min_ms': min(self.execution_times) * 1000,
            'max_ms': max(self.execution_times) * 1000,
            'avg_ms': np.mean(self.execution_times) * 1000,
            'latest_ms': self.execution_times[-1] * 1000,
            'samples': len(self.execution_times)
        }
    
    def log_stats(self):
        """Log performance statistics"""
        stats = self.get_stats()
        if stats:
            self.logger.info(
                f"{self.name} Performance: "
                f"avg={stats['avg_ms']:.2f}ms, "
                f"min={stats['min_ms']:.2f}ms, "
                f"max={stats['max_ms']:.2f}ms"
            )
