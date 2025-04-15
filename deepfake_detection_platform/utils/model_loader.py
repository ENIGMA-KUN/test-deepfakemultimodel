import os
import logging
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional, Union
import time

from core.config import settings
from models.model_registry import get_available_models, get_model_by_id, get_default_model

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading models
model_cache = {}
# Track selected model for each type
selected_models = {}

async def load_model(model_type: str, model_id: str = None) -> Any:
    """
    Load a machine learning model for deepfake detection.
    
    Args:
        model_type: Type of model to load (image, audio, video)
        model_id: Optional ID of specific model to load
    
    Returns:
        Loaded model object
    """
    # Create cache key that includes both model type and specific model ID
    cache_key = f"{model_type}_{model_id}" if model_id else model_type
    
    # Check if model is already in cache
    if cache_key in model_cache:
        logger.debug(f"Using cached model: {cache_key}")
        return model_cache[cache_key]
    
    # Demo mode checks
    if settings.DEMO_MODE:
        logger.info(f"Running in demo mode, returning dummy {model_type} model")
        dummy_model = create_dummy_model(model_type)
        model_cache[cache_key] = dummy_model
        return dummy_model
    
    # Get model metadata based on ID or default
    model_metadata = None
    if model_id:
        model_metadata = get_model_by_id(model_id)
        if model_metadata:
            # Store as selected model for this type
            selected_models[model_type] = model_id
    
    # If no specific model requested or not found, use default
    if not model_metadata:
        # Try to get model from settings first
        model_path = get_model_path(model_type)
        
        if model_path and os.path.exists(model_path):
            # Use model from settings
            logger.info(f"Using {model_type} model from settings: {model_path}")
        else:
            # Try to get default model from registry
            model_metadata = get_default_model(model_type)
            
            if not model_metadata or not os.path.exists(model_metadata["file_path"]):
                logger.warning(f"No {model_type} model available, using dummy model")
                dummy_model = create_dummy_model(model_type)
                model_cache[cache_key] = dummy_model
                return dummy_model
                
            model_path = model_metadata["file_path"]
    else:
        # Use specified model from registry
        model_path = model_metadata["file_path"]
        logger.info(f"Using {model_type} model: {model_metadata['name']}")
        
    # Check if the model file exists
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}, using dummy model")
        dummy_model = create_dummy_model(model_type)
        model_cache[cache_key] = dummy_model
        return dummy_model
    
    try:
        # Based on file extension, load the appropriate model type
        start_time = time.time()
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext == '.pkl' or file_ext == '.pickle':
            # Load pickle model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        elif file_ext == '.h5' or file_ext == '.keras':
            # Load Keras model
            try:
                from tensorflow import keras
                model = keras.models.load_model(model_path)
            except ImportError:
                raise ImportError("TensorFlow/Keras is required to load .h5/.keras models")
        
        elif file_ext == '.pt' or file_ext == '.pth':
            # Load PyTorch model
            try:
                import torch
                model = torch.load(model_path, map_location=torch.device('cuda' if settings.USE_GPU else 'cpu'))
            except ImportError:
                raise ImportError("PyTorch is required to load .pt/.pth models")
        
        elif file_ext == '.onnx':
            # Load ONNX model
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if settings.USE_GPU else ['CPUExecutionProvider']
                model = ort.InferenceSession(model_path, providers=providers)
            except ImportError:
                raise ImportError("ONNX Runtime is required to load .onnx models")
        
        elif file_ext == '.tflite':
            # Load TFLite model
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                model = interpreter
            except ImportError:
                raise ImportError("TensorFlow Lite is required to load .tflite models")
        
        else:
            raise ValueError(f"Unsupported model format: {file_ext}")
        
        # Cache the model
        model_cache[model_type] = model
        
        logger.info(f"Loaded {model_type} model from {model_path} in {time.time() - start_time:.2f} seconds")
        return model
        
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {str(e)}", exc_info=True)
        
        # Fall back to dummy model
        logger.warning(f"Falling back to dummy {model_type} model")
        dummy_model = create_dummy_model(model_type)
        model_cache[model_type] = dummy_model
        return dummy_model

def get_model_path(model_type: str) -> Optional[str]:
    """
    Get the path to a model file based on model type.
    
    Args:
        model_type: Type of model (image, audio, video)
    
    Returns:
        Path to the model file or None if not found
    """
    if model_type == "image":
        return settings.IMAGE_MODEL_PATH
    elif model_type == "audio":
        return settings.AUDIO_MODEL_PATH
    elif model_type == "video":
        return settings.VIDEO_MODEL_PATH
    else:
        return None

def create_dummy_model(model_type: str) -> Any:
    """
    Create a dummy model for demonstration or when a real model is unavailable.
    
    Args:
        model_type: Type of model to create
    
    Returns:
        Dummy model object
    """
    # Create a simple callable class that mimics model prediction
    class DummyModel:
        def __init__(self, model_type):
            self.model_type = model_type
        
        def predict(self, input_data):
            # Simulate processing delay
            time.sleep(0.5)
            
            # For demonstration, return random predictions
            import random
            import hashlib
            
            # Make predictions deterministic based on input hash
            if isinstance(input_data, np.ndarray):
                input_hash = hashlib.md5(input_data.tobytes()).hexdigest()
            else:
                input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
            
            # Use hash to seed random generator for consistent outputs
            random.seed(int(input_hash[:8], 16))
            
            # Generate a confidence score between 0.6 and 0.95
            confidence = 0.6 + (random.random() * 0.35)
            
            # 50% chance of being classified as real or fake
            is_fake = random.random() > 0.5
            
            # Return prediction in the format expected by the application
            return {
                "prediction": "Fake" if is_fake else "Real",
                "confidence": confidence,
                "details": self._generate_details(is_fake, confidence)
            }
        
        def _generate_details(self, is_fake, confidence):
            # Generate plausible detection details
            import random
            
            # Create features with values that align with the prediction
            features = {}
            if self.model_type == "image":
                features = {
                    "Noise Patterns": self._aligned_score(is_fake, 0.1),
                    "Facial Inconsistency": self._aligned_score(is_fake, 0.15),
                    "ELA Analysis": self._aligned_score(is_fake, 0.12),
                    "Metadata Consistency": self._aligned_score(is_fake, 0.08),
                    "Compression Artifacts": self._aligned_score(is_fake, 0.05)
                }
            elif self.model_type == "audio":
                features = {
                    "Spectral Analysis": self._aligned_score(is_fake, 0.1),
                    "Voice Coherence": self._aligned_score(is_fake, 0.15),
                    "Background Noise": self._aligned_score(is_fake, 0.12),
                    "Temporal Patterns": self._aligned_score(is_fake, 0.08),
                    "Formant Analysis": self._aligned_score(is_fake, 0.05)
                }
            elif self.model_type == "video":
                features = {
                    "Facial Movement": self._aligned_score(is_fake, 0.1),
                    "Lip Sync": self._aligned_score(is_fake, 0.15),
                    "Temporal Coherence": self._aligned_score(is_fake, 0.12),
                    "Blinking Patterns": self._aligned_score(is_fake, 0.08),
                    "Visual Artifacts": self._aligned_score(is_fake, 0.05)
                }
            
            # Generate explanation text
            if is_fake:
                explanation = f"The analysis detected several indicators of manipulation in the {self.model_type}. "
                explanation += "The patterns observed are consistent with AI-generated or manipulated content. "
                explanation += f"The confidence score of {confidence:.2f} is based on multiple detection features, "
                explanation += "with the strongest indicators being in the texture and consistency patterns."
            else:
                explanation = f"The analysis found no significant indicators of manipulation in the {self.model_type}. "
                explanation += "The patterns observed are consistent with authentic content. "
                explanation += f"The confidence score of {confidence:.2f} is based on multiple detection features, "
                explanation += "all of which indicate natural variations typical of genuine content."
            
            return {
                "technical_explanation": explanation,
                "features": features,
                "model_version": "dummy-1.0.0"
            }
        
        def _aligned_score(self, is_fake, variation):
            # Generate a score that aligns with the fake/real classification
            import random
            
            # For fake: higher scores (0.6-0.9)
            # For real: lower scores (0.1-0.4)
            base = 0.7 if is_fake else 0.25
            return base + (random.random() * variation * 2) - variation
    
    return DummyModel(model_type)

def get_selected_model_id(model_type: str) -> Optional[str]:
    """
    Get the currently selected model ID for a media type.
    
    Args:
        model_type: Media type (image, audio, video)
    
    Returns:
        Selected model ID or None if no model selected
    """
    return selected_models.get(model_type)

def get_model_info(model_type: str = None, model_id: str = None) -> Dict[str, Any]:
    """
    Get information about available models.
    
    Args:
        model_type: Optional filter by media type
        model_id: Optional specific model ID
    
    Returns:
        Dictionary with model information
    """
    if model_id:
        # Return info for specific model
        model = get_model_by_id(model_id)
        if model:
            return {
                "id": model["id"],
                "name": model["name"],
                "description": model["description"],
                "framework": model["framework"],
                "version": model["version"],
                "performance": model["performance"],
                "citation": model["citation"]
            }
        return {"error": f"Model not found: {model_id}"}
    
    # Get all models by type
    available_models = get_available_models(model_type)
    
    # Format response
    result = {}
    
    for media_type, models in available_models.items():
        model_list = []
        for model in models:
            model_list.append({
                "id": model["id"],
                "name": model["name"],
                "description": model["description"],
                "framework": model["framework"],
                "version": model["version"],
                "selected": model["id"] == selected_models.get(media_type)
            })
        result[media_type] = model_list
    
    return result

async def unload_models():
    """
    Unload all models from memory.
    This is useful when shutting down the application or needing to free memory.
    """
    global model_cache
    global selected_models
    
    logger.info("Unloading all models from memory")
    
    # Clear the cache
    model_cache.clear()
    
    # Clear selected models
    selected_models.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    logger.info("All models unloaded")
