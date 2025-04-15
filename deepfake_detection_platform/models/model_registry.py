"""
Model registry for deepfake detection models.
This file defines metadata for available pre-trained models that can be used for analysis.
"""

import os
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# Base directory for model weights
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

# Define model registry with metadata for each available model
MODEL_REGISTRY = {
    "image": [
        {
            "id": "cnn_efficientnet",
            "name": "EfficientNet-B0 Deepfake Detector",
            "description": "CNN-based deepfake detector using EfficientNet-B0 architecture",
            "file_path": os.path.join(WEIGHTS_DIR, "image", "efficientnet_b0_deepfake.h5"),
            "framework": "tensorflow",
            "version": "1.0.0",
            "input_size": (224, 224),
            "performance": {
                "accuracy": 0.94,
                "f1_score": 0.93,
                "precision": 0.92,
                "recall": 0.94
            },
            "citation": "Efficient Deep Learning for Fake Media Detection, 2022"
        },
        {
            "id": "xception_faceforensics",
            "name": "Xception FaceForensics",
            "description": "Xception model trained on FaceForensics++ dataset for facial manipulation detection",
            "file_path": os.path.join(WEIGHTS_DIR, "image", "xception_faceforensics.h5"),
            "framework": "tensorflow",
            "version": "2.1.0",
            "input_size": (299, 299),
            "performance": {
                "accuracy": 0.96,
                "f1_score": 0.95,
                "precision": 0.94,
                "recall": 0.96
            },
            "citation": "FaceForensics++: Learning to Detect Manipulated Facial Images, 2019"
        },
        {
            "id": "resnet_meso4",
            "name": "MesoNet ResNet",
            "description": "ResNet-based MesoNet architecture for facial forgery detection",
            "file_path": os.path.join(WEIGHTS_DIR, "image", "mesonet_resnet.pth"),
            "framework": "pytorch",
            "version": "1.2.0",
            "input_size": (256, 256),
            "performance": {
                "accuracy": 0.93,
                "f1_score": 0.92,
                "precision": 0.91,
                "recall": 0.93
            },
            "citation": "MesoNet: a Compact Facial Video Forgery Detection Network, 2018"
        }
    ],
    
    "audio": [
        {
            "id": "wavlm_asvspoof",
            "name": "WavLM ASVSpoof",
            "description": "WavLM model fine-tuned on ASVSpoof dataset for synthetic speech detection",
            "file_path": os.path.join(WEIGHTS_DIR, "audio", "wavlm_asvspoof.pth"),
            "framework": "pytorch",
            "version": "1.0.0",
            "input_size": "variable",
            "performance": {
                "accuracy": 0.97,
                "f1_score": 0.96,
                "precision": 0.95,
                "recall": 0.97
            },
            "citation": "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing, 2022"
        },
        {
            "id": "lcnn_rawnet2",
            "name": "LCNN RawNet2",
            "description": "Light CNN combined with RawNet2 for voice deepfake detection",
            "file_path": os.path.join(WEIGHTS_DIR, "audio", "lcnn_rawnet2.pth"),
            "framework": "pytorch",
            "version": "1.1.0",
            "input_size": "variable",
            "performance": {
                "accuracy": 0.95,
                "f1_score": 0.94,
                "precision": 0.93,
                "recall": 0.95
            },
            "citation": "End-to-End Audio Deepfake Detection with LCNN-RawNet2, 2022"
        },
        {
            "id": "mel_spectrogram_lstm",
            "name": "Mel-Spectrogram LSTM",
            "description": "LSTM neural network operating on mel-spectrogram features for synthetic audio detection",
            "file_path": os.path.join(WEIGHTS_DIR, "audio", "melspec_lstm.h5"),
            "framework": "tensorflow",
            "version": "1.2.0",
            "input_size": (128, 128, 1),
            "performance": {
                "accuracy": 0.92,
                "f1_score": 0.91,
                "precision": 0.90,
                "recall": 0.92
            },
            "citation": "Mel-Spectrogram LSTM Architecture for Voice Deepfake Detection, 2021"
        }
    ],
    
    "video": [
        {
            "id": "slowfast_dfdc",
            "name": "SlowFast DFDC",
            "description": "SlowFast video understanding architecture fine-tuned on DFDC dataset",
            "file_path": os.path.join(WEIGHTS_DIR, "video", "slowfast_dfdc.pth"),
            "framework": "pytorch",
            "version": "1.0.0",
            "input_size": "variable",
            "performance": {
                "accuracy": 0.90,
                "f1_score": 0.89,
                "precision": 0.88,
                "recall": 0.90
            },
            "citation": "SlowFast Networks for Video Recognition, 2019"
        },
        {
            "id": "timesformer_deepfake",
            "name": "TimeSformer Deepfake",
            "description": "TimeSformer model for temporal video analysis of deepfakes",
            "file_path": os.path.join(WEIGHTS_DIR, "video", "timesformer_deepfake.pth"),
            "framework": "pytorch",
            "version": "1.1.0",
            "input_size": (224, 224),
            "performance": {
                "accuracy": 0.93,
                "f1_score": 0.92,
                "precision": 0.91,
                "recall": 0.93
            },
            "citation": "Is Space-Time Attention All You Need for Video Understanding?, 2021"
        },
        {
            "id": "convlstm_lippingnet",
            "name": "ConvLSTM LippingNet",
            "description": "ConvLSTM architecture for detecting lip-sync inconsistencies in deepfake videos",
            "file_path": os.path.join(WEIGHTS_DIR, "video", "convlstm_lippingnet.h5"),
            "framework": "tensorflow",
            "version": "1.2.0",
            "input_size": (128, 128, 3),
            "performance": {
                "accuracy": 0.91,
                "f1_score": 0.90,
                "precision": 0.89,
                "recall": 0.91
            },
            "citation": "LippingNet: Detecting Face Forgeries via Lip Synchronization Analysis, 2023"
        }
    ]
}

def get_available_models(media_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get metadata for available models.
    
    Args:
        media_type: Optional filter by media type (image, audio, video)
    
    Returns:
        Dictionary with model metadata by media type
    """
    if media_type:
        if media_type not in MODEL_REGISTRY:
            logger.warning(f"Unknown media type: {media_type}")
            return {}
        return {media_type: MODEL_REGISTRY[media_type]}
    
    return MODEL_REGISTRY

def get_model_by_id(model_id: str) -> Dict[str, Any]:
    """
    Get model metadata by model ID.
    
    Args:
        model_id: ID of the model to find
    
    Returns:
        Model metadata dictionary or None if not found
    """
    for media_type, models in MODEL_REGISTRY.items():
        for model in models:
            if model["id"] == model_id:
                return model
    
    logger.warning(f"Model not found: {model_id}")
    return None

def get_default_model(media_type: str) -> Dict[str, Any]:
    """
    Get the default model for a media type.
    
    Args:
        media_type: Media type (image, audio, video)
    
    Returns:
        Default model metadata dictionary or None if no models available
    """
    if media_type not in MODEL_REGISTRY or not MODEL_REGISTRY[media_type]:
        logger.warning(f"No models available for {media_type}")
        return None
    
    # Return the first model as default
    return MODEL_REGISTRY[media_type][0]
