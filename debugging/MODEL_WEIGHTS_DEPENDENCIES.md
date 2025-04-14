# Model Weights and Dependencies Management Guide

## Problem Description

After reviewing the DeepFake Detection Platform codebase, we've identified issues with model weight file handling and dependency management that could be causing inconsistent behavior:

### 1. Inconsistent Model Weight File Extensions

In the `backend/app/models/audio_models.py` file, the `get_audio_model` function uses inconsistent file extensions for different model weights:

```python
if model_type == "wav2vec2":
    model = Wav2Vec2Model()
    weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "wav2vec2_deepfake.pt")  # .pt extension
elif model_type == "rawnet2":
    model = RawNet2()
    weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "rawnet2_deepfake.pth")  # .pth extension
elif model_type == "melspec":
    model = MelSpecResNet()
    weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "melspec_deepfake.onnx")  # .onnx extension
```

This inconsistency creates several problems:
- Confusion when saving models during training
- Potential compatibility issues when loading models
- Increased chance of using the wrong model format

### 2. Weak Error Handling for Missing Model Weights

The current implementation uses a warning when model weights are not found, and then proceeds with default initialization:

```python
if os.path.exists(weights_path):
    logger.info(f"Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
else:
    logger.warning(f"Weights file not found at {weights_path}, using model with default initialization")
```

This silent fallback can lead to:
- Using untrained models without clear indication to the user
- False results being presented with high confidence
- Difficult-to-diagnose behavior in production

### 3. External Dependencies Issues

The system relies on several external dependencies, with insufficient validation:

1. **FFmpeg**: Required for audio extraction from videos
2. **Transformers Library**: Required for Wav2Vec2 model
3. **PyTorch**: Required for all models, with potential version compatibility issues

## Solution Approach

### 1. Standardize Model Weight File Extensions

#### Recommended File Extensions:

- PyTorch models: Use `.pt` extension consistently
- ONNX models: Use `.onnx` extension only for models explicitly exported to ONNX format

#### Implementation:

```python
def get_audio_model(model_type=None):
    """
    Get or initialize the audio detection model.
    
    Args:
        model_type (str, optional): Type of model to use. If None, uses the default from settings.
    
    Returns:
        nn.Module: The loaded model
    """
    if model_type is None:
        model_type = settings.AUDIO_MODEL_TYPE
    
    # Check if model is already loaded
    if model_type in _audio_models:
        return _audio_models[model_type]
    
    # Create model based on type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for audio model")
    
    if model_type == "wav2vec2":
        model = Wav2Vec2Model()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "wav2vec2_deepfake.pt")
    elif model_type == "rawnet2":
        model = RawNet2()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "rawnet2_deepfake.pt")  # Changed to .pt
    elif model_type == "melspec":
        model = MelSpecResNet()
        # Only use .onnx if the model is specifically in ONNX format, otherwise use .pt
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "melspec_deepfake.pt")  # Changed to .pt
    else:
        raise ValueError(f"Unsupported audio model type: {model_type}")
    
    # Load weights if available with improved error handling
    if os.path.exists(weights_path):
        try:
            logger.info(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded weights for {model_type} model")
        except Exception as e:
            logger.error(f"Failed to load weights for {model_type} model: {str(e)}")
            raise RuntimeError(f"Error loading model weights for {model_type}: {str(e)}")
    else:
        # In production, we should raise an error if weights are missing
        if settings.ENVIRONMENT == "production":
            raise FileNotFoundError(f"Model weights file not found at {weights_path}")
        else:
            logger.warning(f"Weights file not found at {weights_path}, using model with default initialization")
            logger.warning("This will likely result in poor detection performance")
    
    model = model.to(device)
    model.eval()
    
    # Cache the model
    _audio_models[model_type] = model
    
    return model
```

### 2. Improve Dependency Validation

Create a system health check function that validates all required dependencies:

```python
def validate_system_dependencies():
    """
    Validate that all required system dependencies are available.
    
    Raises:
        RuntimeError: If any required dependency is missing
    """
    missing_deps = []
    
    # Check for FFmpeg
    import shutil
    if shutil.which("ffmpeg") is None:
        missing_deps.append("FFmpeg (required for audio extraction from videos)")
    
    # Check for PyTorch
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        missing_deps.append("PyTorch")
    
    # Check for Transformers
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        missing_deps.append("Transformers (required for Wav2Vec2 model)")
    
    # Check for librosa
    try:
        import librosa
        logger.info(f"Librosa version: {librosa.__version__}")
    except ImportError:
        missing_deps.append("Librosa (required for audio processing)")
    
    # Raise error if any dependencies are missing
    if missing_deps:
        error_msg = "Missing required dependencies: " + ", ".join(missing_deps)
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("All system dependencies validated successfully")
    return True
```

Add this validation to the application startup in `app/core/events.py`:

```python
from app.models.audio_models import validate_system_dependencies

async def startup_event():
    # Validate system dependencies
    try:
        validate_system_dependencies()
    except Exception as e:
        logger.error(f"System dependency validation failed: {str(e)}")
        # You might want to exit here in production or let it continue with limited functionality
```

### 3. Create Model Weights Management Scripts

Create a script to download and verify model weights:

```python
#!/usr/bin/env python
# scripts/download_weights.py

import os
import sys
import argparse
import hashlib
import requests
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model weights information
MODEL_WEIGHTS = {
    "wav2vec2_deepfake.pt": {
        "url": "https://example.com/models/wav2vec2_deepfake.pt",  # Replace with actual URL
        "md5": "abcdef1234567890abcdef1234567890",  # Replace with actual MD5
        "size": 102400000  # Approximate size in bytes
    },
    "rawnet2_deepfake.pt": {
        "url": "https://example.com/models/rawnet2_deepfake.pt",  # Replace with actual URL
        "md5": "1234567890abcdef1234567890abcdef",  # Replace with actual MD5
        "size": 25600000  # Approximate size in bytes
    },
    "melspec_deepfake.pt": {
        "url": "https://example.com/models/melspec_deepfake.pt",  # Replace with actual URL
        "md5": "567890abcdef1234567890abcdef1234",  # Replace with actual MD5
        "size": 51200000  # Approximate size in bytes
    }
}

def calculate_md5(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, filepath, expected_size=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0)) or expected_size
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "wb") as file, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def download_weights(weights_dir):
    """Download all model weights."""
    os.makedirs(weights_dir, exist_ok=True)
    
    for filename, info in MODEL_WEIGHTS.items():
        filepath = os.path.join(weights_dir, filename)
        
        # Check if file already exists and has correct MD5
        if os.path.exists(filepath):
            logger.info(f"Checking {filename}...")
            if calculate_md5(filepath) == info["md5"]:
                logger.info(f"✅ {filename} already exists and MD5 matches. Skipping download.")
                continue
            else:
                logger.warning(f"⚠️ {filename} exists but MD5 doesn't match. Re-downloading...")
        
        # Download the file
        try:
            logger.info(f"Downloading {filename}...")
            download_file(info["url"], filepath, info["size"])
            
            # Verify MD5
            if calculate_md5(filepath) == info["md5"]:
                logger.info(f"✅ {filename} downloaded and verified successfully.")
            else:
                logger.error(f"❌ {filename} MD5 verification failed!")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {str(e)}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download model weights for DeepFake Detection Platform")
    parser.add_argument("--weights-dir", default="app/models/weights", 
                        help="Directory to store the model weights")
    args = parser.parse_args()
    
    download_weights(args.weights_dir)
    logger.info("All model weights downloaded and verified successfully!")

if __name__ == "__main__":
    main()
```

### 4. Add Model Health-Check Before Processing

Add a health-check function to verify models before processing:

```python
def check_model_health(model_type=None):
    """
    Check if the model is healthy and ready for inference.
    
    Args:
        model_type (str, optional): Type of model to check. If None, checks the default.
        
    Returns:
        bool: True if the model is healthy, False otherwise
    """
    try:
        if model_type is None:
            model_type = settings.AUDIO_MODEL_TYPE
        
        # Generate a simple test input
        if model_type == "wav2vec2":
            # Generate 1 second of silence at 16kHz
            test_input = torch.zeros(1, 16000)
        elif model_type == "rawnet2":
            # Generate 1 second of silence at 16kHz with batch and channel dims
            test_input = torch.zeros(1, 1, 16000)
        elif model_type == "melspec":
            # Generate a dummy spectrogram
            test_input = torch.zeros(1, 1, 128, 32)
        else:
            logger.error(f"Unknown model type for health check: {model_type}")
            return False
        
        # Get the model
        model = get_audio_model(model_type)
        
        # Move test input to the same device as the model
        device = next(model.parameters()).device
        test_input = test_input.to(device)
        
        # Run inference
        with torch.no_grad():
            _ = model(test_input)
        
        logger.info(f"Model health check passed for {model_type}")
        return True
    
    except Exception as e:
        logger.error(f"Model health check failed for {model_type}: {str(e)}")
        return False
```

Add this health check to the detection function:

```python
def detect_deepfake_audio(audio_path: str, model_type=None, detailed=False) -> Dict[str, Any]:
    """
    Detect if an audio file is a deepfake.
    
    Args:
        audio_path (str): Path to the audio file
        model_type (str, optional): Type of model to use
        detailed (bool): Whether to return detailed analysis
    
    Returns:
        Dict: Detection results
    """
    if model_type is None:
        model_type = settings.AUDIO_MODEL_TYPE
    
    # Check model health before proceeding
    if not check_model_health(model_type):
        logger.error(f"Model {model_type} failed health check, cannot process audio")
        return {
            "error": f"Model {model_type} is not healthy",
            "is_fake": False,
            "confidence_score": 0.0
        }
    
    # Rest of the function...
```

## Implementation Plan

1. **Update Model Handling**:
   - Standardize all model weight file extensions to `.pt` format
   - Improve error handling for missing weights
   - Add health checks before processing

2. **Create Dependency Validation**:
   - Implement `validate_system_dependencies()` function
   - Add validation to application startup
   - Add clear error messages for missing dependencies

3. **Add Weight Management**:
   - Create script for downloading and validating weights
   - Document model weights sources and verification
   - Include version compatibility information

4. **Testing**:
   - Test model loading with correct and incorrect weights
   - Verify dependency validation
   - Test health check functionality

## Benefits

1. **Improved Reliability**: More robust error handling prevents silent failures
2. **Better User Experience**: Clear error messages when dependencies or weights are missing
3. **Consistency**: Standard file extensions and validation processes
4. **Maintainability**: Easier to update models and weights in the future
5. **Debuggability**: Better logging and health checks make issues easier to diagnose

## Conclusion

By implementing these changes, we'll address the model weight inconsistencies and dependency issues in the DeepFake Detection Platform. This will significantly improve the reliability and maintainability of the system, making it more robust in production environments.
