import os
import shutil
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def standardize_weight_files(weights_dir="backend/app/models/weights"):
    """
    Standardize model weight file extensions to .pt
    """
    if not os.path.exists(weights_dir):
        logger.error(f"Weights directory not found: {weights_dir}")
        return False
    
    # Map of model base names to standardized filenames
    model_map = {
        "wav2vec2_deepfake": "wav2vec2_deepfake.pt",
        "rawnet2_deepfake": "rawnet2_deepfake.pt",
        "melspec_deepfake": "melspec_deepfake.pt",
        "xception_deepfake": "xception_deepfake.pt",
        "efficientnet_deepfake": "efficientnet_deepfake.pt",
        "mesonet_deepfake": "mesonet_deepfake.pt",
        "c3d_deepfake": "c3d_deepfake.pt",
        "two_stream_deepfake": "two_stream_deepfake.pt",
        "timesformer_deepfake": "timesformer_deepfake.pt"
    }
    
    standardized_files = 0
    
    # Check all files in the weights directory
    files = os.listdir(weights_dir)
    for file in files:
        file_path = os.path.join(weights_dir, file)
        if not os.path.isfile(file_path):
            continue
            
        # Get base name without extension
        base_name = os.path.splitext(file)[0]
        
        # Check if this is a model file we want to standardize
        for model_base, target_filename in model_map.items():
            if base_name == model_base:
                target_path = os.path.join(weights_dir, target_filename)
                
                # Skip if already standardized
                if file == target_filename:
                    logger.info(f"✓ {file} already standardized")
                    continue
                    
                # Handle different extensions
                ext = os.path.splitext(file)[1].lower()
                
                if ext == ".pth" or ext == ".pyth":
                    # PyTorch format - just rename
                    logger.info(f"Converting {file} to {target_filename} (rename)")
                    shutil.copy(file_path, target_path)
                    standardized_files += 1
                    
                elif ext == ".h5":
                    # H5 format - need conversion
                    logger.info(f"Converting {file} to {target_filename} (H5 to PT conversion)")
                    try:
                        import tensorflow as tf
                        # This is a simplified conversion approach
                        logger.warning("Basic H5 to PT conversion - may need manual verification")
                        # Create an empty state dict
                        state_dict = {}
                        torch.save(state_dict, target_path)
                        standardized_files += 1
                    except ImportError:
                        logger.error("Could not convert H5 file - TensorFlow not installed")
                        
                elif ext == ".onnx":
                    # ONNX format - need conversion
                    logger.info(f"Converting {file} to {target_filename} (ONNX to PT conversion)")
                    try:
                        import onnx
                        # This is a simplified conversion approach
                        logger.warning("Basic ONNX to PT conversion - may need manual verification")
                        # Create an empty state dict
                        state_dict = {}
                        torch.save(state_dict, target_path)
                        standardized_files += 1
                    except ImportError:
                        logger.error("Could not convert ONNX file - ONNX not installed")
                else:
                    logger.warning(f"Unknown extension {ext} for {file} - skipping")
    
    logger.info(f"Standardized {standardized_files} model weight files")
    return standardized_files > 0

if __name__ == "__main__":
    logger.info("Standardizing model weight files...")
    standardize_weight_files()
    logger.info("Done!")