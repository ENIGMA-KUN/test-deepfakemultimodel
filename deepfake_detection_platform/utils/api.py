import streamlit as st
import requests
import json
import os
import uuid
from typing import Dict, Any, Optional, List
import time

# Define API base URL - configure based on your FastAPI backend location
API_URL = os.environ.get("API_URL", "http://localhost:8000")

def send_file_for_analysis(file, media_type: str, model_id: str = None) -> Optional[str]:
    """
    Sends a file to the backend API for deepfake analysis.
    
    Args:
        file: The file object from st.file_uploader
        media_type: Type of media (image, audio, video)
        model_id: Optional ID of the model to use for analysis
    
    Returns:
        result_id: ID of the analysis result or None if failed
    """
    try:
        # Create a unique ID for this analysis
        result_id = str(uuid.uuid4())
        
        # Prepare the file for upload
        files = {"file": (file.name, file.getvalue(), get_content_type(file.name))}
        
        # Prepare additional data
        data = {
            "media_type": media_type,
            "result_id": result_id
        }
        
        # Add model_id if specified
        if model_id:
            data["model_id"] = model_id
        
        # Send the file to the API
        response = requests.post(
            f"{API_URL}/upload",
            files=files,
            data=data
        )
        
        # Handle the response
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("result_id")
        else:
            st.error(f"Error sending file: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error sending file for analysis: {str(e)}")
        return None

def get_analysis_result(result_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves analysis results from the backend API.
    
    Args:
        result_id: ID of the analysis result
    
    Returns:
        Dictionary with analysis results or None if not found
    """
    try:
        # For demo purposes, simulate a delay if it's a sample result
        if result_id.startswith("sample"):
            return generate_sample_result(result_id)
        
        # Real API request for actual results
        response = requests.get(f"{API_URL}/results/{result_id}")
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            # Analysis still in progress
            progress = response.json().get("progress", 0)
            st.warning(f"Analysis in progress: {progress}% complete. Please wait...")
            time.sleep(1)  # Add a small delay before checking again
            return None
        else:
            st.error(f"Error retrieving results: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error getting analysis results: {str(e)}")
        
        # For demonstration purposes, return sample results if API is not available
        if os.environ.get("DEMO_MODE", "false").lower() == "true":
            return generate_sample_result(result_id)
        
        return None

def get_analysis_history() -> List[Dict[str, Any]]:
    """
    Retrieves analysis history from the backend API.
    
    Returns:
        List of dictionaries with analysis history
    """
    try:
        # For demo purposes, check if we're in demo mode
        if os.environ.get("DEMO_MODE", "false").lower() == "true":
            # Return sample history
            return generate_sample_history()
        
        # Real API request
        response = requests.get(f"{API_URL}/results/history")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error retrieving history: {response.text}")
            return []
            
    except Exception as e:
        st.error(f"Error getting analysis history: {str(e)}")
        
        # For demonstration purposes, return sample history if API is not available
        return generate_sample_history()

def get_content_type(filename: str) -> str:
    """
    Determines the content type based on file extension.
    
    Args:
        filename: Name of the file
    
    Returns:
        Content type string
    """
    ext = os.path.splitext(filename)[1].lower()
    
    content_types = {
        # Image formats
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        
        # Audio formats
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        
        # Video formats
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska'
    }
    
    return content_types.get(ext, 'application/octet-stream')

# Sample data generation for demo or testing purposes
def generate_sample_result(result_id: str) -> Dict[str, Any]:
    """
    Generates a sample result for demonstration purposes.
    
    Args:
        result_id: ID of the sample result
    
    Returns:
        Dictionary with sample analysis results
    """
    # Extract media type from the sample ID if possible
    if "_" in result_id:
        sample_number = result_id.split("_")[-1]
        if sample_number == "0":
            media_type = "image"
        elif sample_number == "1":
            media_type = "audio"
        else:
            media_type = "video"
    else:
        media_type = "image"  # Default to image
    
    # Randomize some aspects but keep the structure consistent
    import random
    from datetime import datetime
    
    is_fake = random.choice([True, False])
    confidence = 0.5 + (random.random() * 0.5) if is_fake else 0.6 + (random.random() * 0.4)
    
    result = {
        "result_id": result_id,
        "media_type": media_type,
        "prediction": "Fake" if is_fake else "Real",
        "confidence_score": confidence,
        "timestamp": datetime.now().isoformat(),
        "filename": f"sample_{media_type}.{media_type[:3]}",
        "analysis_duration": round(random.uniform(1.5, 5.2), 2),
        "detection_details": {
            "technical_explanation": get_sample_explanation(media_type, is_fake),
            "features": generate_sample_features(media_type, is_fake),
            "metadata": generate_sample_metadata(media_type)
        }
    }
    
    return result

def generate_sample_history() -> List[Dict[str, Any]]:
    """
    Generates sample analysis history for demonstration purposes.
    
    Returns:
        List of dictionaries with sample analysis history
    """
    import random
    from datetime import datetime, timedelta
    
    # Generate 10 random historical records
    history = []
    
    for i in range(10):
        # Alternate between media types
        media_types = ["image", "audio", "video"]
        media_type = media_types[i % 3]
        
        # Randomize real/fake
        is_fake = random.choice([True, False])
        confidence = 0.5 + (random.random() * 0.5) if is_fake else 0.6 + (random.random() * 0.4)
        
        # Random timestamp within the last 30 days
        days_ago = random.randint(0, 30)
        timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
        
        # Sample extensions based on media type
        extensions = {
            "image": ["jpg", "png"],
            "audio": ["mp3", "wav"],
            "video": ["mp4", "mov"]
        }
        
        ext = random.choice(extensions[media_type])
        
        record = {
            "id": f"sample_{i}",
            "timestamp": timestamp,
            "filename": f"sample_file_{i}.{ext}",
            "media_type": media_type,
            "prediction": "Fake" if is_fake else "Real",
            "confidence_score": confidence
        }
        
        history.append(record)
    
    # Sort by timestamp, newest first
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return history

def get_sample_explanation(media_type: str, is_fake: bool) -> str:
    """
    Returns a sample technical explanation based on media type and prediction.
    
    Args:
        media_type: Type of media
        is_fake: Whether the result is fake
    
    Returns:
        Sample explanation string
    """
    if media_type == "image":
        if is_fake:
            return """
            The image analysis detected several telltale signs of manipulation:
            
            1. Inconsistent noise patterns across different regions of the image
            2. Subtle artifacts around facial features, particularly in the eye and mouth regions
            3. Unusual color distribution in skin tones
            4. Irregular shadow patterns that don't match the apparent light source
            
            These artifacts are consistent with GAN-generated content, specifically using techniques like StyleGAN2 or similar architectures.
            """
        else:
            return """
            The image analysis found no significant indicators of manipulation:
            
            1. Noise patterns are consistent throughout the image
            2. No detectable artifacts in key regions
            3. Natural color distribution and skin tone variations
            4. Shadow and lighting patterns are physically plausible
            
            The natural imperfections and consistent noise patterns across the image strongly suggest this is an authentic photograph.
            """
    
    elif media_type == "audio":
        if is_fake:
            return """
            The audio analysis detected several indicators of synthetic voice generation:
            
            1. Unnatural cadence and rhythm in speech patterns
            2. Missing or unnatural breathing sounds between phrases
            3. Spectral artifacts typical of voice synthesis algorithms
            4. Inconsistent formant frequencies across the audio sample
            
            These patterns are consistent with text-to-speech or voice cloning technologies, likely using WaveNet or similar neural vocoder architectures.
            """
        else:
            return """
            The audio analysis found no significant indicators of voice synthesis:
            
            1. Natural speech cadence and rhythm
            2. Present and consistent breathing patterns
            3. No spectral artifacts typically associated with synthesis
            4. Consistent and natural formant transitions
            
            The minor imperfections and natural variations in the speech pattern strongly suggest this is authentic human speech.
            """
    
    else:  # video
        if is_fake:
            return """
            The video analysis detected multiple indicators of deepfake manipulation:
            
            1. Temporal inconsistencies in facial movements
            2. Unnatural blinking patterns
            3. Poor synchronization between lip movements and audio
            4. Boundary artifacts around the face region
            5. Inconsistent lighting effects on facial features
            
            These artifacts are consistent with face-swapping or facial reenactment technologies, likely using methods similar to DeepFaceLab or FaceSwap.
            """
        else:
            return """
            The video analysis found no significant indicators of deepfake manipulation:
            
            1. Consistent temporal coherence in facial movements
            2. Natural blinking patterns and micro-expressions
            3. Proper synchronization between lip movements and audio
            4. No detectable boundary artifacts or blending issues
            5. Consistent lighting effects across frames
            
            The natural imperfections and consistent motion patterns strongly suggest this is authentic video footage.
            """

def generate_sample_features(media_type: str, is_fake: bool) -> Dict[str, float]:
    """
    Generates sample detection features based on media type and prediction.
    
    Args:
        media_type: Type of media
        is_fake: Whether the result is fake
    
    Returns:
        Dictionary of feature names and scores
    """
    import random
    
    # Base random factor - higher for fake content
    base = 0.7 if is_fake else 0.3
    
    # Add some randomness but keep it aligned with the prediction
    def fake_score():
        return min(0.95, base + random.uniform(-0.1, 0.2))
    
    def real_score():
        return max(0.05, base - random.uniform(-0.1, 0.2))
    
    if media_type == "image":
        return {
            "Noise Consistency": real_score() if not is_fake else fake_score(),
            "Facial Landmark Coherence": real_score() if not is_fake else fake_score(),
            "Color Distribution": real_score() if not is_fake else fake_score(),
            "ELA Analysis": real_score() if not is_fake else fake_score(),
            "Texture Consistency": real_score() if not is_fake else fake_score()
        }
    
    elif media_type == "audio":
        return {
            "Spectral Consistency": real_score() if not is_fake else fake_score(),
            "Breathing Patterns": real_score() if not is_fake else fake_score(),
            "Formant Analysis": real_score() if not is_fake else fake_score(),
            "Phoneme Transitions": real_score() if not is_fake else fake_score(),
            "Background Noise": real_score() if not is_fake else fake_score()
        }
    
    else:  # video
        return {
            "Temporal Coherence": real_score() if not is_fake else fake_score(),
            "Blink Detection": real_score() if not is_fake else fake_score(),
            "Lip Sync Accuracy": real_score() if not is_fake else fake_score(),
            "Facial Boundary Analysis": real_score() if not is_fake else fake_score(),
            "Pulse Detection": real_score() if not is_fake else fake_score()
        }

def get_models(media_type: str = None) -> Dict[str, Any]:
    """
    Get available models for deepfake detection, optionally filtered by media type.
    
    Args:
        media_type: Optional filter for media type (image, audio, video)
    
    Returns:
        Dictionary with models metadata
    """
    try:
        # Construct URL with optional filter
        url = f"{API_URL}/models/"
        if media_type:
            url += f"?media_type={media_type}"
        
        # Make API request
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error retrieving models: {response.text}")
            return {}
            
    except Exception as e:
        st.error(f"Error getting models: {str(e)}")
        
        # Return sample data for demonstration
        return generate_sample_models()

def select_model(model_id: str) -> Dict[str, Any]:
    """
    Select a model for a specific media type.
    
    Args:
        model_id: ID of the model to select
    
    Returns:
        Dictionary with selection response
    """
    try:
        # Make API request to select model
        response = requests.post(f"{API_URL}/models/{model_id}/select")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error selecting model: {response.text}")
            return {"status": "error", "message": response.text}
            
    except Exception as e:
        st.error(f"Error selecting model: {str(e)}")
        return {"status": "error", "message": str(e)}

def get_selected_model(media_type: str) -> Dict[str, Any]:
    """
    Get the currently selected model for a media type.
    
    Args:
        media_type: Media type (image, audio, video)
    
    Returns:
        Dictionary with selected model information
    """
    try:
        # Make API request to get selected model
        response = requests.get(f"{API_URL}/models/selected/{media_type}")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting selected model: {response.text}")
            return {"status": "error", "message": response.text}
            
    except Exception as e:
        st.error(f"Error getting selected model: {str(e)}")
        return {"status": "error", "message": str(e)}

def get_model_details(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: ID of the model
    
    Returns:
        Dictionary with model details
    """
    try:
        # Make API request to get model details
        response = requests.get(f"{API_URL}/models/{model_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting model details: {response.text}")
            return {"error": response.text}
            
    except Exception as e:
        st.error(f"Error getting model details: {str(e)}")
        return {"error": str(e)}

def generate_sample_models() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generates sample model data for demonstration purposes when API is not available.
    
    Returns:
        Dictionary with sample model data
    """
    return {
        "image": [
            {
                "id": "cnn_efficientnet",
                "name": "EfficientNet-B0 Deepfake Detector",
                "description": "CNN-based deepfake detector using EfficientNet-B0 architecture",
                "framework": "tensorflow",
                "version": "1.0.0",
                "selected": True,
                "performance": {
                    "accuracy": 0.94,
                    "f1_score": 0.93,
                    "precision": 0.92,
                    "recall": 0.94
                }
            },
            {
                "id": "xception_faceforensics",
                "name": "Xception FaceForensics",
                "description": "Xception model trained on FaceForensics++ dataset",
                "framework": "tensorflow",
                "version": "2.1.0",
                "selected": False,
                "performance": {
                    "accuracy": 0.96,
                    "f1_score": 0.95,
                    "precision": 0.94,
                    "recall": 0.96
                }
            }
        ],
        "audio": [
            {
                "id": "wavlm_asvspoof",
                "name": "WavLM ASVSpoof",
                "description": "WavLM model fine-tuned on ASVSpoof dataset",
                "framework": "pytorch",
                "version": "1.0.0",
                "selected": True,
                "performance": {
                    "accuracy": 0.97,
                    "f1_score": 0.96,
                    "precision": 0.95,
                    "recall": 0.97
                }
            },
            {
                "id": "mel_spectrogram_lstm",
                "name": "Mel-Spectrogram LSTM",
                "description": "LSTM network on mel-spectrogram features",
                "framework": "tensorflow",
                "version": "1.2.0",
                "selected": False,
                "performance": {
                    "accuracy": 0.92,
                    "f1_score": 0.91,
                    "precision": 0.90,
                    "recall": 0.92
                }
            }
        ],
        "video": [
            {
                "id": "slowfast_dfdc",
                "name": "SlowFast DFDC",
                "description": "SlowFast network fine-tuned on DFDC dataset",
                "framework": "pytorch",
                "version": "1.0.0",
                "selected": True,
                "performance": {
                    "accuracy": 0.90,
                    "f1_score": 0.89,
                    "precision": 0.88,
                    "recall": 0.90
                }
            },
            {
                "id": "convlstm_lippingnet",
                "name": "ConvLSTM LippingNet",
                "description": "ConvLSTM for lip-sync inconsistency detection",
                "framework": "tensorflow",
                "version": "1.2.0",
                "selected": False,
                "performance": {
                    "accuracy": 0.91,
                    "f1_score": 0.90,
                    "precision": 0.89,
                    "recall": 0.91
                }
            }
        ]
    }

def generate_sample_metadata(media_type: str) -> Dict[str, Any]:
    """
    Generates sample metadata based on media type.
    
    Args:
        media_type: Type of media
    
    Returns:
        Dictionary of metadata
    """
    import random
    from datetime import datetime, timedelta
    
    # Common metadata
    created_date = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y:%m:%d %H:%M:%S")
    software = random.choice(["Adobe Photoshop", "GIMP", "Lightroom", "Camera Raw", "None"])
    
    if media_type == "image":
        return {
            "Camera": random.choice(["iPhone 14 Pro", "Canon EOS R5", "Sony A7IV", "Google Pixel 7", "Unknown"]),
            "Resolution": random.choice(["3024x4032", "4000x3000", "2048x1536", "1920x1080"]),
            "Created": created_date,
            "Software": software,
            "GPS": None if random.random() > 0.3 else {"Latitude": f"{random.uniform(25, 48):.6f}", "Longitude": f"{random.uniform(-123, -70):.6f}"},
            "ColorSpace": random.choice(["sRGB", "Adobe RGB", "ProPhoto RGB"]),
            "ExifVersion": "0231"
        }
    
    elif media_type == "audio":
        return {
            "Format": random.choice(["MP3", "WAV", "FLAC", "AAC"]),
            "Duration": f"{random.randint(10, 300)} seconds",
            "Bitrate": f"{random.choice([128, 192, 256, 320])} kbps",
            "SampleRate": f"{random.choice([44100, 48000, 96000])} Hz",
            "Channels": random.choice([1, 2]),
            "Created": created_date,
            "Software": random.choice(["Audacity", "Adobe Audition", "Logic Pro", "GarageBand", "None"]),
            "ID3Tags": {
                "Title": "Sample Audio",
                "Artist": "Unknown",
                "Album": "Demo"
            }
        }
    
    else:  # video
        return {
            "Format": random.choice(["MP4", "MOV", "AVI", "WebM"]),
            "Codec": random.choice(["H.264", "H.265", "VP9", "AV1"]),
            "Duration": f"{random.randint(5, 120)} seconds",
            "Framerate": f"{random.choice([24, 30, 60])} fps",
            "Resolution": random.choice(["1920x1080", "3840x2160", "1280x720"]),
            "Bitrate": f"{random.randint(1, 15)} Mbps",
            "Created": created_date,
            "Software": random.choice(["Adobe Premiere", "Final Cut Pro", "DaVinci Resolve", "None"]),
            "Camera": random.choice(["iPhone 14 Pro", "Canon EOS R5", "Sony A7IV", "GoPro Hero 10", "Unknown"])
        }
