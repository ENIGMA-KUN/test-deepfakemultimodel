import os
import logging
import numpy as np
from typing import Any, Dict, Optional, Union
import tempfile
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)

async def preprocess_media(file_path: str, media_type: str) -> Any:
    """
    Preprocess media for analysis by deepfake detection models.
    
    Args:
        file_path: Path to the media file
        media_type: Type of media (image, audio, video)
    
    Returns:
        Preprocessed media data ready for model input
    """
    try:
        if media_type == "image":
            return await preprocess_image(file_path)
        elif media_type == "audio":
            return await preprocess_audio(file_path)
        elif media_type == "video":
            return await preprocess_video(file_path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
            
    except Exception as e:
        logger.error(f"Error preprocessing {media_type}: {str(e)}", exc_info=True)
        raise

async def preprocess_image(file_path: str) -> np.ndarray:
    """
    Preprocess an image for analysis.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        Preprocessed image as a numpy array
    """
    try:
        # Read the image
        img = cv2.imread(file_path)
        
        if img is None:
            # Try using PIL if OpenCV fails
            with Image.open(file_path) as pil_img:
                # Convert to RGB if needed
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img = np.array(pil_img)
                # Convert from RGB to BGR for OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize to a standard size for model input
        target_size = (224, 224)  # Common input size for many models
        img_resized = cv2.resize(img, target_size)
        
        # Convert to RGB for model input (many models expect RGB)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return {
            "processed": img_batch,
            "original": img,
            "original_path": file_path
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}", exc_info=True)
        raise

async def preprocess_audio(file_path: str) -> Dict[str, Any]:
    """
    Preprocess an audio file for analysis.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Dictionary with preprocessed audio data
    """
    try:
        # Try to use librosa for audio preprocessing
        try:
            import librosa
            import librosa.display
            
            # Load audio file (limit to 30 seconds for efficiency)
            y, sr = librosa.load(file_path, sr=None, duration=30)
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1]
            mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            # Resize to a standard size for model input
            target_size = (128, 128)  # Example size, adjust based on model requirements
            
            # Ensure spectrogram is the right size
            if mel_spec_normalized.shape[1] < target_size[1]:
                # Pad if too short
                pad_width = target_size[1] - mel_spec_normalized.shape[1]
                mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncate if too long
                mel_spec_normalized = mel_spec_normalized[:, :target_size[1]]
            
            # Resize to target height
            mel_spec_resized = cv2.resize(mel_spec_normalized, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Add channel and batch dimensions
            mel_spec_batch = np.expand_dims(np.expand_dims(mel_spec_resized, axis=0), axis=3)
            
            return {
                "processed": mel_spec_batch,
                "waveform": y,
                "sample_rate": sr,
                "original_path": file_path
            }
            
        except ImportError:
            logger.warning("Librosa not available, using fallback audio preprocessing")
            # Fallback to a simpler preprocessing if librosa is not available
            return {
                "processed": np.zeros((1, 128, 128, 1)),  # Dummy data
                "original_path": file_path
            }
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}", exc_info=True)
        raise

async def preprocess_video(file_path: str) -> Dict[str, Any]:
    """
    Preprocess a video file for analysis.
    
    Args:
        file_path: Path to the video file
    
    Returns:
        Dictionary with preprocessed video data
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames at regular intervals
        # For efficiency, we'll extract a maximum of 30 frames
        max_frames = 30
        frame_interval = max(1, frame_count // max_frames)
        
        frames = []
        frame_positions = []
        
        for i in range(0, frame_count, frame_interval):
            if len(frames) >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Resize frame
            target_size = (224, 224)  # Common input size for many models
            frame_resized = cv2.resize(frame, target_size)
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            frames.append(frame_normalized)
            frame_positions.append(i)
        
        cap.release()
        
        if not frames:
            raise Exception("Could not extract frames from video")
        
        # Stack frames into a batch
        frames_batch = np.stack(frames)
        
        return {
            "processed": frames_batch,
            "frame_positions": frame_positions,
            "fps": fps,
            "frame_count": frame_count,
            "original_path": file_path
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing video: {str(e)}", exc_info=True)
        raise

def extract_faces(image: np.ndarray) -> Dict[str, Any]:
    """
    Extract faces from an image for facial forensics analysis.
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Dictionary with extracted faces and their locations
    """
    try:
        # Try to use OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Extract face regions
        face_images = []
        face_locations = []
        
        for (x, y, w, h) in faces:
            # Add some margin around the face
            margin = int(0.2 * w)
            
            # Calculate coordinates with margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            
            # Resize to standard size
            face_resized = cv2.resize(face_img, (224, 224))
            
            # Convert to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            face_images.append(face_normalized)
            face_locations.append((x1, y1, x2, y2))
        
        # If no faces found, use the whole image
        if not face_images:
            # Resize the whole image
            image_resized = cv2.resize(image, (224, 224))
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            image_normalized = image_rgb.astype(np.float32) / 255.0
            
            face_images = [image_normalized]
            face_locations = [(0, 0, image.shape[1], image.shape[0])]
        
        return {
            "faces": face_images,
            "locations": face_locations,
            "count": len(face_images)
        }
        
    except Exception as e:
        logger.error(f"Error extracting faces: {str(e)}", exc_info=True)
        
        # Return empty result if face extraction fails
        return {
            "faces": [],
            "locations": [],
            "count": 0,
            "error": str(e)
        }

def extract_audio_from_video(video_path: str) -> Optional[str]:
    """
    Extract audio track from a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Path to the extracted audio file or None if failed
    """
    try:
        # Create a temporary file for the audio
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # Try using OpenCV to get video info
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Release the capture object
        cap.release()
        
        # Try using ffmpeg for audio extraction
        import subprocess
        
        # Construct ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            temp_audio_path,
            "-y"  # Overwrite if exists
        ]
        
        # Run the command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if the output file exists and has content
        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
            return temp_audio_path
        else:
            logger.warning(f"Audio extraction produced empty file: {temp_audio_path}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return None
        
    except Exception as e:
        logger.error(f"Error extracting audio from video: {str(e)}", exc_info=True)
        
        # Clean up temporary file if it exists
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return None
