import os
import shutil
import logging
import aiofiles
import mimetypes
import magic
from fastapi import UploadFile
from typing import Dict, Any, Optional, Tuple
import time
from datetime import datetime

from core.config import settings

logger = logging.getLogger(__name__)

async def save_uploaded_file(file: UploadFile, media_type: str, result_id: str) -> str:
    """
    Save an uploaded file to the appropriate directory.
    
    Args:
        file: The uploaded file object
        media_type: Type of media (image, audio, video)
        result_id: ID for the result
    
    Returns:
        Path to the saved file
    """
    # Determine the directory path based on media type
    media_dir = os.path.join(settings.MEDIA_DIR, media_type)
    os.makedirs(media_dir, exist_ok=True)
    
    # Get file extension from original filename
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    # Generate target filename using the result_id
    target_filename = f"{result_id}{file_ext}"
    target_path = os.path.join(media_dir, target_filename)
    
    try:
        # Save the file using aiofiles for async I/O
        async with aiofiles.open(target_path, 'wb') as f:
            # Read and write in chunks to avoid loading large files into memory
            CHUNK_SIZE = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                await f.write(chunk)
        
        logger.info(f"Saved uploaded file to {target_path}")
        return target_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}", exc_info=True)
        # Clean up if file was partially created
        if os.path.exists(target_path):
            os.remove(target_path)
        raise

def validate_file(file: UploadFile, media_type: str) -> Dict[str, Any]:
    """
    Validate an uploaded file.
    
    Args:
        file: The uploaded file object
        media_type: Type of media (image, audio, video)
    
    Returns:
        Dictionary with validation result
    """
    try:
        # Check if the file is empty
        if not file.filename:
            return {
                "valid": False,
                "message": "No file provided"
            }
        
        # Validate file size
        file_size = get_file_size(file)
        max_size = get_max_file_size(media_type)
        
        if file_size > max_size:
            return {
                "valid": False,
                "message": f"File too large. Maximum size for {media_type} is {max_size/(1024*1024):.1f}MB, got {file_size/(1024*1024):.1f}MB"
            }
        
        # Validate file type
        content_type = file.content_type
        
        # Get allowed types based on media_type
        allowed_types = get_allowed_types(media_type)
        
        if content_type not in allowed_types:
            return {
                "valid": False,
                "message": f"Invalid file type: {content_type}. Allowed types for {media_type}: {', '.join(allowed_types)}"
            }
        
        return {
            "valid": True,
            "message": "File is valid"
        }
        
    except Exception as e:
        logger.error(f"Error validating file: {str(e)}", exc_info=True)
        return {
            "valid": False,
            "message": f"Error validating file: {str(e)}"
        }

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dictionary with file information
    """
    try:
        # Get file stats
        stats = os.stat(file_path)
        
        # Get file extension and guess media type
        file_ext = os.path.splitext(file_path)[1].lower()
        media_type = guess_media_type(file_path)
        
        # Basic file info
        file_info = {
            "filename": os.path.basename(file_path),
            "size": stats.st_size,
            "size_human": format_file_size(stats.st_size),
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "extension": file_ext,
            "media_type": media_type
        }
        
        # Add mime type using python-magic if available
        try:
            mime = magic.Magic(mime=True)
            file_info["mime_type"] = mime.from_file(file_path)
        except:
            # Fallback to mimetypes module
            file_info["mime_type"] = mimetypes.guess_type(file_path)[0] or "unknown"
        
        # Get specific metadata based on media type
        if media_type == "image":
            image_info = get_image_info(file_path)
            if image_info:
                file_info.update(image_info)
        elif media_type == "audio":
            audio_info = get_audio_info(file_path)
            if audio_info:
                file_info.update(audio_info)
        elif media_type == "video":
            video_info = get_video_info(file_path)
            if video_info:
                file_info.update(video_info)
        
        return file_info
        
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}", exc_info=True)
        return {
            "filename": os.path.basename(file_path),
            "error": str(e)
        }

def get_file_size(file: UploadFile) -> int:
    """
    Get the size of an uploaded file in bytes.
    
    Args:
        file: The uploaded file object
    
    Returns:
        File size in bytes
    """
    # Try to get the size from file.file if available
    try:
        if hasattr(file, "file"):
            file_obj = file.file
            if hasattr(file_obj, "seek") and hasattr(file_obj, "tell"):
                # Save current position
                current_pos = file_obj.tell()
                
                # Seek to end and get position
                file_obj.seek(0, 2)
                size = file_obj.tell()
                
                # Restore position
                file_obj.seek(current_pos)
                
                return size
    except:
        pass
    
    # Fallback: if we can't determine size, assume it's within limits
    return 0

def get_max_file_size(media_type: str) -> int:
    """
    Get the maximum allowed file size for a media type in bytes.
    
    Args:
        media_type: Type of media (image, audio, video)
    
    Returns:
        Maximum file size in bytes
    """
    if media_type == "image":
        return settings.MAX_IMAGE_SIZE * 1024 * 1024  # Convert MB to bytes
    elif media_type == "audio":
        return settings.MAX_AUDIO_SIZE * 1024 * 1024
    elif media_type == "video":
        return settings.MAX_VIDEO_SIZE * 1024 * 1024
    else:
        # Default: 10MB
        return 10 * 1024 * 1024

def get_allowed_types(media_type: str) -> list:
    """
    Get allowed MIME types for a media type.
    
    Args:
        media_type: Type of media (image, audio, video)
    
    Returns:
        List of allowed MIME types
    """
    if media_type == "image":
        return settings.ALLOWED_IMAGE_TYPES
    elif media_type == "audio":
        return settings.ALLOWED_AUDIO_TYPES
    elif media_type == "video":
        return settings.ALLOWED_VIDEO_TYPES
    else:
        return []

def guess_media_type(file_path: str) -> str:
    """
    Guess the media type of a file based on its extension.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Media type string (image, audio, video, or unknown)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    video_extensions = ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv']
    
    if ext in image_extensions:
        return "image"
    elif ext in audio_extensions:
        return "audio"
    elif ext in video_extensions:
        return "video"
    else:
        return "unknown"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in a human-readable format.
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        Human-readable file size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

# Media-specific info functions
def get_image_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about an image file.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        Dictionary with image information
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        with Image.open(file_path) as img:
            info = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "is_animated": getattr(img, "is_animated", False),
                "n_frames": getattr(img, "n_frames", 1)
            }
            
            # Extract EXIF data if available
            exif_data = {}
            if hasattr(img, "_getexif") and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except:
                            value = str(value)
                    exif_data[tag] = value
                
                info["exif"] = exif_data
            
            return info
    except Exception as e:
        logger.warning(f"Could not extract image info: {str(e)}")
        return None

def get_audio_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about an audio file.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Dictionary with audio information
    """
    try:
        import wave
        import contextlib
        
        if file_path.lower().endswith('.wav'):
            with contextlib.closing(wave.open(file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                
                return {
                    "channels": f.getnchannels(),
                    "sample_width": f.getsampwidth(),
                    "frame_rate": rate,
                    "n_frames": frames,
                    "duration": duration,
                    "duration_human": format_duration(duration)
                }
        else:
            # For non-WAV files, try to use librosa or just return basic info
            return {
                "format": os.path.splitext(file_path)[1][1:].upper()
            }
    except Exception as e:
        logger.warning(f"Could not extract audio info: {str(e)}")
        return None

def get_video_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a video file.
    
    Args:
        file_path: Path to the video file
    
    Returns:
        Dictionary with video information
    """
    try:
        # Try using OpenCV to extract video metadata
        import cv2
        
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Extract basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        # Release the capture object
        cap.release()
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "duration_human": format_duration(duration),
            "format": os.path.splitext(file_path)[1][1:].upper()
        }
    except Exception as e:
        logger.warning(f"Could not extract video info: {str(e)}")
        return None

def format_duration(seconds: float) -> str:
    """
    Format duration in a human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Human-readable duration string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"
    else:
        return f"{int(m)}:{s:05.2f}"
