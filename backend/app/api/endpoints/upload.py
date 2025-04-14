import os
import shutil
import hashlib
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional

from app.core.config import settings
from app.db.session import get_db
from app.tasks.celery_app import celery_app
from app.schemas.detection import DetectionRequest, DetectionResponse
from app.utils.image_utils import is_valid_image
from app.utils.audio_utils import is_valid_audio
from app.utils.video_utils import is_valid_video


router = APIRouter()


def get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            h.update(chunk)
    return h.hexdigest()


@router.post("/upload", response_model=DetectionResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    media_type: str = Form("image"),  # Default to image instead of auto
    detailed_analysis: bool = Form(False),
    confidence_threshold: float = Form(settings.DEFAULT_CONFIDENCE_THRESHOLD),
    detection_params: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # Print debugging info
    print(f"Received media_type: {media_type}, detailed_analysis: {detailed_analysis}")
    
    # Ensure media_type is one of the allowed values
    if media_type not in ["image", "audio", "video"]:
        media_type = "image"  # Default to image if not a valid type
    
    print(f"Using media_type: {media_type}")
    
    # Create detection params object
    detection_params = DetectionRequest(
        media_type=media_type,
        detailed_analysis=detailed_analysis,
        confidence_threshold=confidence_threshold
    )
    
    # Determine media type if auto
    content_type = file.content_type
    if detection_params.media_type == "auto":
        if content_type in settings.ALLOWED_IMAGE_TYPES:
            detection_params.media_type = "image"
        elif content_type in settings.ALLOWED_AUDIO_TYPES:
            detection_params.media_type = "audio"
        elif content_type in settings.ALLOWED_VIDEO_TYPES:
            detection_params.media_type = "video"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Validate file type against declared media type
    if detection_params.media_type == "image" and content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="File is not a supported image format")
    elif detection_params.media_type == "audio" and content_type not in settings.ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=400, detail="File is not a supported audio format")
    elif detection_params.media_type == "video" and content_type not in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="File is not a supported video format")
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Calculate file hash
    file_hash = get_file_hash(file_path)
    
    # Start appropriate detection task based on media type
    if detection_params.media_type == "image":
        # Validate image
        if not is_valid_image(file_path):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        task = celery_app.send_task(
            "app.tasks.image_tasks.detect_image",
            args=[file_path, file_hash, detection_params.detailed_analysis, detection_params.confidence_threshold]
        )
        estimated_time = 2  # seconds
        
    elif detection_params.media_type == "audio":
        # Validate audio
        if not is_valid_audio(file_path):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        task = celery_app.send_task(
            "app.tasks.audio_tasks.detect_audio",
            args=[file_path, file_hash, detection_params.detailed_analysis, detection_params.confidence_threshold]
        )
        estimated_time = 5  # seconds
        
    elif detection_params.media_type == "video":
        # Validate video
        if not is_valid_video(file_path):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        task = celery_app.send_task(
            "app.tasks.video_tasks.detect_video",
            args=[file_path, file_hash, detection_params.detailed_analysis, detection_params.confidence_threshold]
        )
        estimated_time = 10  # seconds
    
    return DetectionResponse(
        task_id=task.id,
        status="processing",
        media_type=detection_params.media_type,
        estimated_time=estimated_time
    )
