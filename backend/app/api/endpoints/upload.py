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


@router.post("/upload")
async def upload_simple_file(file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=DetectionResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detection_params: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload a file for deepfake detection.
    
    - **file**: The file to analyze (image, audio, or video)
    - **detection_params**: Optional JSON string with detection parameters
    """
    # Print detailed debugging info
    print("=== DEBUG INFO ===")
    print(f"File name: {file.filename}")
    print(f"File content type: {file.content_type}")
    print(f"Raw detection_params: {detection_params}")
    print("=== END DEBUG INFO ===")
    
    # Parse detection parameters if provided
    params = {}
    if detection_params:
        try:
            params = json.loads(detection_params)
            print(f"Parsed parameters: {params}")
        except json.JSONDecodeError as e:
            print(f"Error parsing detection parameters: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid detection parameters format: {str(e)}"
            )
    
    # Ensure media_type is one of the allowed values
    media_type = params.get("media_type", "image")
    if media_type not in ["image", "audio", "video", "auto"]:
        media_type = "image"  # Default to image if not a valid type
    
    # Create detection params object
    params = {
        "media_type": media_type,
        "detailed_analysis": params.get("detailed_analysis", False),
        "confidence_threshold": params.get("confidence_threshold", settings.DEFAULT_CONFIDENCE_THRESHOLD)
    }
    
    # Determine media type if auto
    content_type = file.content_type
    if media_type == "auto":
        if content_type in settings.ALLOWED_IMAGE_TYPES:
            media_type = "image"
        elif content_type in settings.ALLOWED_AUDIO_TYPES:
            media_type = "audio"
        elif content_type in settings.ALLOWED_VIDEO_TYPES:
            media_type = "video"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Validate file type against declared media type
    if media_type == "image" and content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="File is not a supported image format")
    elif media_type == "audio" and content_type not in settings.ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=400, detail="File is not a supported audio format")
    elif media_type == "video" and content_type not in settings.ALLOWED_VIDEO_TYPES:
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
    if media_type == "image":
        # Validate image
        if not is_valid_image(file_path):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        task = celery_app.send_task(
            "app.tasks.image_tasks.detect_image",
            args=[file_path, file_hash, params["detailed_analysis"], params["confidence_threshold"]]
        )
        estimated_time = 2  # seconds
        
    elif media_type == "audio":
        # Validate audio
        if not is_valid_audio(file_path):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        task = celery_app.send_task(
            "app.tasks.audio_tasks.detect_audio",
            args=[file_path, file_hash, params["detailed_analysis"], params["confidence_threshold"]]
        )
        estimated_time = 5  # seconds
        
    elif media_type == "video":
        # Validate video
        if not is_valid_video(file_path):
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        task = celery_app.send_task(
            "app.tasks.video_tasks.detect_video",
            args=[file_path, file_hash, params["detailed_analysis"], params["confidence_threshold"]]
        )
        estimated_time = 10  # seconds
    
    return DetectionResponse(
        task_id=task.id,
        status="processing",
        media_type=media_type,
        estimated_time=estimated_time
    )
