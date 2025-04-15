from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
import logging
from typing import Optional
import time
from datetime import datetime

# Import utility functions
from utils.file_handling import validate_file, save_uploaded_file
from utils.model_loader import load_model
from models.model_registry import get_model_by_id
from core.database import get_db_connection

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    media_type: str = Form(...),
    result_id: Optional[str] = Form(None),
    model_id: Optional[str] = Form(None)
):
    """
    Upload a file for deepfake analysis.
    
    Args:
        file: The file to analyze
        media_type: Type of media (image, audio, video)
        result_id: Optional custom ID for the result
        model_id: Optional ID for the model to use
    
    Returns:
        JSON response with result_id and status
    """
    try:
        # Validate the file
        validation_result = validate_file(file, media_type)
        if not validation_result["valid"]:
            logger.warning(f"Invalid file upload: {validation_result['message']}")
            raise HTTPException(status_code=400, detail=validation_result["message"])
        
        # Generate a result ID if not provided
        if not result_id:
            result_id = str(uuid.uuid4())
        
        # Save the file
        file_path = await save_uploaded_file(file, media_type, result_id)
        
        # Load the appropriate model
        try:
            # If model_id is provided, validate it
            if model_id:
                model_info = get_model_by_id(model_id)
                if not model_info:
                    raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
                
                # Check if model type matches media type
                if model_info.get("type") != media_type:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Model type mismatch: {model_id} is not a {media_type} model"
                    )
            
            # Load the model with optional model_id
            model = await load_model(media_type, model_id)
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

        # Record the upload in the database
        db = await get_db_connection()
        await db.execute(
            """
            INSERT INTO uploads (id, filename, media_type, upload_time, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                result_id, 
                file.filename, 
                media_type, 
                datetime.now().isoformat(),
                "pending"
            )
        )
        await db.commit()
        
        # Trigger analysis in the background
        background_tasks.add_task(
            trigger_analysis, 
            file_path=file_path,
            media_type=media_type,
            result_id=result_id
        )
        
        return {
            "result_id": result_id,
            "status": "success",
            "message": f"File uploaded successfully. Analysis is in progress."
        }
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

async def trigger_analysis(file_path: str, media_type: str, result_id: str):
    """
    Triggers the analysis process for the uploaded file.
    This runs as a background task after file upload.
    
    Args:
        file_path: Path to the uploaded file
        media_type: Type of media (image, audio, video)
        result_id: ID for the analysis result
    """
    try:
        # Import here to avoid circular imports
        from routers.analysis import process_media
        
        # Update status in database
        db = await get_db_connection()
        await db.execute(
            "UPDATE uploads SET status = ? WHERE id = ?",
            ("processing", result_id)
        )
        await db.commit()
        
        # Start the analysis
        await process_media(file_path, media_type, result_id)
        
    except Exception as e:
        logger.error(f"Error triggering analysis: {str(e)}", exc_info=True)
        
        # Update status to error
        try:
            db = await get_db_connection()
            await db.execute(
                "UPDATE uploads SET status = ? WHERE id = ?",
                ("error", result_id)
            )
            await db.commit()
        except Exception as db_error:
            logger.error(f"Error updating database: {str(db_error)}")

@router.get("/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """
    Get the status of an uploaded file.
    
    Args:
        upload_id: ID of the upload
    
    Returns:
        JSON response with upload status
    """
    try:
        db = await get_db_connection()
        result = await db.execute(
            "SELECT status, upload_time FROM uploads WHERE id = ?",
            (upload_id,)
        )
        record = await result.fetchone()
        
        if not record:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {
            "id": upload_id,
            "status": record["status"],
            "upload_time": record["upload_time"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving upload status: {str(e)}")
