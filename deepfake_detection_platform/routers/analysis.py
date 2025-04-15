from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import logging
import asyncio
import time
from datetime import datetime
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel

# Import utility functions and models
from utils.file_handling import get_file_info
from utils.preprocessing import preprocess_media
from models.image_models import analyze_image
from models.audio_models import analyze_audio
from models.video_models import analyze_video
from core.database import get_db_connection
from utils.model_loader import load_model
from models.model_registry import get_model_by_id, get_selected_model_id

# Set up router
router = APIRouter()
logger = logging.getLogger(__name__)

class AnalysisRequest(BaseModel):
    result_id: str
    media_type: str
    model_id: Optional[str] = None

@router.post("/{result_id}")
async def start_analysis(
    result_id: str,
    background_tasks: BackgroundTasks,
    media_type: Optional[str] = None,
    model_id: Optional[str] = None
):
    """
    Start the analysis process for a previously uploaded file.
    
    Args:
        result_id: ID of the upload to analyze
        media_type: Type of media (image, audio, video)
        model_id: Optional model ID to use for analysis
    
    Returns:
        JSON response with analysis status
    """
    try:
        # Check if the upload exists
        db = await get_db_connection()
        result = await db.execute(
            "SELECT filename, media_type, status FROM uploads WHERE id = ?",
            (result_id,)
        )
        record = await result.fetchone()
        
        if not record:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        # Check if analysis is already in progress or complete
        if record["status"] in ["processing", "complete"]:
            return {
                "result_id": result_id,
                "status": record["status"],
                "message": f"Analysis is already {record['status']}"
            }
        
        # Get file path
        media_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media")
        media_type_from_db = record["media_type"]
        # Use provided media_type if available, otherwise use from DB
        if not media_type:
            media_type = media_type_from_db
        file_path = os.path.join(media_dir, media_type_from_db, f"{result_id}{os.path.splitext(record['filename'])[1]}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Update status in database
        await db.execute(
            "UPDATE uploads SET status = ? WHERE id = ?",
            ("processing", result_id)
        )
        await db.commit()
        
        # Determine which model to use - use the one provided in the request
        
        # If no model specified in request, check if one was specified during upload
        if not model_id:
            # Check if we have a model ID stored in the database for this analysis
            model_id_row = await db.execute_fetchone(
                "SELECT model_id FROM uploads WHERE result_id = ?",
                (result_id,)
            )
            
            if model_id_row and model_id_row[0]:
                model_id = model_id_row[0]
        
        # If still no model_id, get the currently selected model for this media type
        if not model_id:
            model_id = get_selected_model_id(media_type)
        
        # Validate model if specified
        if model_id:
            model_info = get_model_by_id(model_id)
            if not model_info:
                error_msg = f"Model not found: {model_id}"
                await db.execute(
                    "UPDATE analysis_results SET status = ?, error = ? WHERE result_id = ?", 
                    ("failed", error_msg, result_id)
                )
                await db.commit()
                raise HTTPException(status_code=404, detail=error_msg)
            
            # Verify model type matches media type
            if model_info.get("type") != media_type:
                error_msg = f"Model type mismatch: {model_id} is not a {media_type} model"
                await db.execute(
                    "UPDATE analysis_results SET status = ?, error = ? WHERE result_id = ?", 
                    ("failed", error_msg, result_id)
                )
                await db.commit()
                raise HTTPException(status_code=400, detail=error_msg)
        
        # Load model
        try:
            model = await load_model(media_type, model_id)
        except Exception as e:
            # Update status to failed
            await db.execute(
                "UPDATE analysis_results SET status = ?, error = ? WHERE result_id = ?", 
                ("failed", str(e), result_id)
            )
            await db.commit()
            raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
        
        # Start analysis in background
        background_tasks.add_task(
            process_media,
            file_path=file_path,
            media_type=media_type,
            result_id=result_id,
            model=model
        )
        
        return {
            "result_id": result_id,
            "status": "processing",
            "message": "Analysis started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting analysis: {str(e)}")

@router.get("/{result_id}/status")
async def get_analysis_status(result_id: str):
    """
    Get the status of an analysis.
    
    Args:
        result_id: ID of the analysis
    
    Returns:
        JSON response with analysis status
    """
    try:
        db = await get_db_connection()
        result = await db.execute(
            """
            SELECT u.status, u.upload_time, r.completion_time, r.prediction, r.confidence
            FROM uploads u
            LEFT JOIN results r ON u.id = r.id
            WHERE u.id = ?
            """,
            (result_id,)
        )
        record = await result.fetchone()
        
        if not record:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        response = {
            "result_id": result_id,
            "status": record["status"],
            "upload_time": record["upload_time"]
        }
        
        if record["status"] == "complete":
            response.update({
                "completion_time": record["completion_time"],
                "prediction": record["prediction"],
                "confidence": record["confidence"]
            })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis status: {str(e)}")

async def process_media(file_path: str, media_type: str, result_id: str, model):
    """
    Process a media file for deepfake detection.
    This runs as a background task.
    
    Args:
        file_path: Path to the media file
        media_type: Type of media (image, audio, video)
        result_id: ID for the analysis result
        model: Loaded model to use for analysis
    """
    try:
        logger.info(f"Starting analysis for {result_id} ({media_type})")
        start_time = time.time()
        
        # Get file info
        file_info = get_file_info(file_path)
        
        # Preprocess the media
        processed_media = await preprocess_media(file_path, media_type)
        
        # Analyze based on media type
        if media_type == "image":
            results = await analyze_image(processed_media, model)
        elif media_type == "audio":
            results = await analyze_audio(processed_media, model)
        elif media_type == "video":
            results = await analyze_video(processed_media, model)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Calculate analysis duration
        analysis_duration = time.time() - start_time
        
        # Prepare the result data
        result_data = {
            "result_id": result_id,
            "media_type": media_type,
            "filename": os.path.basename(file_path),
            "prediction": results["prediction"],
            "confidence_score": results["confidence"],
            "timestamp": datetime.now().isoformat(),
            "analysis_duration": analysis_duration,
            "detection_details": results["details"]
        }
        
        # Add file info
        result_data["file_info"] = file_info
        
        # Store results in database with model info
        model_info = {"model_id": model.id} if model else {}
        results_with_model = {**results, "model": model_info}
        
        await save_results(result_id, results_with_model)
        
        # Save detailed results to a JSON file
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        result_file_path = os.path.join(results_dir, f"{result_id}.json")
        with open(result_file_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Analysis completed for {result_id} in {analysis_duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing media: {str(e)}", exc_info=True)
        
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

async def save_results(result_id: str, result_data: Dict[str, Any]):
    """
    Save analysis results to the database.
    
    Args:
        result_id: ID of the analysis
        result_data: Dictionary containing analysis results
    """
    try:
        db = await get_db_connection()
        
        # Update upload status
        await db.execute(
            "UPDATE uploads SET status = ? WHERE id = ?",
            ("complete", result_id)
        )
        
        # Save results
        await db.execute(
            """
            INSERT INTO results (
                id, prediction, confidence, completion_time, duration, details
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                result_data["prediction"],
                result_data["confidence_score"],
                result_data["timestamp"],
                result_data["analysis_duration"],
                json.dumps(result_data["detection_details"])
            )
        )
        
        await db.commit()
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}", exc_info=True)
        raise
