from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
import json
from typing import List, Optional
import base64
from datetime import datetime

# Import utility functions
from core.database import get_db_connection

# Set up router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{result_id}")
async def get_result(result_id: str, include_media: bool = False):
    """
    Get the results of an analysis.
    
    Args:
        result_id: ID of the analysis
        include_media: Whether to include the media content in the response
    
    Returns:
        JSON response with analysis results
    """
    try:
        # Check if this is a sample result for demonstrations
        if result_id.startswith("sample"):
            return generate_sample_result(result_id)
        
        # Check if the result exists in the database
        db = await get_db_connection()
        result = await db.execute(
            """
            SELECT u.filename, u.media_type, u.upload_time, 
                   r.prediction, r.confidence, r.completion_time, r.duration, r.details
            FROM uploads u
            LEFT JOIN results r ON u.id = r.id
            WHERE u.id = ?
            """,
            (result_id,)
        )
        record = await result.fetchone()
        
        if not record:
            raise HTTPException(status_code=404, detail="Result not found")
        
        # Check if analysis is complete
        if record["prediction"] is None:
            # Analysis still in progress
            return JSONResponse(
                status_code=202,
                content={
                    "message": "Analysis in progress",
                    "result_id": result_id,
                    "progress": await get_analysis_progress(result_id)
                }
            )
        
        # Build the basic response
        response = {
            "result_id": result_id,
            "media_type": record["media_type"],
            "filename": record["filename"],
            "prediction": record["prediction"],
            "confidence_score": record["confidence"],
            "timestamp": record["completion_time"],
            "analysis_duration": record["duration"]
        }
        
        # Add detection details
        if record["details"]:
            response["detection_details"] = json.loads(record["details"])
        
        # Check if detailed result file exists
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        result_file_path = os.path.join(results_dir, f"{result_id}.json")
        
        if os.path.exists(result_file_path):
            # Load additional details from the result file
            with open(result_file_path, "r") as f:
                detailed_results = json.load(f)
                
                # Add any additional fields not already in the response
                for key, value in detailed_results.items():
                    if key not in response and key != "media_content":
                        response[key] = value
        
        # Add media content if requested
        if include_media:
            media_path = await get_media_path(result_id, record["media_type"], record["filename"])
            
            if media_path and os.path.exists(media_path):
                if record["media_type"] == "image":
                    # For images, encode as base64
                    with open(media_path, "rb") as img_file:
                        response["media_content"] = base64.b64encode(img_file.read()).decode("utf-8")
                else:
                    # For audio and video, just include the path
                    response["media_path"] = media_path
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting result: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving result: {str(e)}")

@router.get("/")
async def get_all_results(
    skip: int = 0, 
    limit: int = 10,
    media_type: Optional[str] = None,
    prediction: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get all analysis results with pagination and filtering.
    
    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        media_type: Filter by media type
        prediction: Filter by prediction (Real/Fake)
        start_date: Filter by date range start (ISO format)
        end_date: Filter by date range end (ISO format)
    
    Returns:
        JSON response with paginated results
    """
    try:
        db = await get_db_connection()
        
        # Build the SQL query with filters
        query = """
        SELECT u.id, u.filename, u.media_type, u.upload_time,
               r.prediction, r.confidence, r.completion_time
        FROM uploads u
        LEFT JOIN results r ON u.id = r.id
        WHERE u.status = 'complete'
        """
        params = []
        
        # Apply filters
        if media_type:
            query += " AND u.media_type = ?"
            params.append(media_type)
        
        if prediction:
            query += " AND r.prediction = ?"
            params.append(prediction)
        
        if start_date:
            query += " AND r.completion_time >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND r.completion_time <= ?"
            params.append(end_date)
        
        # Add ordering and pagination
        query += " ORDER BY r.completion_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, skip])
        
        # Execute the query
        result = await db.execute(query, params)
        records = await result.fetchall()
        
        # Get total count (without pagination)
        count_query = """
        SELECT COUNT(*) as total
        FROM uploads u
        LEFT JOIN results r ON u.id = r.id
        WHERE u.status = 'complete'
        """
        count_params = []
        
        # Apply the same filters to the count query
        if media_type:
            count_query += " AND u.media_type = ?"
            count_params.append(media_type)
        
        if prediction:
            count_query += " AND r.prediction = ?"
            count_params.append(prediction)
        
        if start_date:
            count_query += " AND r.completion_time >= ?"
            count_params.append(start_date)
        
        if end_date:
            count_query += " AND r.completion_time <= ?"
            count_params.append(end_date)
        
        count_result = await db.execute(count_query, count_params)
        count_record = await count_result.fetchone()
        total = count_record["total"] if count_record else 0
        
        # Format the results
        results = []
        for record in records:
            results.append({
                "id": record["id"],
                "filename": record["filename"],
                "media_type": record["media_type"],
                "prediction": record["prediction"],
                "confidence_score": record["confidence"],
                "timestamp": record["completion_time"]
            })
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error getting all results: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

@router.get("/history")
async def get_analysis_history(limit: int = 20):
    """
    Get the analysis history for the current user.
    
    Args:
        limit: Maximum number of records to return
    
    Returns:
        JSON response with analysis history
    """
    try:
        db = await get_db_connection()
        result = await db.execute(
            """
            SELECT u.id, u.filename, u.media_type, u.upload_time,
                   r.prediction, r.confidence, r.completion_time
            FROM uploads u
            LEFT JOIN results r ON u.id = r.id
            WHERE u.status = 'complete'
            ORDER BY r.completion_time DESC
            LIMIT ?
            """,
            (limit,)
        )
        records = await result.fetchall()
        
        # Format the results
        history = []
        for record in records:
            history.append({
                "id": record["id"],
                "timestamp": record["completion_time"],
                "filename": record["filename"],
                "media_type": record["media_type"],
                "prediction": record["prediction"],
                "confidence_score": record["confidence"]
            })
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis history: {str(e)}")

@router.get("/media/{result_id}")
async def get_media_file(result_id: str):
    """
    Get the media file for a result.
    
    Args:
        result_id: ID of the analysis
    
    Returns:
        Media file as a response
    """
    try:
        # Get media info from database
        db = await get_db_connection()
        result = await db.execute(
            "SELECT filename, media_type FROM uploads WHERE id = ?",
            (result_id,)
        )
        record = await result.fetchone()
        
        if not record:
            raise HTTPException(status_code=404, detail="Result not found")
        
        # Get the file path
        media_path = await get_media_path(result_id, record["media_type"], record["filename"])
        
        if not media_path or not os.path.exists(media_path):
            raise HTTPException(status_code=404, detail="Media file not found")
        
        # Return the file
        return FileResponse(
            media_path,
            filename=record["filename"],
            media_type=get_content_type(record["media_type"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting media file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving media file: {str(e)}")

@router.delete("/{result_id}")
async def delete_result(result_id: str):
    """
    Delete an analysis result.
    
    Args:
        result_id: ID of the analysis
    
    Returns:
        JSON response with deletion status
    """
    try:
        # Check if the result exists
        db = await get_db_connection()
        result = await db.execute(
            "SELECT filename, media_type FROM uploads WHERE id = ?",
            (result_id,)
        )
        record = await result.fetchone()
        
        if not record:
            raise HTTPException(status_code=404, detail="Result not found")
        
        # Delete from database
        await db.execute("DELETE FROM results WHERE id = ?", (result_id,))
        await db.execute("DELETE FROM uploads WHERE id = ?", (result_id,))
        await db.commit()
        
        # Delete the media file
        media_path = await get_media_path(result_id, record["media_type"], record["filename"])
        if media_path and os.path.exists(media_path):
            os.remove(media_path)
        
        # Delete the result file
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        result_file_path = os.path.join(results_dir, f"{result_id}.json")
        if os.path.exists(result_file_path):
            os.remove(result_file_path)
        
        return {
            "status": "success",
            "message": f"Result {result_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting result: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting result: {str(e)}")

# Helper functions
async def get_media_path(result_id: str, media_type: str, filename: str) -> str:
    """
    Get the path to a media file.
    
    Args:
        result_id: ID of the analysis
        media_type: Type of media
        filename: Original filename
    
    Returns:
        Path to the media file
    """
    media_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media")
    type_dir = os.path.join(media_dir, media_type)
    
    # Check if directory exists
    if not os.path.exists(type_dir):
        os.makedirs(type_dir)
    
    # Get file extension from original filename
    ext = os.path.splitext(filename)[1]
    
    # Construct path with result_id as filename
    return os.path.join(type_dir, f"{result_id}{ext}")

async def get_analysis_progress(result_id: str) -> int:
    """
    Get the progress of an ongoing analysis.
    
    Args:
        result_id: ID of the analysis
    
    Returns:
        Progress percentage (0-100)
    """
    # This is a placeholder. In a real implementation, you'd have a way to track
    # progress of the analysis job (e.g., in a database or a cache)
    
    # For now, return a random progress value
    import random
    return random.randint(10, 90)

def get_content_type(media_type: str) -> str:
    """
    Get the content type for a media type.
    
    Args:
        media_type: Type of media
    
    Returns:
        Content type string
    """
    content_types = {
        "image": "image/jpeg",
        "audio": "audio/mpeg",
        "video": "video/mp4"
    }
    
    return content_types.get(media_type, "application/octet-stream")

def generate_sample_result(result_id: str):
    """
    Generate a sample result for demonstration.
    
    Args:
        result_id: ID of the sample result
    
    Returns:
        Sample result data
    """
    # Import here to avoid circular imports
    import random
    from datetime import datetime, timedelta
    
    # Extract sample number from ID if possible
    if "_" in result_id:
        sample_number = result_id.split("_")[-1]
        try:
            sample_number = int(sample_number)
        except:
            sample_number = 0
    else:
        sample_number = 0
    
    # Determine media type based on sample number
    media_types = ["image", "audio", "video"]
    media_type = media_types[sample_number % len(media_types)]
    
    # Randomize prediction but ensure it's consistent for the same result_id
    import hashlib
    hash_value = int(hashlib.md5(result_id.encode()).hexdigest(), 16)
    is_fake = hash_value % 2 == 0
    
    # Generate sample result data
    timestamp = (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat()
    
    result = {
        "result_id": result_id,
        "media_type": media_type,
        "filename": f"sample_{media_type}.{media_type[:3]}",
        "prediction": "Fake" if is_fake else "Real",
        "confidence_score": 0.65 + (hash_value % 30) / 100,
        "timestamp": timestamp,
        "analysis_duration": round(random.uniform(1.5, 5.2), 2)
    }
    
    # Add detection details
    if media_type == "image":
        result["detection_details"] = {
            "technical_explanation": "This is a sample image analysis. In a real analysis, this would contain technical details about the detection process.",
            "features": {
                "Facial Inconsistency": round(random.uniform(0.3, 0.9), 2),
                "Noise Pattern": round(random.uniform(0.3, 0.9), 2),
                "Artifact Detection": round(random.uniform(0.3, 0.9), 2),
                "Metadata Analysis": round(random.uniform(0.3, 0.9), 2)
            },
            "metadata": {
                "Camera": "Sample Camera",
                "Resolution": "1920x1080",
                "Software": "Unknown"
            }
        }
    elif media_type == "audio":
        result["detection_details"] = {
            "technical_explanation": "This is a sample audio analysis. In a real analysis, this would contain technical details about the detection process.",
            "features": {
                "Voice Pattern": round(random.uniform(0.3, 0.9), 2),
                "Background Noise": round(random.uniform(0.3, 0.9), 2),
                "Frequency Analysis": round(random.uniform(0.3, 0.9), 2),
                "Temporal Consistency": round(random.uniform(0.3, 0.9), 2)
            },
            "metadata": {
                "Format": "MP3",
                "Duration": "01:23",
                "Sample Rate": "44.1 kHz"
            }
        }
    else:  # video
        result["detection_details"] = {
            "technical_explanation": "This is a sample video analysis. In a real analysis, this would contain technical details about the detection process.",
            "features": {
                "Facial Movement": round(random.uniform(0.3, 0.9), 2),
                "Lip Sync": round(random.uniform(0.3, 0.9), 2),
                "Blinking Pattern": round(random.uniform(0.3, 0.9), 2),
                "Temporal Consistency": round(random.uniform(0.3, 0.9), 2)
            },
            "metadata": {
                "Format": "MP4",
                "Resolution": "1920x1080",
                "Duration": "00:42",
                "Frame Rate": "30 fps"
            }
        }
    
    return result
