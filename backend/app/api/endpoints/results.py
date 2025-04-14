from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from typing import Optional

from app.db.session import get_db
from app.db.models import DetectionResult
from app.schemas.results import ResultQuery, ResultStatus, DetailedResult


router = APIRouter()


@router.post("/query", response_model=ResultStatus)
async def query_result(
    query: ResultQuery,
    db: Session = Depends(get_db)
):
    """Query the status or result of a detection task."""
    if query.task_id:
        # Check task status in Celery
        from app.tasks.celery_app import celery_app
        task = celery_app.AsyncResult(query.task_id)
        
        if task.state == 'PENDING':
            return ResultStatus(status="pending", progress=0, message="Task is pending")
        elif task.state == 'PROGRESS':
            return ResultStatus(
                status="processing",
                progress=task.info.get('progress', 0),
                message=task.info.get('message', 'Processing')
            )
        elif task.state == 'SUCCESS':
            result_id = task.result
            return ResultStatus(status="complete", progress=100, result_id=result_id)
        else:
            return ResultStatus(status="failed", progress=0, message=str(task.info))
    
    elif query.result_id:
        # Check if result exists in database
        result = db.query(DetectionResult).filter(DetectionResult.id == query.result_id).first()
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return ResultStatus(status="complete", progress=100, result_id=result.id)
    
    else:
        raise HTTPException(status_code=400, detail="Either task_id or result_id must be provided")


@router.get("/detail/{result_id}", response_model=DetailedResult)
async def get_detailed_result(
    result_id: str = Path(..., description="Result ID to retrieve"),
    db: Session = Depends(get_db)
):
    """Get detailed result for a completed detection."""
    result = db.query(DetectionResult).filter(DetectionResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Transform database model to response schema
    visualizations = {}
    
    # Check if we have visualization paths in detection details
    visualization_paths = result.detection_details.get("visualization_paths", {})
    
    # Standard visualizations
    if result.heatmap_path:
        visualizations["heatmap"] = {
            "url": f"/visualizations/{result.heatmap_path}",
            "width": 512,
            "height": 512,
            "regions": result.detection_details.get("regions", [])
        }
    
    # Temporal analysis
    if result.temporal_analysis_path:
        temporal_data = result.detection_details.get("temporal_analysis", {})
        visualizations["temporal"] = {
            "url": f"/visualizations/{result.temporal_analysis_path}",
            "timestamps": temporal_data.get("timestamps", []),
            "values": temporal_data.get("scores", []),  # Changed to "scores" as that's what we use in the data structure
            "threshold": temporal_data.get("threshold", 0.5)
        }
    elif "temporal_analysis" in visualization_paths:
        temporal_data = result.detection_details.get("temporal_analysis", {})
        visualizations["temporal"] = {
            "url": f"/visualizations/{visualization_paths['temporal_analysis']}",
            "timestamps": temporal_data.get("timestamps", []),
            "values": temporal_data.get("scores", []),
            "threshold": temporal_data.get("threshold", 0.5)
        }
    
    # Frequency analysis
    if "frequency_analysis" in result.detection_details:
        visualizations["frequency"] = result.detection_details["frequency_analysis"]
    
    # Confidence gauge visualization
    if "confidence_gauge" in visualization_paths:
        if "combined_confidence_score" in result.detection_details:
            confidence = result.detection_details["combined_confidence_score"]
        else:
            confidence = result.confidence_score
            
        visualizations["confidence_gauge"] = {
            "url": f"/visualizations/{visualization_paths['confidence_gauge']}",
            "score": confidence
        }
    
    # Audio-specific visualizations for audio media type
    if result.media_type == "audio" and visualization_paths:
        # Spectral discontinuity visualization
        if "spectral_discontinuity" in visualization_paths:
            spectral_data = result.detection_details.get("comprehensive_analysis", {}).get("spectral_analysis", {})
            if spectral_data:
                visualizations["spectral_discontinuity"] = {
                    "url": f"/visualizations/{visualization_paths['spectral_discontinuity']}",
                    "splice_times": spectral_data.get("splice_times", []),
                    "threshold": spectral_data.get("threshold", 0.5)
                }
        
        # Voice consistency visualization
        if "voice_consistency" in visualization_paths:
            vc_data = result.detection_details.get("comprehensive_analysis", {}).get("voice_consistency", {})
            if vc_data and "segment_diffs" in vc_data:
                visualizations["voice_consistency"] = {
                    "url": f"/visualizations/{visualization_paths['voice_consistency']}",
                    "segment_diffs": vc_data.get("segment_diffs", []),
                    "mean_diff": vc_data.get("mean_segment_diff", 0)
                }
        
        # Silence segments visualization
        if "silence_analysis" in visualization_paths:
            silence_data = result.detection_details.get("comprehensive_analysis", {}).get("silence_analysis", {})
            if silence_data and "silence_segments" in silence_data:
                visualizations["silence_analysis"] = {
                    "url": f"/visualizations/{visualization_paths['silence_analysis']}",
                    "segments": silence_data.get("silence_segments", []),
                    "total_duration": silence_data.get("total_silence_duration", 0)
                }
    
    return DetailedResult(
        id=result.id,
        is_fake=result.is_fake,
        confidence_score=result.confidence_score,
        media_type=result.media_type,
        detection_details=result.detection_details,
        models_used=result.models_used,
        visualizations=visualizations if visualizations else None
    )
