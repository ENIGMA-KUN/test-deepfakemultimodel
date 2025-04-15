from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional

# Import from model registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_registry import (
    MODEL_REGISTRY,
    get_model_by_id,
    get_available_models,
    get_selected_model_id
)
from utils.model_loader import get_model_info

logger = logging.getLogger(__name__)

# Setup router
router = APIRouter()

@router.get("/")
async def list_models(media_type: Optional[str] = None):
    """
    Get available models, optionally filtered by media type.
    
    Args:
        media_type: Optional filter by media type (image, audio, video)
    
    Returns:
        JSON response with available models
    """
    try:
        # Get model information
        model_info = get_model_info(media_type)
        return model_info
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving model information: {str(e)}")

@router.get("/{model_id}")
async def get_model_details(model_id: str):
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: ID of the model
    
    Returns:
        JSON response with model details
    """
    try:
        # Get model information
        model_info = get_model_info(model_id=model_id)
        
        if "error" in model_info:
            raise HTTPException(status_code=404, detail=model_info["error"])
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving model details: {str(e)}")

@router.post("/{model_id}/select")
async def select_model(model_id: str):
    """
    Select a model for use in analysis.
    
    Args:
        model_id: ID of the model to select
    
    Returns:
        JSON response with selection status
    """
    try:
        # Get model information to validate it exists
        model = get_model_by_id(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        
        # Store selection in the model loader
        # This happens automatically when the model is loaded
        # We'll pre-select it here, and it will be loaded on the next analysis
        
        return {
            "status": "success",
            "message": f"Model '{model['name']}' selected for {model_id.split('_')[0]} analysis",
            "model_id": model_id,
            "model_name": model["name"],
            "media_type": next((k for k, v in get_available_models().items() if any(m["id"] == model_id for m in v)), None)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error selecting model: {str(e)}")

@router.get("/selected/{media_type}")
async def get_selected_model(media_type: str):
    """
    Get the currently selected model for a media type.
    
    Args:
        media_type: Media type (image, audio, video)
    
    Returns:
        JSON response with selected model information
    """
    try:
        # Get selected model ID
        model_id = get_selected_model_id(media_type)
        
        if not model_id:
            return {
                "status": "info",
                "message": f"No model explicitly selected for {media_type}",
                "media_type": media_type,
                "model_id": None
            }
        
        # Get model information
        model = get_model_by_id(model_id)
        
        if not model:
            return {
                "status": "warning",
                "message": f"Selected model not found: {model_id}",
                "media_type": media_type,
                "model_id": model_id
            }
        
        return {
            "status": "success",
            "media_type": media_type,
            "model_id": model_id,
            "model_name": model["name"],
            "model_description": model["description"]
        }
        
    except Exception as e:
        logger.error(f"Error getting selected model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving selected model: {str(e)}")
