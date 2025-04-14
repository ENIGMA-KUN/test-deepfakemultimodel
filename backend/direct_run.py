from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import uvicorn
import logging
import uuid
import time
import shutil
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Initialize app
app = FastAPI(
    title="DeepFake Detection Platform",
    description="API for deepfake detection across images, audio, and video",
    debug=True,
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for visualizations
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "environment": "development"
    }

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the DeepFake Detection API",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.post("/api/v1/upload/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    use_image: bool = Form(True), 
    use_audio: bool = Form(False),
    use_video: bool = Form(False),
    confidence_threshold: float = Form(0.5)
):
    """Upload files for deepfake detection."""
    if not any([use_image, use_audio, use_video]):
        raise HTTPException(
            status_code=400, 
            detail="At least one modality must be selected"
        )
    
    task_id = str(uuid.uuid4())
    task_dir = os.path.join("uploads", task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    file_paths = []
    for file in files:
        file_path = os.path.join(task_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
    
    logger.info(f"Uploaded {len(files)} files for task {task_id}")
    
    # In a real implementation, we would pass this to a background task
    # Here we're just returning a simple mock response
    return {
        "task_id": task_id,
        "status": "pending",
        "message": f"Processing started with {len(files)} files",
        "timestamp": datetime.now().isoformat(),
        "modalities": {
            "image": use_image,
            "audio": use_audio,
            "video": use_video
        },
        "confidence_threshold": confidence_threshold
    }

@app.get("/api/v1/detection/status/{task_id}")
async def check_status(task_id: str):
    """Check the status of a detection task."""
    task_dir = os.path.join("uploads", task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Task with ID {task_id} not found"
        )
    
    # In a real implementation, we would check the actual status
    # Here we're just returning a mock response
    return {
        "task_id": task_id,
        "status": "completed",
        "message": "Processing completed successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/results/result/{task_id}")
async def get_results(task_id: str):
    """Get the results of a detection task."""
    task_dir = os.path.join("uploads", task_id)
    if not os.path.exists(task_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Task with ID {task_id} not found"
        )
    
    # Generate mock results for demonstration
    files = os.listdir(task_dir)
    results = []
    
    for file_name in files:
        result = {
            "file_name": file_name,
            "prediction": "fake" if file_name.startswith("f") else "real",
            "confidence": 0.85,
            "modality": "image",
            "processing_time": 1.5,
        }
        results.append(result)
    
    return {
        "task_id": task_id,
        "status": "completed",
        "message": "Results available",
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("Starting DeepFake Detection API")
    uvicorn.run(app, host="0.0.0.0", port=8000) 