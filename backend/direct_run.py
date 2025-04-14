from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import uvicorn
import logging
import uuid
import time
import json
import shutil
from datetime import datetime, timedelta
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

# Set up CORS middleware with wildcard to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins with wildcard
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Content-Length", "Content-Type"],
    max_age=600  # Cache preflight requests for 10 minutes
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

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    detection_params: Optional[str] = Form(None)
):
    """Upload files for deepfake detection."""
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Parse detection parameters
    params = {}
    media_type = "image"  # Default
    detailed_analysis = False
    confidence_threshold = 0.5
    
    if detection_params:
        try:
            params = json.loads(detection_params)
            media_type = params.get("media_type", "image")
            detailed_analysis = params.get("detailed_analysis", False)
            confidence_threshold = params.get("confidence_threshold", 0.5)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid detection parameters: {str(e)}")
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    file_path = os.path.join("uploads", file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"Uploaded file {file.filename} for task {task_id}")
    logger.info(f"Using model for {media_type} with detailed_analysis={detailed_analysis}")
    
    # Load the appropriate model based on media type
    # This is where your actual model loading and inference would happen
    model_name = ""  # Will be populated below
    
    if media_type == "image":
        # Simulate loading the image model
        # In a real implementation, you would load your actual pretrained model here
        model_name = "FaceForensics++" if detailed_analysis else "EfficientNet-B4"
        # Apply model to the uploaded image
        logger.info(f"Running {model_name} model on image {file.filename}")
        
    elif media_type == "audio":
        # Simulate loading the audio model
        model_name = "RawNet2" if detailed_analysis else "LightCNN"
        logger.info(f"Running {model_name} model on audio {file.filename}")
        
    elif media_type == "video":
        # Simulate loading the video model
        model_name = "3D-CNN" if detailed_analysis else "TSN"
        logger.info(f"Running {model_name} model on video {file.filename}")
    
    # Start a background task to process the file (simulated)
    # In a real implementation, this would be a Celery task or similar
    
    # Return task information
    return {
        "task_id": task_id,
        "status": "processing",
        "media_type": media_type,
        "estimated_time": 5  # seconds
    }

@app.get("/status/{task_id}")
async def check_status(task_id: str):
    """Check the status of a detection task."""
    # In a real implementation, you would check a task queue or database
    # For demonstration, we'll simulate task completion
    
    # Simulate extracting media type from the task info
    # In a real implementation, this would come from your database
    import random
    
    # Look up the uploaded file for this task (simulated)
    # Check the filename to determine media type
    uploaded_files = os.listdir("uploads")
    media_type = "image"  # Default to image
    
    # Try to find a file associated with this task
    task_file = None
    for filename in uploaded_files:
        if task_id in filename or (len(uploaded_files) > 0 and random.random() < 0.8):
            task_file = filename
            break
    
    # If we found a file, determine media type from extension
    if task_file:
        if task_file.lower().endswith((".mp4", ".avi", ".mov", ".wmv")):
            media_type = "video"
            logger.info(f"Detected video file: {task_file} for task {task_id}")
        elif task_file.lower().endswith((".mp3", ".wav", ".ogg", ".flac")):
            media_type = "audio"
            logger.info(f"Detected audio file: {task_file} for task {task_id}")
        else:
            logger.info(f"Detected image file: {task_file} for task {task_id}")
    else:
        # If no file found, randomly assign a media type with image being most common
        rand_val = random.random()
        if rand_val > 0.7:
            media_type = "video"
        elif rand_val > 0.5:
            media_type = "audio"
    
    # Generate a result ID based on the task ID and media type
    result_id = f"result-{task_id[:8]}_{media_type}"
    logger.info(f"Generated result ID: {result_id} for task {task_id}")
    
    # Return completed status with result ID that includes media type
    return {
        "status": "success",
        "progress": 100,
        "result_id": result_id,
        "media_type": media_type,
        "message": "Processing completed successfully"
    }

@app.get("/results/{result_id}")
async def get_results(result_id: str):
    """Get the results of a detection task."""
    # In a real implementation, you would retrieve results from a database
    # Here we'll create a realistic result that would come from your models
    
    # Import for random generation
    import random
    
    # Determine media type from result_id or task content
    # In a real implementation, this would come from your database
    media_type = "image"  # Default is image
    if "_video" in result_id.lower():
        media_type = "video"
    elif "_audio" in result_id.lower():
        media_type = "audio"
    
    # Log the request
    logger.info(f"Retrieving results for {result_id} with detected media type: {media_type}")
    
    # Randomize prediction but adjust based on media type
    # Different media types have different fake/real distributions
    fake_probability = 0.3  # Default 30% chance of being fake
    if media_type == "video":
        fake_probability = 0.5  # Videos more likely to be fake
    elif media_type == "audio":
        fake_probability = 0.4  # Audio moderate chance
        
    is_fake = random.random() < fake_probability
    
    # Create a more realistic confidence score range based on the prediction
    if is_fake:
        # For fake predictions, confidence between 80-99%
        confidence = random.uniform(0.8, 0.99)
    else:
        # For real predictions, confidence between 75-97%
        confidence = random.uniform(0.75, 0.97)
    
    # Technical details vary by media type
    technical_details = {}
    frame_analysis = []
    key_indicators = []
    visual_explanation = None
    
    if media_type == "image":
        # Image-specific model details
        model_used = "DenseNet-121" if confidence > 0.9 else "EfficientNet-B4"
        
        # Generate a visual explanation if fake
        if is_fake:
            visual_explanation = "/visualizations/heatmap.jpg"
            
        # Technical details for images
        technical_details = {
            "frame_count": 1,  # Images only have one frame
            "processing_time": round(random.uniform(0.8, 3.2), 2),
            "detection_method": "CNN + Attention" if confidence > 0.9 else "EfficientNet",
            "resolution": "1920x1080",
            "color_analysis": "natural" if not is_fake else "manipulated",
            "texture_coherence": "coherent" if not is_fake else "distorted"
        }
        
        # Image-specific indicators of manipulation if fake
        if is_fake:
            possible_indicators = [
                "Facial texture inconsistency",
                "Unnatural eye blinking patterns",
                "Inconsistent lighting effects",
                "Boundary artifacts",
                "Unusual color distribution"
            ]
            
            # Choose 1-3 indicators randomly
            num_indicators = random.randint(1, 3)
            key_indicators = random.sample(possible_indicators, num_indicators)
            
            # Add frame analysis data for the single image frame
            frame_analysis = [{
                "frame": 0,
                "confidence": round(confidence * 100, 1),
                "key_points": "facial_features"
            }]
                
    elif media_type == "video":
        # Video-specific model details
        model_used = "SlowFast" if confidence > 0.9 else "I3D"
        
        # Generate a visual explanation if fake
        if is_fake:
            visual_explanation = "/visualizations/video_heatmap.jpg"
            
        # Video-specific technical details
        frame_count = random.randint(90, 450)
        fps = 30
        duration = round(frame_count / fps, 1)
        
        technical_details = {
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
            "processing_time": round(random.uniform(4.2, 12.5), 2),
            "detection_method": "3D-CNN + Attention" if confidence > 0.9 else "I3D",
            "resolution": "1280x720",
            "temporal_coherence": "consistent" if not is_fake else "inconsistent"
        }
        
        # Video-specific indicators if fake
        if is_fake:
            possible_indicators = [
                "Temporal inconsistencies",
                "Audio-visual desynchronization",
                "Facial expression mismatches",
                "Unnatural head movements",
                "Blinking irregularities",
                "Edge artifacts around face",
                "Audio artifacts"
            ]
            
            # Choose 2-4 indicators randomly
            num_indicators = random.randint(2, 4)
            key_indicators = random.sample(possible_indicators, num_indicators)
            
            # Add frame analysis data for key frames
            for i in range(3):
                frame_num = random.randint(10, frame_count - 20)
                frame_conf = round(confidence * 100 * random.uniform(0.9, 1.1), 1)
                frame_analysis.append({
                    "frame": frame_num,
                    "confidence": min(frame_conf, 99.9),  # Cap at 99.9%
                    "key_points": "lip_sync" if i == 0 else "facial_movement"
                })
        
    elif media_type == "audio":
        # Audio-specific model details
        model_used = "WavLM" if confidence > 0.9 else "RawNet2"
        
        # Generate a visual explanation if fake
        if is_fake:
            visual_explanation = "/visualizations/audio_spectrogram.jpg"
            
        # Audio-specific technical details
        duration = round(random.uniform(8.5, 45.0), 1)
        
        technical_details = {
            "sample_rate": "44.1 kHz",
            "duration": duration,
            "processing_time": round(random.uniform(2.1, 8.2), 2),
            "detection_method": "MFCC + RNN" if confidence > 0.9 else "Spectrogram CNN",
            "frequency_range": "20Hz-20kHz",
            "spectral_coherence": "natural" if not is_fake else "artificial"
        }
        
        # Audio-specific indicators if fake
        if is_fake:
            possible_indicators = [
                "Voice pattern irregularities",
                "Unnatural pauses",
                "Missing ambient noise",
                "Frequency distribution anomalies",
                "Robotic voice qualities",
                "Abnormal transitions"
            ]
            
            # Choose 1-3 indicators randomly
            num_indicators = random.randint(1, 3)
            key_indicators = random.sample(possible_indicators, num_indicators)
    
    # Create the file extension based on media type
    file_extension = "jpg" if media_type == "image" else "mp4" if media_type == "video" else "mp3"
    
    # Return a comprehensive result object that varies by media type
    return {
        "id": result_id,
        "status": "completed",
        "created_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
        "updated_at": datetime.now().isoformat(),
        "file_name": f"sample_{result_id[-6:]}.{file_extension}",
        "file_hash": f"hash_{result_id[-8:]}",
        "media_type": media_type,
        "result": {
            "prediction": "fake" if is_fake else "real",
            "confidence": round(confidence, 2),
            "analyzed_at": datetime.now().isoformat(),
            "processing_time": technical_details["processing_time"],
            "model_used": model_used,
            "technical_details": technical_details,
            "key_indicators": key_indicators,
            "frame_analysis": frame_analysis,
            "visual_explanation": visual_explanation
        }
    }

if __name__ == "__main__":
    logger.info("Starting DeepFake Detection API")
    uvicorn.run(app, host="0.0.0.0", port=8001) 