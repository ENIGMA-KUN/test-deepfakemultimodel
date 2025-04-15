from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import sys
import time
from datetime import datetime

# Import routers
from routers import upload, analysis, results, models

# Import core modules
from core.config import settings
from core.database import init_db, close_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfakes in images, audio, and video",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
app.include_router(results.router, prefix="/results", tags=["results"])
app.include_router(models.router, prefix="/models", tags=["models"])

# Create required directories
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application")
    
    # Create required directories if they don't exist
    for dir_path in [
        os.path.join(os.path.dirname(__file__), "media"),
        os.path.join(os.path.dirname(__file__), "results"),
        os.path.join(os.path.dirname(__file__), "data")
    ]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
    
    # Initialize database
    await init_db()

# Close database connections on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application")
    await close_db()

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Welcome to the Deepfake Detection API",
        "version": "1.0.0",
        "status": "operational",
        "server_time": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
