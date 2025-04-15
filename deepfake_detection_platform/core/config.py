import os
from pydantic import BaseSettings
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings configuration"""
    
    # Application config
    APP_NAME: str = "Deepfake Detection Platform"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"
    
    # Media file limits (in MB)
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "10"))
    MAX_AUDIO_SIZE: int = int(os.getenv("MAX_AUDIO_SIZE", "25"))
    MAX_VIDEO_SIZE: int = int(os.getenv("MAX_VIDEO_SIZE", "100"))
    
    # Database settings
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "data/deepfake_detection.db")
    
    # Media storage paths
    MEDIA_DIR: str = os.getenv("MEDIA_DIR", "media")
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")
    
    # Model settings
    IMAGE_MODEL_PATH: Optional[str] = os.getenv("IMAGE_MODEL_PATH")
    AUDIO_MODEL_PATH: Optional[str] = os.getenv("AUDIO_MODEL_PATH")
    VIDEO_MODEL_PATH: Optional[str] = os.getenv("VIDEO_MODEL_PATH")
    
    # API rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        *os.getenv("ADDITIONAL_CORS_ORIGINS", "").split(",") if os.getenv("ADDITIONAL_CORS_ORIGINS") else []
    ]
    
    # Security settings
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: Optional[str] = os.getenv("API_KEY")
    
    # Streamlit settings
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_ADDRESS: str = os.getenv("STREAMLIT_ADDRESS", "localhost")
    
    # Log settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # File validation settings
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
    ALLOWED_AUDIO_TYPES: List[str] = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/flac"]
    ALLOWED_VIDEO_TYPES: List[str] = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    
    # Compute settings
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Paths relative to the application root
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(APP_ROOT, settings.DATABASE_PATH)
MEDIA_DIR = os.path.join(APP_ROOT, settings.MEDIA_DIR)
RESULTS_DIR = os.path.join(APP_ROOT, settings.RESULTS_DIR)

# Ensure directories exist
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Update settings with absolute paths
settings.DATABASE_PATH = DATABASE_PATH
settings.MEDIA_DIR = MEDIA_DIR
settings.RESULTS_DIR = RESULTS_DIR
