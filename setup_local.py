#!/usr/bin/env python
"""
Local setup script for DeepFake Detection Platform.
"""

import os
import sys
import subprocess
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logger.error(f"❌ Python {required_version[0]}.{required_version[1]} or higher is required.")
        logger.error(f"   Current version: {current_version[0]}.{current_version[1]}")
        sys.exit(1)
    
    logger.info(f"✅ Python version: {current_version[0]}.{current_version[1]}")

def create_virtual_environment():
    """Create a virtual environment for the project."""
    if os.path.exists("venv"):
        logger.info("✅ Virtual environment already exists")
        return
    
    logger.info("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    logger.info("✅ Virtual environment created")

def install_dependencies():
    """Install required Python dependencies."""
    logger.info("Installing dependencies...")
    
    # Determine pip command based on platform
    if platform.system() == "Windows":
        pip_cmd = os.path.join("venv", "Scripts", "pip")
    else:
        pip_cmd = os.path.join("venv", "bin", "pip")
    
    # Install base requirements
    subprocess.run([pip_cmd, "install", "-U", "pip"], check=True)
    subprocess.run([pip_cmd, "install", "-r", "backend/requirements.txt"], check=True)
    
    logger.info("✅ Dependencies installed")

def create_env_file():
    """Create a .env file for local development."""
    if os.path.exists(".env"):
        logger.info("✅ .env file already exists")
        return
    
    if os.path.exists(".env.example"):
        with open(".env.example", "r") as example_file:
            content = example_file.read()
        
        # Modify for local setup
        content = content.replace("ENVIRONMENT=development", "ENVIRONMENT=local")
        content = content.replace("DATABASE_URL=postgresql", "DATABASE_TYPE=sqlite")
        content = content.replace("CELERY_TASK_ALWAYS_EAGER=false", "CELERY_TASK_ALWAYS_EAGER=true")
        
        with open(".env", "w") as env_file:
            env_file.write(content)
        
        logger.info("✅ Created .env file for local development")
    else:
        logger.error("❌ .env.example file not found")
        sys.exit(1)

def download_model_weights():
    """Download model weights."""
    logger.info("Downloading model weights...")
    
    # Determine python command based on platform
    if platform.system() == "Windows":
        python_cmd = os.path.join("venv", "Scripts", "python")
    else:
        python_cmd = os.path.join("venv", "bin", "python")
    
    subprocess.run([python_cmd, "scripts/download_weights.py"], check=True)
    logger.info("✅ Model weights downloaded")

def initialize_database():
    """Initialize the SQLite database."""
    logger.info("Initializing database...")
    
    # Determine python command based on platform
    if platform.system() == "Windows":
        python_cmd = os.path.join("venv", "Scripts", "python")
    else:
        python_cmd = os.path.join("venv", "bin", "python")
    
    # Run a simple script to initialize the database
    init_script = """
from backend.app.db.session import init_db
init_db()
print("Database initialized")
"""
    
    with open("init_db.py", "w") as f:
        f.write(init_script)
    
    subprocess.run([python_cmd, "init_db.py"], check=True)
    os.remove("init_db.py")
    
    logger.info("✅ Database initialized")

def main():
    """Run the local setup process."""
    logger.info("=== DeepFake Detection Platform: Local Setup ===")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file
    create_env_file()
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("backend/app/models/weights", exist_ok=True)
    
    # Download model weights
    download_model_weights()
    
    # Initialize database
    initialize_database()
    
    logger.info("=== Local setup completed successfully! ===")
    logger.info("")
    logger.info("To start the backend:")
    if platform.system() == "Windows":
        logger.info("  venv\\Scripts\\python -m uvicorn backend.app.main:app --reload")
    else:
        logger.info("  venv/bin/python -m uvicorn backend.app.main:app --reload")
    
    logger.info("")
    logger.info("To start the frontend:")
    logger.info("  cd frontend")
    logger.info("  npm install")
    logger.info("  npm start")

if __name__ == "__main__":
    main()