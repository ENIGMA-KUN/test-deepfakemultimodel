#!/usr/bin/env python
"""
Script to run the DeepFake Detection Platform locally.
"""

import os
import sys
import platform
import subprocess
import threading
import time
import signal

# Handle Ctrl+C gracefully
running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_python_command():
    """Get the Python command based on platform."""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "python")
    else:
        return os.path.join("venv", "bin", "python")

def run_backend():
    """Run the backend server."""
    python_cmd = get_python_command()
    
    # Start the backend server
    backend_cmd = [
        python_cmd, 
        "-m", "uvicorn", 
        "backend.app.main:app", 
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    print("Starting backend server...")
    subprocess.run(backend_cmd)

def run_frontend():
    """Run the frontend development server."""
    # Change to frontend directory
    os.chdir("frontend")
    
    # Check if node_modules exists
    if not os.path.exists("node_modules"):
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"])
    
    # Start the frontend server
    print("Starting frontend server...")
    subprocess.run(["npm", "start"])

def main():
    """Run both backend and frontend servers."""
    print("=== Starting DeepFake Detection Platform ===")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Give backend a moment to start
    time.sleep(2)
    
    # Start frontend
    run_frontend()

if __name__ == "__main__":
    main()