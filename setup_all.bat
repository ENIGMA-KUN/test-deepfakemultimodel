@echo off
echo =========================================================
echo        DeepFake Detection Platform - Complete Setup
echo =========================================================

:: 1. Set up Python environment and dependencies
echo Setting up Python environment and dependencies...
call setup.bat

:: 2. Fix RawNet2 model implementation
echo Fixing RawNet2 model implementation...
call venv\Scripts\python fix_rawnet2.py

:: 3. Standardize model weight files
echo Standardizing model weight files...
call venv\Scripts\python standardize_weights.py

:: 4. Set up frontend dependencies
echo Setting up frontend dependencies...
call setup_frontend.bat

:: 5. Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
)

:: 6. Create necessary directories
echo Creating necessary directories...
if not exist uploads mkdir uploads
if not exist results mkdir results
if not exist visualizations mkdir visualizations

echo =========================================================
echo              Setup completed successfully!
echo =========================================================
echo.
echo To start the application:
echo 1. Activate the environment: venv\Scripts\activate
echo 2. Start the backend: python -m uvicorn backend.app.main:app --reload
echo 3. Start the frontend: cd frontend ^& npm start
echo.
echo Or use the start_app.bat script to run both simultaneously.