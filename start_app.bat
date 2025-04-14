@echo off
echo =========================================================
echo      DeepFake Detection Platform - Starting Application
echo =========================================================

:: Activate the virtual environment
call venv\Scripts\activate

:: Start the backend in a separate window
start cmd /k "title DeepFake Backend & python -m uvicorn backend.app.main:app --reload"

:: Wait a few seconds for the backend to start
echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

:: Start the frontend in a separate window
start cmd /k "title DeepFake Frontend & cd frontend && npm start"

echo =========================================================
echo            Application started successfully!
echo =========================================================
echo.
echo Backend running at: http://localhost:8000
echo Frontend running at: http://localhost:3000
echo.
echo Close the command windows to stop the application.