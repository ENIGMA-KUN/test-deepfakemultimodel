@echo off
echo Setting up frontend dependencies...

:: Check if Node.js is installed
node --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js not found. Please install Node.js 16 or higher.
    exit /b 1
)

:: Navigate to frontend directory
cd frontend

:: Install dependencies
echo Installing npm packages...
npm install

echo Frontend setup complete!