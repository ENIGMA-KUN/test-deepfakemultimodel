@echo off
echo Setting up DeepFake Detection Platform...

:: Check if Python is installed
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch with CUDA for RTX 4080
echo Installing PyTorch with CUDA for RTX 4080...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

:: Install backend dependencies
echo Installing backend dependencies...
pip install fastapi==0.95.0 uvicorn==0.21.1 pydantic==1.10.7 python-multipart==0.0.6 python-dotenv==1.0.0
pip install sqlalchemy==2.0.9 psycopg2-binary==2.9.6 aiosqlite==0.18.0
pip install celery==5.2.7 redis==4.5.4
pip install transformers==4.28.1 librosa==0.10.0 numpy==1.24.2
pip install scikit-image==0.20.0 scikit-learn==1.2.2 opencv-python==4.7.0.72
pip install matplotlib==3.7.1 pillow==9.5.0 tqdm==4.65.0 requests==2.28.2 pydub==0.25.1

:: Verify PyTorch can see the GPU
echo Checking GPU availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo Setup complete! Activate the environment with: venv\Scripts\activate