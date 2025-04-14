import subprocess
import sys

# List of packages to install
packages = [
    "fastapi>=0.103.1",
    "uvicorn>=0.23.2",
    "pydantic>=2.4.2",
    "pydantic-settings>=2.0.0",
    "python-multipart==0.0.6",
    "python-dotenv==1.0.0",
    "sqlalchemy==2.0.9",
    "aiosqlite==0.18.0",
    "celery==5.2.7",
    "redis==4.5.4",
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "transformers==4.28.1",
    "librosa==0.10.0",
    "numpy==1.24.2",
    "scikit-image==0.20.0",
    "scikit-learn==1.2.2",
    "opencv-python==4.7.0.72",
    "matplotlib==3.7.1",
    "pillow==9.5.0",
    "tqdm==4.65.0",
    "requests==2.28.2",
    "pydub==0.25.1"
]

print("Installing dependencies...")
for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("All dependencies installed successfully!") 