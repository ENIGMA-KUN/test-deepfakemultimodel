FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install torch and torchvision
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1

# Copy backend code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results visualizations

# Set environment variables
ENV PYTHONPATH=/app
ENV UPLOAD_DIR=/app/uploads
ENV RESULTS_DIR=/app/results
ENV VISUALIZATIONS_DIR=/app/visualizations

# Expose the port for the FastAPI application
EXPOSE 8000

# Start the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]