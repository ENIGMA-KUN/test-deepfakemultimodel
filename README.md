# DeepFake Detection Platform

A comprehensive platform for detecting manipulated content across images, audio, and video using state-of-the-art deep learning models.

## Features

- **Multi-format Detection**: Analyze images, audio, and video for potential manipulation
- **Advanced AI Models**: Utilizes specialized models for each media type
  - Images: XceptionNet, EfficientNet, MesoNet
  - Audio: Wav2Vec 2.0, RawNet2
  - Video: 3D-CNN, Two-Stream Networks
- **Detailed Visualizations**: Heat maps, temporal analysis, and frequency analysis
- **Fast API Backend**: Scalable architecture with background processing
- **Modern React Frontend**: Intuitive interface with real-time progress tracking

## System Requirements

- Docker and Docker Compose
- 8GB+ RAM
- NVIDIA GPU with CUDA support (recommended for faster inference)

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-detection.git
   cd deepfake-detection
   ```

2. Copy the environment file:
   ```bash
   cp .env.example .env
   ```

3. Create necessary directories:
   ```bash
   mkdir -p uploads visualizations
   mkdir -p backend/app/models/weights
   ```

4. Download model weights:
   ```bash
   python scripts/download_weights.py
   ```

5. Start the application:
   ```bash
   docker-compose up -d
   ```

### Usage

1. Access the frontend at: http://localhost:3000
2. Upload images, audio, or video for analysis
3. View detailed results with visualizations
4. API documentation: http://localhost:8000/docs

## Project Structure

```
deepfake-detection/
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   ├── core/             # Core configurations
│   │   ├── db/               # Database models and sessions
│   │   ├── models/           # ML models for detection
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   ├── tasks/            # Celery tasks
│   │   └── utils/            # Utility functions
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── contexts/         # React contexts
│   │   ├── hooks/            # Custom hooks
│   │   ├── pages/            # Page components
│   │   ├── services/         # API services
│   │   ├── styles/           # CSS styles
│   │   └── types/            # TypeScript types
│   ├── Dockerfile
│   └── package.json
├── scripts/                  # Utility scripts
├── uploads/                  # Uploaded files directory
├── visualizations/           # Generated visualizations
├── .env.example
├── docker-compose.yml
└── README.md
```

## API Documentation

The API documentation is available at http://localhost:8000/docs when the application is running.

### Key Endpoints

- `/api/v1/upload/upload`: Upload files for detection
- `/api/v1/detection/status/{task_id}`: Check detection task status
- `/api/v1/results/detail/{result_id}`: Get detailed detection results

## Development

### Backend Development

1. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Development

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Run the React development server:
   ```bash
   npm start
   ```

## Model Information

### Image Models

- **XceptionNet**: State-of-the-art performance in deepfake detection
- **EfficientNet-B4**: Excellent efficiency-to-accuracy ratio
- **MesoNet**: Lightweight model for fast inference

### Audio Models

- **Wav2Vec 2.0**: Advanced model for audio deepfake detection
- **RawNet2**: Raw waveform analysis for voice manipulation detection

### Video Models

- **3D-CNN**: Spatio-temporal analysis for video deepfakes
- **Two-Stream Network**: Dual-path network analyzing spatial and temporal features

## License

MIT License