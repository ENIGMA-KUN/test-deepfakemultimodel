Write-Host "=========================================================
 DeepFake Detection Platform - Starting Services
=========================================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info > $null
} catch {
    Write-Host "Error: Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# Create properly encoded __init__.py files
Write-Host "Creating properly encoded __init__.py files..." -ForegroundColor Yellow
""""DeepFake Detection Platform Main Module""" | Out-File -FilePath "backend\app\__init__.py" -Encoding utf8
""""API Module""" | Out-File -FilePath "backend\app\api\__init__.py" -Encoding utf8
""""API Endpoints Module""" | Out-File -FilePath "backend\app\api\endpoints\__init__.py" -Encoding utf8
""""Core Module""" | Out-File -FilePath "backend\app\core\__init__.py" -Encoding utf8
""""Database Module""" | Out-File -FilePath "backend\app\db\__init__.py" -Encoding utf8
""""ML Models Module""" | Out-File -FilePath "backend\app\models\__init__.py" -Encoding utf8
""""Schemas Module""" | Out-File -FilePath "backend\app\schemas\__init__.py" -Encoding utf8
""""Services Module""" | Out-File -FilePath "backend\app\services\__init__.py" -Encoding utf8
""""Async Tasks Module""" | Out-File -FilePath "backend\app\tasks\__init__.py" -Encoding utf8
""""Utility Functions Module""" | Out-File -FilePath "backend\app\utils\__init__.py" -Encoding utf8

# Make sure the required directories exist
Write-Host "Ensuring required directories exist..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "uploads" -Force | Out-Null
New-Item -ItemType Directory -Path "results" -Force | Out-Null
New-Item -ItemType Directory -Path "visualizations" -Force | Out-Null

# Stop any existing containers
Write-Host "Stopping any existing containers..." -ForegroundColor Yellow
docker-compose down

# Start the services
Write-Host "Starting Docker services..." -ForegroundColor Green
docker-compose up -d

# Check if all containers are running
Start-Sleep -Seconds 5
$containers = docker ps --format "{{.Names}}" | Select-String "lightmultidetect"
if ($containers.Count -lt 5) {
    Write-Host "Warning: Not all containers appear to be running. Check docker logs for errors." -ForegroundColor Yellow
}

Write-Host "=========================================================
 DeepFake Detection Platform is now running!
 Frontend: http://localhost:3000
 API: http://localhost:8000
 API Docs: http://localhost:8000/docs
=========================================================" -ForegroundColor Cyan

Write-Host "Press Enter to view logs, or Ctrl+C to exit." -ForegroundColor Gray
$null = Read-Host

# Show container logs
docker-compose logs -f 