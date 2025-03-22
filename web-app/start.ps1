# Start the PassportPAL Web Application using Docker Compose
Write-Host "Starting PassportPAL Web Application..." -ForegroundColor Green

# Check if Docker is running
if (-not (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Desktop is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Download models if they don't exist
Write-Host "Checking for ML models..." -ForegroundColor Cyan
$modelScript = Join-Path $PSScriptRoot "backend\download_models.ps1"
if (Test-Path $modelScript) {
    & $modelScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to download models. Please check your internet connection and try again." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Model download script not found at: $modelScript" -ForegroundColor Red
    exit 1
}

# Check if required ports are available
$port80InUse = Get-NetTCPConnection -LocalPort 80 -ErrorAction SilentlyContinue
$port5000InUse = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue

if ($port80InUse) {
    Write-Host "Warning: Port 80 is already in use. The frontend service might fail to start." -ForegroundColor Yellow
}

if ($port5000InUse) {
    Write-Host "Warning: Port 5000 is already in use. The backend service might fail to start." -ForegroundColor Yellow
}

# Stop any existing containers
Write-Host "Stopping any existing containers..." -ForegroundColor Cyan
docker compose down

# Remove existing images to ensure fresh build
Write-Host "Removing existing images..." -ForegroundColor Cyan
try {
    docker image rm web-app-frontend:latest web-app-backend:latest -f
} catch {
    Write-Host "No existing images to remove. Continuing..." -ForegroundColor Yellow
}

# Build and start containers
Write-Host "Building and starting containers..." -ForegroundColor Cyan
docker compose up --build -d

# Wait for services to be ready
Write-Host "Waiting for services to start..." -ForegroundColor Cyan
$maxAttempts = 60
$attempts = 0
$backendReady = $false
$frontendReady = $false

while (($attempts -lt $maxAttempts) -and (-not ($backendReady -and $frontendReady))) {
    $attempts++
    
    # Check backend health
    try {
        $backendResponse = Invoke-WebRequest -Uri "http://localhost:5000/api/status" -TimeoutSec 2
        if ($backendResponse.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host "Backend is ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "Waiting for backend..." -ForegroundColor Yellow
    }

    # Check frontend health
    try {
        $frontendResponse = Invoke-WebRequest -Uri "http://localhost" -TimeoutSec 2
        if ($frontendResponse.StatusCode -eq 200) {
            $frontendReady = $true
            Write-Host "Frontend is ready!" -ForegroundColor Green
        }
    } catch {
        # Not ready yet
    }

    if (-not ($backendReady -and $frontendReady)) {
        Start-Sleep -Seconds 2
    }
}

if (-not $backendReady) {
    Write-Host "Error: Backend failed to start properly." -ForegroundColor Red
    docker compose logs backend
    exit 1
}

if (-not $frontendReady) {
    Write-Host "Error: Frontend failed to start properly." -ForegroundColor Red
    docker compose logs frontend
    exit 1
}

Write-Host "`nPassportPAL application is now running!" -ForegroundColor Green
Write-Host "Frontend: http://localhost" -ForegroundColor Yellow
Write-Host "Backend API: http://localhost:5000" -ForegroundColor Yellow
Write-Host "`nUseful Docker commands:" -ForegroundColor Cyan
Write-Host "- View logs:        docker compose logs -f" -ForegroundColor Cyan
Write-Host "- Stop application: docker compose down" -ForegroundColor Cyan
Write-Host "- Rebuild:         docker compose up --build -d" -ForegroundColor Cyan