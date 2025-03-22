# Start the PassportPAL Web Application using Docker Compose
Write-Host "Starting PassportPAL Web Application..." -ForegroundColor Green

# Check if Docker is running
if (-not (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Desktop is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Download models if they don't exist
Write-Host "Checking for ML models..." -ForegroundColor Cyan
$scriptDir = $PSScriptRoot
$rootDir = Split-Path -Parent $scriptDir
$modelScript = Join-Path $scriptDir "download_models.ps1"

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

# Navigate to the root directory where docker-compose.yml is located
Set-Location $rootDir

# Check existing containers and images
$backendExists = docker ps -a --filter "name=passport-pal-backend" --format "{{.Names}}" | Select-String "passport-pal-backend"
$frontendExists = docker ps -a --filter "name=passport-pal-frontend" --format "{{.Names}}" | Select-String "passport-pal-frontend"
$backendImageExists = docker images "passport-pal-backend" -q
$frontendImageExists = docker images "passport-pal-frontend" -q

# Stop any existing containers
Write-Host "Stopping any existing containers..." -ForegroundColor Cyan
docker compose down

# Ask if user wants to rebuild
$rebuild = "partial"
if ($backendImageExists -or $frontendImageExists) {
    Write-Host "`nExisting images found:" -ForegroundColor Cyan
    if ($backendImageExists) { Write-Host "- Backend image exists" -ForegroundColor White }
    if ($frontendImageExists) { Write-Host "- Frontend image exists" -ForegroundColor White }
    
    Write-Host "`nRebuild options:" -ForegroundColor Cyan
    Write-Host "1. Use existing images (fastest)" -ForegroundColor White
    Write-Host "2. Rebuild only missing or failed images (recommended)" -ForegroundColor White
    Write-Host "3. Force rebuild all images (slowest)" -ForegroundColor White
    $option = Read-Host "Select an option (1-3) [2]"
    
    if ($option -eq "1") {
        $rebuild = "none"
    } elseif ($option -eq "3") {
        $rebuild = "all"
    }
}

# Conditionally rebuild images
if ($rebuild -eq "all") {
    # Force rebuild all
    Write-Host "Removing existing images for complete rebuild..." -ForegroundColor Cyan
    if ($backendImageExists) { docker image rm passport-pal-backend:latest -f }
    if ($frontendImageExists) { docker image rm passport-pal-frontend:latest -f }
    
    Write-Host "Building all containers from scratch..." -ForegroundColor Cyan
    docker compose build --no-cache
    
} elseif ($rebuild -eq "partial") {
    # Selective rebuild
    $buildCommand = "docker compose build"
    $needRebuild = $false
    
    if (-not $backendImageExists) {
        Write-Host "Backend image not found, will build it..." -ForegroundColor Cyan
        $buildCommand += " backend"
        $needRebuild = $true
    }
    
    if (-not $frontendImageExists) {
        Write-Host "Frontend image not found, will build it..." -ForegroundColor Cyan
        $buildCommand += " frontend"
        $needRebuild = $true
    }
    
    if ($needRebuild) {
        Write-Host "Building only necessary containers..." -ForegroundColor Cyan
        Invoke-Expression $buildCommand
    } else {
        Write-Host "All images exist, skipping build..." -ForegroundColor Green
    }
}

# Start the containers
Write-Host "Starting containers..." -ForegroundColor Cyan
docker compose up -d

# Wait for services to be ready
Write-Host "Waiting for services to start..." -ForegroundColor Cyan
$maxAttempts = 15  # Reduced from 60 to 15 to avoid long wait times
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
        if ($attempts % 3 -eq 0) {  # Only check logs every 3 attempts
            Write-Host "Checking frontend logs..." -ForegroundColor Yellow
            docker logs passport-pal-frontend --tail 5
        }
    }

    if (-not ($backendReady -and $frontendReady)) {
        Start-Sleep -Seconds 2
    }
}

# Check if at least the backend is working
if ($backendReady -and -not $frontendReady) {
    Write-Host "Frontend failed to start properly, but backend is working." -ForegroundColor Yellow
    Write-Host "You can still use the API directly at http://localhost:5000" -ForegroundColor Yellow
    Write-Host "Checking frontend container logs for issues:" -ForegroundColor Yellow
    docker logs passport-pal-frontend --tail 20
    
    # Ask if user wants to force restart frontend
    $restart = Read-Host "`nDo you want to try restarting the frontend container? (y/n)"
    if ($restart -eq "y") {
        Write-Host "Restarting frontend container..." -ForegroundColor Cyan
        docker restart passport-pal-frontend
        Write-Host "Frontend container restarted. Try accessing http://localhost in your browser." -ForegroundColor Green
    }
} elseif (-not $backendReady) {
    Write-Host "Error: Backend failed to start properly." -ForegroundColor Red
    docker compose logs backend
    exit 1
}

Write-Host "`nPassportPAL application is now running!" -ForegroundColor Green
if ($frontendReady) {
    Write-Host "Frontend: http://localhost" -ForegroundColor Yellow
}
Write-Host "Backend API: http://localhost:5000" -ForegroundColor Yellow
Write-Host "`nUseful Docker commands:" -ForegroundColor Cyan
Write-Host "- View logs:        docker compose logs -f" -ForegroundColor Cyan
Write-Host "- Stop application: docker compose down" -ForegroundColor Cyan
Write-Host "- Cleanup space:    docker system prune -a" -ForegroundColor Cyan