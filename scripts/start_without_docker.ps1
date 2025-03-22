# PassportPAL Web Application - Start Without Docker
# This script starts both the frontend and backend without using Docker

# Get the current script directory and project root
$scriptDir = $PSScriptRoot
$rootDir = Split-Path -Parent $scriptDir
$backendDir = Join-Path $rootDir "backend"
$frontendDir = Join-Path $rootDir "frontend"

# Set environment variables
$Env:NODE_ENV = "development"

# Show banner
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "  PassportPAL Web Application   " -ForegroundColor Cyan
Write-Host "  Non-Docker Version Launcher   " -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-CommandExists {
    param ($command)
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

# Check prerequisites
$missingPrereqs = @()

if (-Not (Test-CommandExists "python")) {
    $missingPrereqs += "Python 3.10 or higher"
}

if (-Not (Test-CommandExists "npm")) {
    $missingPrereqs += "Node.js and npm"
}

if ($missingPrereqs.Count -gt 0) {
    Write-Host "Error: Missing prerequisites:" -ForegroundColor Red
    foreach ($prereq in $missingPrereqs) {
        Write-Host " - $prereq" -ForegroundColor Red
    }
    Write-Host "Please install missing prerequisites and try again." -ForegroundColor Red
    exit 1
}

# Check Python version
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pythonVerMajor, $pythonVerMinor = $pythonVersion.Split('.')
if ([int]$pythonVerMajor -lt 3 -or ([int]$pythonVerMajor -eq 3 -and [int]$pythonVerMinor -lt 10)) {
    Write-Host "Error: Python 3.10 or higher is required. Found version $pythonVersion" -ForegroundColor Red
    exit 1
}

# Create and activate Python virtual environment for the backend
Write-Host "Setting up Python virtual environment..." -ForegroundColor Cyan

# Check if venv exists, create if it doesn't
$venvDir = Join-Path $backendDir "venv"
if (-Not (Test-Path $venvDir)) {
    Write-Host "Creating new Python virtual environment..." -ForegroundColor Yellow
    python -m venv $venvDir
    if (-Not $?) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
& $activateScript

# Install backend requirements
Write-Host "Installing backend dependencies..." -ForegroundColor Cyan
Set-Location $backendDir
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if (-Not $?) {
    Write-Host "Failed to install backend dependencies." -ForegroundColor Red
    deactivate
    exit 1
}

# Download ML models
Write-Host "Checking for ML models..." -ForegroundColor Cyan
$modelScript = Join-Path $scriptDir "download_models.ps1"

if (Test-Path $modelScript) {
    & $modelScript
    if (-Not $?) {
        Write-Host "Failed to download models. Please check your internet connection and try again." -ForegroundColor Red
        deactivate
        exit 1
    }
} else {
    Write-Host "Model download script not found at: $modelScript" -ForegroundColor Red
    deactivate
    exit 1
}

# Setup frontend
Write-Host "Setting up frontend..." -ForegroundColor Cyan
Set-Location $frontendDir

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
npm install
if (-Not $?) {
    Write-Host "Failed to install frontend dependencies." -ForegroundColor Red
    deactivate
    exit 1
}

# Start backend and frontend in separate processes
Write-Host "Starting the application..." -ForegroundColor Green

# Start backend first in a new PowerShell window
Write-Host "Starting backend..." -ForegroundColor Yellow
$backendProcess = Start-Process powershell -ArgumentList "-Command", "Set-Location '$backendDir'; & '$venvDir\Scripts\Activate.ps1'; python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload" -WindowStyle Normal -PassThru

# Wait for backend to be ready
Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
$backendReady = $false
$maxAttempts = 5
$attempts = 0

while (-Not $backendReady -and $attempts -lt $maxAttempts) {
    $attempts++
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/api/status" -Method Get -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $backendContent = $response.Content | ConvertFrom-Json
            Write-Host "Backend status: $($backendContent.status), Device: $($backendContent.device)" -ForegroundColor Green
            $backendReady = $true
            Write-Host "Backend is ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "Waiting for backend to start (attempt $attempts of $maxAttempts)..." -ForegroundColor Yellow
    }
    
    if (-Not $backendReady -and $attempts -lt $maxAttempts) {
        Write-Host "Waiting 30 seconds before next check..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
    }
}

if (-Not $backendReady) {
    Write-Host "Backend failed to start after $maxAttempts attempts. Please check the backend window for errors." -ForegroundColor Red
    exit 1
}

# Now start the frontend in a new window
Write-Host "Starting frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-Command", "Set-Location '$frontendDir'; npm run dev; Read-Host 'Press Enter to close'" -WindowStyle Normal

# Show instructions
Write-Host "`nApplication is now running!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Backend: http://localhost:5000" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop the application" -ForegroundColor Yellow

# Keep the script running to maintain the PowerShell windows
try {
    while ($true) {
        Start-Sleep -Seconds 5
        # Check if backend is still running
        if ($backendProcess.HasExited) {
            Write-Host "Backend process has stopped. Exiting..." -ForegroundColor Red
            exit 1
        }
    }
} finally {
    # This block will run when Ctrl+C is pressed
    Write-Host "Shutting down..." -ForegroundColor Yellow
}