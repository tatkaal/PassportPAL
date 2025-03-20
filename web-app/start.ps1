# PassportPAL Unified Startup Script
Write-Host "Starting PassportPAL Application..." -ForegroundColor Cyan

# Check if Python and Node.js are installed
try {
    $pythonVersion = (python --version 2>&1)
    Write-Host "Found $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH. Please install Python 3.8+ and try again." -ForegroundColor Red
    exit 1
}

try {
    $nodeVersion = (node --version 2>&1)
    Write-Host "Found Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Node.js is not installed or not in PATH. Please install Node.js 16+ and try again." -ForegroundColor Red
    exit 1
}

# Define directories
$backendDir = Join-Path $PSScriptRoot "backend"
$frontendDir = Join-Path $PSScriptRoot "frontend"
$venvPath = Join-Path $backendDir "venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

# Check if virtual environment exists, create if not
if (-not (Test-Path $pythonPath)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    Push-Location $backendDir
    python -m venv venv
    Pop-Location
}

# Install backend dependencies
Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
Push-Location $backendDir

# Try to use the activation script if execution policy allows
$activationScript = Join-Path $venvPath "Scripts\Activate.ps1"
$useDirectPython = $true

try {
    # Check execution policy
    $policy = Get-ExecutionPolicy
    Write-Host "Current PowerShell execution policy: $policy" -ForegroundColor Yellow
    
    if ($policy -ne "Restricted") {
        # Try to use activation script
        & $activationScript
        Write-Host "Virtual environment activated with script" -ForegroundColor Green
        $useDirectPython = $false
        
        # Install dependencies
        pip install -r requirements.txt
    }
} catch {
    Write-Host "Cannot use activation script due to execution policy restrictions." -ForegroundColor Yellow
    Write-Host "Using direct Python path instead." -ForegroundColor Yellow
    $useDirectPython = $true
}

if ($useDirectPython) {
    # Use direct Python path
    & $pythonPath -m pip install -r requirements.txt
}

# Start backend server
Write-Host "Starting backend server..." -ForegroundColor Yellow
if ($useDirectPython) {
    # Start backend directly with Python path
    Start-Process -FilePath $pythonPath -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000" -WorkingDirectory $backendDir -NoNewWindow
} else {
    # Start backend with activated environment
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backendDir'; python -m uvicorn main:app --host 0.0.0.0 --port 5000"
}
Pop-Location

# Wait a moment for the backend to start
Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Install frontend dependencies and start dev server
Write-Host "Setting up frontend..." -ForegroundColor Yellow
Push-Location $frontendDir
npm install
npm run dev
Pop-Location 