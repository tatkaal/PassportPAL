# Get the current script directory and backend directory
$scriptDir = $PSScriptRoot
$rootDir = Split-Path -Parent $scriptDir
$backendDir = Join-Path $rootDir "backend"
$modelsDir = Join-Path $backendDir "models"

# Create models directory if it doesn't exist
if (-Not (Test-Path $modelsDir)) {
    Write-Host "Creating models directory: $modelsDir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

# Set to TLS 1.2 for HTTPS downloads
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Check if models already exist
$REQUIRED_FILES = @(
    "custom_instance_segmentation.pt",
    "custom_cnn_model_scripted.pt",
    "custom_cnn_model_metadata.json"
)

$ALL_FILES_EXIST = $true
foreach ($file in $REQUIRED_FILES) {
    if (-not (Test-Path (Join-Path $modelsDir $file))) {
        $ALL_FILES_EXIST = $false
        break
    }
}

if ($ALL_FILES_EXIST) {
    Write-Host "Models already exist, skipping download" -ForegroundColor Green
    exit 0
}

Write-Host "Downloading models from Google Drive..." -ForegroundColor Yellow

# Create a temporary directory for downloads
$TMP_DIR = New-TemporaryFile | ForEach-Object { 
    Remove-Item $_ -Force
    New-Item -ItemType Directory -Path $_ 
}
Push-Location $TMP_DIR

# Check if gdown is installed, if not install it
try {
    python -c "import gdown" 2>$null
} catch {
    Write-Host "Installing gdown..." -ForegroundColor Yellow
    python -m pip install gdown --quiet
}

# Download the models folder
python -m gdown "https://drive.google.com/drive/folders/1qG6xU7eGEwTXxQWP5L6s2zuJ7FXs3SQB?usp=sharing" --folder

# Find the downloaded directory (usually passportpal_models)
$DOWNLOAD_DIR = Get-ChildItem -Directory -Filter "*models" | Select-Object -First 1

if (-not $DOWNLOAD_DIR) {
    Write-Host "Error: Could not find downloaded models directory" -ForegroundColor Red
    Pop-Location
    Remove-Item -Recurse -Force $TMP_DIR
    exit 1
}

Push-Location $DOWNLOAD_DIR.FullName

# Verify that downloaded files exist
$ALL_FILES_DOWNLOADED = $true
foreach ($file in $REQUIRED_FILES) {
    if (-not (Test-Path $file)) {
        Write-Host "Error: Missing file $file after download" -ForegroundColor Red
        $ALL_FILES_DOWNLOADED = $false
        break
    }
}

if (-not $ALL_FILES_DOWNLOADED) {
    Pop-Location
    Pop-Location
    Remove-Item -Recurse -Force $TMP_DIR
    exit 1
}

# Move the models to the correct location
foreach ($file in $REQUIRED_FILES) {
    Move-Item -Force $file (Join-Path $modelsDir $file)
}

# Cleanup
Pop-Location
Pop-Location
Remove-Item -Recurse -Force $TMP_DIR

Write-Host "Models successfully downloaded and moved to $modelsDir!" -ForegroundColor Green 