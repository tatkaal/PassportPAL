# Set the models directory path relative to the script location
$SCRIPT_PATH = $MyInvocation.MyCommand.Path
$SCRIPT_DIR = Split-Path $SCRIPT_PATH -Parent
$MODELS_DIR = Join-Path $SCRIPT_DIR "models"

# Ensure models directory exists
if (-not (Test-Path $MODELS_DIR)) {
    New-Item -ItemType Directory -Path $MODELS_DIR | Out-Null
    Write-Host "Created models directory at $MODELS_DIR"
}

# Check if models already exist
$REQUIRED_FILES = @(
    "custom_instance_segmentation.pt",
    "custom_cnn_model_scripted.pt",
    "custom_cnn_model_metadata.json"
)

$ALL_FILES_EXIST = $true
foreach ($file in $REQUIRED_FILES) {
    if (-not (Test-Path (Join-Path $MODELS_DIR $file))) {
        $ALL_FILES_EXIST = $false
        break
    }
}

if ($ALL_FILES_EXIST) {
    Write-Host "Models already exist, skipping download"
    exit 0
}

Write-Host "Downloading models from Google Drive..."

# Create a temporary directory for downloads
$TMP_DIR = New-TemporaryFile | ForEach-Object { 
    Remove-Item $_ -Force
    New-Item -ItemType Directory -Path $_ 
}
Push-Location $TMP_DIR

# Download the models folder
python -m gdown "https://drive.google.com/drive/folders/1qG6xU7eGEwTXxQWP5L6s2zuJ7FXs3SQB?usp=sharing" --folder

# Find the downloaded directory (usually passportpal_models)
$DOWNLOAD_DIR = Get-ChildItem -Directory -Filter "*models" | Select-Object -First 1

if (-not $DOWNLOAD_DIR) {
    Write-Host "Error: Could not find downloaded models directory"
    Pop-Location
    Remove-Item -Recurse -Force $TMP_DIR
    exit 1
}

Push-Location $DOWNLOAD_DIR.FullName

# Verify that downloaded files exist
$ALL_FILES_DOWNLOADED = $true
foreach ($file in $REQUIRED_FILES) {
    if (-not (Test-Path $file)) {
        Write-Host "Error: Missing file $file after download"
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
    Move-Item -Force $file (Join-Path $MODELS_DIR $file)
}

# Cleanup
Pop-Location
Pop-Location
Remove-Item -Recurse -Force $TMP_DIR

Write-Host "Models successfully downloaded and moved to $MODELS_DIR!" 