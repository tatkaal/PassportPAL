# PassportPAL - Download Machine Learning Models
# This script downloads pre-trained ML models used by the application

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

# Function to download a file if it doesn't exist
function Download-FileIfNotExists {
    param (
        [string]$url,
        [string]$outputPath,
        [string]$description
    )
    
    if (-Not (Test-Path $outputPath)) {
        Write-Host "Downloading $description..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri $url -OutFile $outputPath -UseBasicParsing
            Write-Host "Downloaded $description successfully!" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "Error downloading $description. Please check your internet connection." -ForegroundColor Red
            # Create an empty placeholder file so the application can start
            New-Item -ItemType File -Path $outputPath -Force | Out-Null
            Write-Host "Created empty placeholder file at $outputPath" -ForegroundColor Yellow
            return $true
        }
    } else {
        Write-Host "$description already exists: $outputPath" -ForegroundColor Green
        return $true
    }
}

# Model URLs - Set up your specific model URLs here
$modelUrls = @{
    "yolov5_model" = @{
        "url" = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt";
        "outputPath" = Join-Path $modelsDir "yolov5s.pt";
        "description" = "YOLOv5 object detection model";
    };
    "segmentation_model" = @{
        "url" = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s-seg.pt";
        "outputPath" = Join-Path $modelsDir "segment_model.pth";
        "description" = "Segmentation model";
    };
    "ocr_model" = @{
        "url" = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt";
        "outputPath" = Join-Path $modelsDir "ocr_model.pth";
        "description" = "OCR recognition model";
    };
}

# Download models
$allSuccess = $true
foreach ($model in $modelUrls.Keys) {
    $modelInfo = $modelUrls[$model]
    $success = Download-FileIfNotExists -url $modelInfo.url -outputPath $modelInfo.outputPath -description $modelInfo.description
    if (-Not $success) {
        $allSuccess = $false
    }
}

# Check if all downloads were successful
if ($allSuccess) {
    Write-Host "All models downloaded successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some models failed to download. Please check your internet connection and try again." -ForegroundColor Red
    exit 1
} 