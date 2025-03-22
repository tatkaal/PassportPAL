#!/bin/bash
set -e

# Set the models directory path
MODELS_DIR="/app/models"

# Ensure models directory exists
mkdir -p "$MODELS_DIR"

# Check if models already exist
if [ -f "$MODELS_DIR/custom_instance_segmentation.pt" ] && \
   [ -f "$MODELS_DIR/custom_cnn_model_scripted.pt" ] && \
   [ -f "$MODELS_DIR/custom_cnn_model_metadata.json" ]; then
    echo "Models already exist, skipping download"
    exit 0
fi

echo "Downloading models from Google Drive..."

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install --no-cache-dir gdown
fi

# Create a temporary directory for downloads
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# Download the models folder (retry up to 3 times if it fails)
max_attempts=3
attempt=1
success=false

while [ $attempt -le $max_attempts ] && [ "$success" = false ]; do
    echo "Download attempt $attempt of $max_attempts..."
    if gdown "https://drive.google.com/drive/folders/1qG6xU7eGEwTXxQWP5L6s2zuJ7FXs3SQB?usp=sharing" --folder; then
        success=true
    else
        echo "Download failed, retrying..."
        attempt=$((attempt+1))
        sleep 2
    fi
done

if [ "$success" = false ]; then
    echo "Error: Failed to download models after $max_attempts attempts"
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    exit 1
fi

# Find the downloaded directory (usually passportpal_models)
DOWNLOAD_DIR=$(find . -type d -name "*models" -print -quit)

if [ -z "$DOWNLOAD_DIR" ]; then
    echo "Error: Could not find downloaded models directory"
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    exit 1
fi

cd "$DOWNLOAD_DIR"

# Verify that downloaded files exist
if [ ! -f "custom_instance_segmentation.pt" ] || \
   [ ! -f "custom_cnn_model_scripted.pt" ] || \
   [ ! -f "custom_cnn_model_metadata.json" ]; then
    echo "Error: One or more model files are missing after download"
    cd - > /dev/null
    rm -rf "$TMP_DIR"
    exit 1
fi

# Move the models to the correct location
mv -f custom_instance_segmentation.pt custom_cnn_model_scripted.pt custom_cnn_model_metadata.json "$MODELS_DIR/"

# Cleanup
cd - > /dev/null
rm -rf "$TMP_DIR"

echo "Models successfully downloaded and moved to $MODELS_DIR!"
exit 0