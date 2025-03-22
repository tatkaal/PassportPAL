#!/bin/bash
# PassportPAL - Download Machine Learning Models
# This script downloads pre-trained ML models used by the application

# Get the current script directory and backend directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"
MODELS_DIR="$BACKEND_DIR/models"

# Set up color for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create models directory if it doesn't exist
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${CYAN}Creating models directory: $MODELS_DIR${NC}"
    mkdir -p "$MODELS_DIR"
fi

# Define the required model files
REQUIRED_FILES=(
    "custom_instance_segmentation.pt"
    "custom_cnn_model_scripted.pt"
    "custom_cnn_model_metadata.json"
)

# Check if all models already exist
ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$MODELS_DIR/$file" ]; then
        ALL_FILES_EXIST=false
        break
    fi
done

if [ "$ALL_FILES_EXIST" = true ]; then
    echo -e "${GREEN}Models already exist, skipping download${NC}"
    exit 0
fi

echo -e "${YELLOW}Downloading models from Google Drive...${NC}"

# Create a temporary directory for downloads
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR" || { echo -e "${RED}Failed to create temporary directory${NC}"; exit 1; }

# Check if gdown is installed, if not install it
if ! command -v gdown &> /dev/null && ! python3 -c "import gdown" &> /dev/null; then
    echo -e "${YELLOW}Installing gdown...${NC}"
    pip install gdown --quiet
fi

# Download the models folder
python -m gdown "https://drive.google.com/drive/folders/1qG6xU7eGEwTXxQWP5L6s2zuJ7FXs3SQB?usp=sharing" --folder

# Find the downloaded directory (usually passportpal_models)
DOWNLOAD_DIR=$(find . -type d -name "*models" | head -n 1)

if [ -z "$DOWNLOAD_DIR" ]; then
    echo -e "${RED}Error: Could not find downloaded models directory${NC}"
    cd "$SCRIPT_DIR" || exit 1
    rm -rf "$TMP_DIR"
    exit 1
fi

cd "$DOWNLOAD_DIR" || { echo -e "${RED}Failed to change to downloaded directory${NC}"; exit 1; }

# Verify that downloaded files exist
ALL_FILES_DOWNLOADED=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: Missing file $file after download${NC}"
        ALL_FILES_DOWNLOADED=false
        break
    fi
done

if [ "$ALL_FILES_DOWNLOADED" = false ]; then
    cd "$SCRIPT_DIR" || exit 1
    rm -rf "$TMP_DIR"
    exit 1
fi

# Move the models to the correct location
for file in "${REQUIRED_FILES[@]}"; do
    mv -f "$file" "$MODELS_DIR/$file"
done

# Cleanup
cd "$SCRIPT_DIR" || exit 1
rm -rf "$TMP_DIR"

echo -e "${GREEN}Models successfully downloaded and moved to $MODELS_DIR!${NC}"
exit 0