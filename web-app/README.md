# PassportPAL - ID Document Classification System

PassportPAL is a Simple document classification system that uses deep learning to accurately detect and classify identity documents from different countries.

![PassportPAL Screenshot](https://raw.githubusercontent.com/username/passport-pal/main/screenshot.png)

## Table of Contents

1. [Features](#features)
2. [Technology Stack](#technology-stack)
3. [Architecture](#architecture)
4. [Machine Learning Approach](#machine-learning-approach)
5. [Running the Application](#running-the-application)
6. [API Reference](#api-reference)
7. [Development Guide](#development-guide)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

## Features

- **Document Detection and Segmentation**: Automatically locates and extracts ID documents from images
- **Document Classification**: Identifies 10 different types of ID documents with high accuracy
- **Interactive UI**: User-friendly interface for uploading and analyzing documents
- **Sample Gallery**: Pre-loaded sample images for immediate testing
- **Real-time Processing**: Fast analysis with visual feedback
- **Containerized Deployment**: Easy setup with Docker and Docker Compose

## Technology Stack

### Backend

- **Python 3.10**: Core programming language
- **FastAPI**: High-performance API framework
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **Ultralytics YOLOv8**: Object detection and segmentation
- **Albumentations**: Image augmentation library

### Frontend

- **React 18**: UI framework
- **TailwindCSS**: Utility-first CSS framework
- **Vite**: Next-generation frontend tooling
- **React-Dropzone**: File upload handling
- **Axios**: HTTP client

### DevOps

- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Web server and reverse proxy

## Architecture

PassportPAL employs a two-stage machine learning pipeline:

1. **Document Segmentation**: The YOLOv8 segmentation model detects and extracts the document region from the input image.
2. **Document Classification**: The extracted document is passed to a Simple CNN classifier that identifies the specific document type.

![Architecture Diagram](https://raw.githubusercontent.com/username/passport-pal/main/architecture.png)

The application follows a standard client-server architecture:

- **Frontend**: React SPA served by Nginx
- **Backend API**: FastAPI service that processes images and runs ML models
- **ML Models**: Custom trained YOLOv11-seg and CNN models

## Machine Learning Approach

### Data Preparation and Model Training

#### Instance Segmentation (Document Detection)

- **Model**: YOLOv8m-seg (medium size variant)
- **Dataset**: 307 images collected and annotated using Roboflow
- **Data Split**: 215 training, 61 validation, 31 testing images
- **Annotation Process**: Initial auto-annotation through Roboflow with manual verification
- **Augmentations**: Applied 5× multiplication to training set only
  - Flip vertical
  - 90° rotation
  - ±15° rotation
  - ±10° horizontal and vertical shear
  - ±18° hue adjustment
  - ±24% brightness variation
  - ±15% exposure variation

#### Document Classification

- **Dataset Creation**: The trained segmentation model was used to crop the region of interest (ROI) containing ID documents from the original dataset
- **Model Architecture**: Custom CNN classification model
- **Performance**: Achieved 100% accuracy on test sets with 1.0 recall and 1.0 F1 score

### Reasoning Behind Technical Choices

1. **Two-Stage Pipeline**: Separating detection and classification provides better modularity and allows optimization of each task independently.

2. **YOLOv8 for Segmentation**:

   - State-of-the-art performance for object detection
   - Faster inference compared to other segmentation models
   - Better handling of varied document orientations and backgrounds

3. **Custom CNN for Classification**:

   - Focused specifically on the cropped document regions
   - Achieves high accuracy by eliminating background noise
   - Faster inference due to smaller input size (cropped images)

4. **Docker Containerization**:
   - Ensures consistent environment across development and production
   - Simplifies deployment and scaling
   - Isolates dependencies for better maintainability

## Running the Application

### Prerequisites

- **Docker**: Version 20.10.0 or higher
- **Docker Compose**: Version 2.0.0 or higher
- **Hardware**:
  - Minimum: 4GB RAM, dual-core CPU
  - Recommended: 8GB+ RAM, quad-core CPU
  - Optional: NVIDIA GPU with CUDA support for faster inference

### Step-by-Step Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/tatkaal/passportpal.git
   cd passport-pal
   ```

2. **Prepare model files**:
   Ensure the following model files are in the correct location:

   - `backend/models/custom_instance_segmentation.pt`
   - `backend/models/custom_cnn_model_scripted.pt`
   - `backend/models/custom_cnn_model_metadata.json`

3. **Start the application using Docker Compose**:

   On Windows:

   ```powershell
   .\start.ps1
   ```

   On Linux/Mac:

   ```bash
   chmod +x ./start.sh
   ./start.sh
   ```

4. **Access the web interface**:
   Open your browser and navigate to:

   ```
   http://localhost
   ```

5. **Stopping the application**:
   ```bash
   docker-compose down
   ```

### Manual Development Setup

For development purposes, you can run the components separately:

#### Backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

#### Frontend:

```bash
cd frontend
npm install
npm run dev
```

## API Reference

The backend exposes the following RESTful endpoints:

### `GET /api/status`

Check API and model availability.

**Response:**

```json
{
  "status": "online",
  "device": "cuda",
  "gpu_available": true
}
```

### `POST /api/analyze`

Analyze an uploaded document image.

**Request:** Form data with `file` field containing the image.

**Response:**

```json
{
  "class": "fin_id",
  "confidence": 0.982,
  "top3_predictions": [
    {
      "class": "fin_id",
      "confidence": 0.982
    },
    {
      "class": "est_id",
      "confidence": 0.015
    },
    {
      "class": "svk_id",
      "confidence": 0.003
    }
  ],
  "segmentation": "base64_encoded_image"
}
```

### `GET /api/get-sample`

Get a sample image for testing.

**Parameters:** `path` - Path to the sample image

**Response:** Binary image data

## Development Guide

### Project Structure

```
passport-pal/
├── backend/
│   ├── models/
│   │   ├── custom_instance_segmentation.pt
│   │   ├── custom_cnn_model_scripted.pt
│   │   └── custom_cnn_model_metadata.json
│   ├── dataset.py
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── public/
│   │   └── samples/
│   ├── src/
│   │   ├── App.jsx
│   │   └── ...
│   ├── package.json
│   ├── vite.config.js
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
├── start.ps1
├── start.sh
└── README.md
```

## Models

The ML models will be automatically downloaded during Docker build from the following Google Drive location:
[Model Files](https://drive.google.com/drive/folders/1qG6xU7eGEwTXxQWP5L6s2zuJ7FXs3SQB?usp=sharing)

- `custom_instance_segmentation.pt`: YOLOv8 segmentation model
- `custom_cnn_model_scripted.pt`: TorchScript classification model
- `custom_cnn_model_metadata.json`: Classification model metadata

Note: Internet connection is required during the first build to download the models.

## Docker Usage Guide

### First Run

The first time you run the application:

```bash
# This will:
# 1. Build Docker images
# 2. Download model files
# 3. Install all dependencies
# 4. Start the containers
.\start.ps1   # Windows
./start.sh    # Linux/Mac
```

### Subsequent Runs

After the initial setup:

```bash
# Quick start using existing images
docker-compose up -d

# Or use start script (will detect existing images)
.\start.ps1   # Windows
./start.sh    # Linux/Mac
```

### Common Docker Commands

```bash
# Stop the application
docker-compose down

# View logs
docker-compose logs

# Force rebuild images
docker-compose up --build -d

# Remove all containers and images (reset)
docker-compose down --rmi all
```

### How Docker Caching Works

1. **Images**: Docker caches built images locally

   - First run: Downloads and builds everything
   - Subsequent runs: Uses cached images
   - Images rebuilt only when Dockerfile or source code changes

2. **Models**: Downloaded during image build

   - Stored within the Docker image
   - No re-download needed unless you rebuild image

3. **Dependencies**: Cached in Docker layers
   - Requirements installed during image build
   - Cached unless requirements.txt changes

### Storage Locations

- Docker images: `/var/lib/docker` (Linux) or Docker Desktop VM (Windows/Mac)
- Models: Embedded in backend Docker image
- Source code: Mounted from host for development

## Acknowledgments

- The YOLOv8 model architecture by Ultralytics
- Roboflow for simplified dataset annotation
- The React and FastAPI communities
