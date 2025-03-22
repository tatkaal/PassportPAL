# PassportPAL - ID Document Classification System

PassportPAL is a Simple document classification system that uses deep learning to accurately detect, segement and classify identity documents from different countries.

## Table of Contents

1. [Features](#features)
2. [Technology Stack](#technology-stack)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Machine Learning Approach](#machine-learning-approach)
6. [Running the Application](#running-the-application)
7. [API Reference](#api-reference)
8. [Development Guide](#development-guide)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

## Features

- **Document Detection and Segmentation**: Automatically locates and extracts ID documents from images
- **Document Classification**: Identifies 10 different types of ID documents with high accuracy
- **Interactive UI**: User-friendly interface for uploading and analyzing documents
- **Sample Gallery**: Pre-loaded sample images for immediate testing
- **Real-time Processing**: Fast analysis with visual feedback
- **Containerized Deployment**: Easy setup with Docker and Docker Compose

## Technology Stack

### Backend

- **Python 3.12**: Core programming language
- **FastAPI**: High-performance API framework
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **Ultralytics YOLOv11**: Dataset preparation for Classification
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

1. **Document Segmentation**: The YOLOv11 segmentation model detects and extracts the document region from the input image.
2. **Document Classification**: The extracted document is passed to a Simple CNN classifier that identifies the specific document type.


The application follows a standard client-server architecture:

- **Frontend**: React SPA served by Nginx
- **Backend API**: FastAPI service that processes images and runs ML models
- **ML Models**: Custom trained YOLOv11-seg and CNN models

## Project Structure

```
PassportPAL/
├── README.md                  # Main project documentation with setup instructions
├── LICENSE                    # License file
├── docker-compose.yml         # Main Docker Compose configuration
├── .gitignore                 # Git ignore file
├── Task1/                     # Task1 folder
│   └── probability_of_finding_plane.pdf  # PDF file
│
├── dataset/                   # Dataset directory
│   ├── images/                # Raw dataset provided
│   ├── annotated_images/      # Images annotated with Roboflow
│   ├── cropped_images/        # Images after segmentation
│   └── samples/               # Sample images for UI and training graphs
│
├── model_development/         # Model development code
│   ├── classification/        # Classification model development
│   │   ├── train.py           # Training script
│   │   ├── simple_cnn_classification.py  # CNN model implementation
│   │   ├── models.py          # Model architecture definitions
│   │   └── dataset.py         # Dataset handling
│   │
│   └── segmentation/          # Segmentation model development
│       ├── roi_instance_segmentation.py  # Segmentation implementation
│       ├── failed_approaches.py # Documentation of approaches that didn't work
│       └── PassportPAL-Segmentation-Model/  # Segmentation model files
│
├── scripts/                   # Scripts directory
│   ├── start.ps1              # Docker startup script (Windows)
│   ├── start.sh               # Docker startup script (Linux/Mac)
│   ├── start_without_docker.ps1  # Local startup script (Windows)
│   ├── start_without_docker.sh   # Local startup script (Linux/Mac)
│   ├── docker-cleanup.ps1     # Docker cleanup script (Windows) 
│   ├── docker-cleanup.sh      # Docker cleanup script (Linux/Mac)
│   ├── download_models.ps1    # Model download script (Windows)
│   └── download_models.sh     # Model download script (Linux/Mac)
│
├── backend/                   # Backend service
│   ├── Dockerfile             # Backend Docker configuration
│   ├── requirements.txt       # Python dependencies
│   ├── main.py                # FastAPI application
│   ├── dataset.py             # Dataset handling code
│   ├── models/                # Directory for storing ML models
│   └── .dockerignore          # Docker ignore file
│
└── frontend/                  # Frontend service
    ├── Dockerfile             # Frontend Docker configuration
    ├── package.json           # NPM package configuration
    ├── public/                # Static assets
    │   └── samples/           # Sample images for testing
    ├── src/                   # React source code
    └── .dockerignore          # Docker ignore file
```

## Machine Learning Approach

### Data Preparation and Model Training

#### Instance Segmentation (Document Detection)

- **Model**: YOLOv11m-seg
- **Dataset**: 307 images collected and annotated using Roboflow
- **Data Split**: 215 training, 61 validation, 31 testing images
- **Annotation Process**: Initial auto-annotation through Roboflow with manual verification
- **Augmentations**: Following augmentation was applied to create 7 versions of each source image:
   - 50% probability of vertical flip
   - Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
   - Random rotation of between -15 and +15 degrees
   - Random shear of between -10° to +10° horizontally and -10° to +10° vertically
   - Random brigthness adjustment of between -24 and +24 percent
   - Random exposure adjustment of between -15 and +15 percent

#### Document Classification

- **Dataset Creation**: The trained segmentation model was used to crop the region of interest (ROI) containing ID documents from the original dataset
- **Model Architecture**: Custom CNN classification model
- **Performance**: 

### Reasoning Behind Technical Choices

1. **Two-Stage Pipeline**: Separating detection and classification provides better modularity and allows optimization of each task independently.

2. **YOLOv11 for Segmentation**:

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

- **Docker**
- **Docker Compose**

### Step-by-Step Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/tatkaal/passportpal.git
   cd passportpal
   ```

2. **Start the application using Docker Compose**:

   On Windows:

   ```powershell
   .\scripts\start.ps1
   ```

   On Linux/Mac:

   ```bash
   chmod +x ./scripts/start.sh
   ./scripts/start.sh
   ```

3. **Access the web interface**:
   Open your browser and navigate to:

   ```
   http://localhost
   ```

### Manual Development Setup

For development purposes, you can run the components separately:

#### Backend and Frontend without Docker:

On Windows:
```powershell
.\scripts\start_without_docker.ps1
```

On Linux/Mac:
```bash
chmod +x ./scripts/start_without_docker.sh
./scripts/start_without_docker.sh
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


3. **Verify model downloads**:
   Make sure the machine learning models were downloaded correctly. You can manually download them using:
   ```bash
   # On Windows
   .\scripts\download_models.ps1
   
   # On Linux/Mac
   ./scripts/download_models.sh
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLO model
- Roboflow for simplified dataset annotation
- The React and FastAPI communities
