# PassportPAL - ID Document Analyzer

A modern web application for analyzing ID documents using deep learning models. The application uses YOLOv8 for instance segmentation and a custom CNN for document classification.

## Features

- Modern, responsive UI built with React and Tailwind CSS
- Drag-and-drop image upload
- Real-time image preview
- Document segmentation using YOLOv8
- Document classification using custom CNN
- Beautiful visualization of results
- Fast inference using ONNX runtime

## Prerequisites

- Python 3.8+
- Node.js 16+
- ONNX Runtime
- OpenCV
- FastAPI
- React
- Tailwind CSS

## Project Structure

```
web-app/
├── frontend/          # React frontend application
│   ├── src/          # Source code
│   ├── public/       # Static files
│   └── package.json  # Frontend dependencies
└── backend/          # FastAPI backend service
    ├── main.py       # Backend application
    └── requirements.txt  # Backend dependencies
```

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
```bash
cd web-app/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

The backend server will run on http://localhost:5000

### Frontend Setup

1. Install dependencies:
```bash
cd web-app/frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend application will run on http://localhost:5173

## Usage

1. Open your browser and navigate to http://localhost:5173
2. Drag and drop an ID document image or click to select one
3. Click "Analyze Image" to process the document
4. View the results including:
   - Document type classification
   - Confidence score
   - Segmentation mask

## API Endpoints

- POST `/api/analyze`: Upload and analyze an image
  - Request: Multipart form data with image file
  - Response: JSON with classification results and segmentation mask

## Model Information

The application uses two ONNX models:
1. YOLOv8 segmentation model for document detection and segmentation
2. Custom CNN model for document classification

## Contributing

Feel free to submit issues and enhancement requests! 