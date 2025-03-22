from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import cv2
import numpy as np
import base64
import os
import logging
import traceback
import torch
from ultralytics import YOLO
import json
import sys
import importlib.util
from dataset import get_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PassportPAL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adjust these paths to your environment
SEGMENTATION_MODEL_PATH = r"models\custom_instance_segmentation.pt"
CLASSIFICATION_MODEL_PATH = r"models\custom_cnn_model_scripted.pt"
CLASSIFICATION_METADATA_PATH = r"models\custom_cnn_model_metadata.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

# Fallback transforms in case your dataset.py is missing
class FallbackTransforms:
    def __call__(self, image):
        img = cv2.resize(image, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return {"image": torch.from_numpy(img)}

def load_models():
    try:
        if not os.path.exists(SEGMENTATION_MODEL_PATH):
            raise FileNotFoundError(f"Segmentation model not found at: {SEGMENTATION_MODEL_PATH}")
        if not os.path.exists(CLASSIFICATION_MODEL_PATH):
            raise FileNotFoundError(f"Classification model not found at: {CLASSIFICATION_MODEL_PATH}")

        seg_model = YOLO(SEGMENTATION_MODEL_PATH)
        cls_model = torch.jit.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE)
        cls_model.to(DEVICE)
        cls_model.eval()

        metadata = load_metadata(CLASSIFICATION_METADATA_PATH)
        class_names = metadata.get("class_names", [])
        img_size = metadata.get("img_size", 224)

        # Get transforms dictionary from dataset.py
        transforms_dict = get_transforms(img_size=img_size, use_augmentation=False)  # Change to False
        
        # Get the appropriate transform for inference (test or val)
        inference_transform = transforms_dict.get('test', transforms_dict.get('val', FallbackTransforms()))
        
        return seg_model, cls_model, class_names, inference_transform
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def draw_thick_bbox(image, x1, y1, x2, y2, label):
    """
    Draw a bounding box with a bigger border and label text
    (no confidence), as requested.
    """
    # 2× thicker than previous (previous was 4, so let's do 8)
    box_thickness = 8

    # 3× bigger text (previous fontScale ~1.3 => let’s do ~4.0)
    font_scale = 4.0

    # Heavier text thickness
    text_thickness = 10

    color_box = (0, 0, 255)       # Red BGR
    color_bg_label = (0, 0, 150)  # Darker red background
    color_text = (255, 255, 255)  # White text

    # Draw thick bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color_box, box_thickness)

    # Label text sizing
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    top_left_label = (x1, y1 - th - 10)
    bottom_right_label = (x1 + tw + 10, y1)

    # Draw background for label
    cv2.rectangle(image, top_left_label, bottom_right_label, color_bg_label, -1)
    cv2.putText(
        image,
        label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color_text,
        text_thickness
    )

def fallback_segmentation(image):
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    image_with_overlay = image.copy()
    margin = min(h, w) // 20
    cv2.rectangle(image_with_overlay, (margin, margin), (w - margin, h - margin), (0, 0, 255), 8)
    cropped_region = image.copy()
    logger.info("Using fallback segmentation (entire image).")
    return mask, image_with_overlay, cropped_region

@app.post("/api/debug")
async def debug_request(request: Request):
    body = await request.body()
    headers = dict(request.headers)
    return {
        "headers": headers,
        "body_length": len(body),
        "content_type": headers.get("content-type", "Not specified")
    }

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Load the image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Load models and transforms
        seg_model, cls_model, class_names, transform = load_models()
        logger.info(f"Loaded models: segmentation and classification with {len(class_names)} classes")
        
        # Copy original image for visualization
        image_overlay = image.copy()
        h, w = image.shape[:2]
        logger.info(f"Input image size: {w}x{h}")

        # Step 1: Run segmentation model to find the document
        logger.info("Running segmentation model...")
        results = seg_model(image)
        
        # Initialize default values for the case when no document is found
        cropped = None
        predicted_class = "Unknown"
        predicted_conf = 0.0
        top3_predictions = [{"class": predicted_class, "confidence": predicted_conf}]
        
        # Check if segmentation was successful
        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            r = results[0]
            mask_data = r.masks.data.cpu().numpy()
            boxes = r.boxes.data.cpu().numpy()
            
            if len(boxes) > 0:
                # Find the best detection by confidence
                best_idx = int(np.argmax(boxes[:, 4]))
                x1, y1, x2, y2, conf, _ = boxes[best_idx]  # Ignore YOLO's class ID
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Ensure coordinates are within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                logger.info(f"Detected document region at [{x1},{y1},{x2},{y2}] with confidence {conf:.4f}")
                
                # Visualize the segmentation on the overlay image
                if best_idx < len(mask_data):
                    # Get segmentation mask
                    mask = mask_data[best_idx]
                    mask = cv2.resize(mask, (w, h))
                    mask_bin = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Draw segmentation overlay
                    color_mask = np.zeros_like(image_overlay, dtype=np.uint8)
                    color_mask[mask_bin == 255] = [0, 0, 255]  # Red in BGR
                    image_overlay = cv2.addWeighted(image_overlay, 1.0, color_mask, 0.5, 0)
                else:
                    # If no mask data, just use the bounding box
                    mask_bin = np.zeros((h, w), np.uint8)
                    cv2.rectangle(mask_bin, (x1, y1), (x2, y2), 255, -1)
                
                # Step 2: Crop the detected region for classification
                if x2 > x1 and y2 > y1:  # Ensure valid region
                    cropped = image[y1:y2, x1:x2].copy()
                    logger.info(f"Cropped region size: {cropped.shape[1]}x{cropped.shape[0]}")
                    
                    # Step 3: Classify the cropped image
                    if cropped.size > 0:
                        # Convert BGR to RGB for the model
                        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        
                        # Apply transforms to prepare for the model
                        logger.info(f"Applying transforms to prepare image for classification")
                        
                        # Fix: Check if transform is a dictionary or callable
                        if isinstance(transform, dict):
                            # If it's still a dictionary, get the test transform
                            transform_func = transform.get('test', transform.get('val', FallbackTransforms()))
                            transformed = transform_func(image=cropped_rgb)
                        else:
                            # If it's already a callable (like a composition)
                            transformed = transform(image=cropped_rgb)
                            
                        inp = transformed["image"].unsqueeze(0).to(DEVICE)
                        
                        # Run classification model
                        logger.info("Running classification model...")
                        with torch.no_grad():
                            out = cls_model(inp)
                            probs = torch.nn.functional.softmax(out, dim=1)
                            
                            # Get top predictions
                            k = min(3, len(class_names))
                            values, indices = torch.topk(probs, k)
                            
                            # Convert to list of predictions
                            top3_predictions = []
                            for i in range(indices.shape[1]):
                                idx = indices[0][i].item()
                                conf = values[0][i].item()
                                
                                # Ensure valid class index
                                if 0 <= idx < len(class_names):
                                    class_name = class_names[idx]
                                    logger.info(f"  Prediction {i+1}: {class_name} ({conf:.4f})")
                                    top3_predictions.append({
                                        "class": class_name,
                                        "confidence": float(conf)
                                    })
                            
                            # Get the top prediction
                            if top3_predictions:
                                predicted_class = top3_predictions[0]["class"]
                                predicted_conf = top3_predictions[0]["confidence"]
                                logger.info(f"Top prediction: {predicted_class} ({predicted_conf:.4f})")
                            else:
                                predicted_class = "Unknown"
                                predicted_conf = 0.0
                                logger.warning("No valid predictions found")
                    else:
                        logger.warning("Cropped region is empty, cannot classify")
                else:
                    logger.warning(f"Invalid crop region: [{x1},{y1},{x2},{y2}]")
                
                # Draw the bounding box with the top prediction label
                draw_thick_bbox(image_overlay, x1, y1, x2, y2, predicted_class)
            else:
                logger.warning("No bounding boxes found in segmentation results")
                mask_bin, image_overlay, cropped = fallback_segmentation(image)
        else:
            logger.warning("Segmentation model did not return masks")
            mask_bin, image_overlay, cropped = fallback_segmentation(image)
        
        # Convert the final visualization to base64
        segmentation_base64 = image_to_base64(image_overlay)
        
        # Prepare the response
        return {
            "class": predicted_class,
            "confidence": float(predicted_conf),
            "top3_predictions": top3_predictions,
            "segmentation": segmentation_base64
        }
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "device": str(DEVICE),
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/api/get-sample")
async def get_sample_image(path: str = Query(...)):
    try:
        # Get the absolute path to the frontend/public directory
        frontend_public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public'))
        
        # Remove any leading slash from the path parameter
        clean_path = path.lstrip('/')
        
        # Construct the full path
        full_path = os.path.join(frontend_public_dir, clean_path)
        
        # Verify the path is within the public directory
        if not os.path.abspath(full_path).startswith(frontend_public_dir):
            raise HTTPException(status_code=400, detail="Invalid sample path")
            
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Sample file not found")

        response = FileResponse(full_path)
        # For dev CORS
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
        return response
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
