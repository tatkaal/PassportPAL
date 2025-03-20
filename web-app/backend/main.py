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
SEGMENTATION_MODEL_PATH = r"C:\Users\zerad\Desktop\Sujan\PassportPAL\notebooks\PassportPAL-Detector\yolo11m-seg-custom2\weights\best.pt"
CLASSIFICATION_MODEL_PATH = r"C:\Users\zerad\Desktop\Sujan\PassportPAL\checkpoints\custom_cnn\custom_cnn_model_scripted.pt"
CLASSIFICATION_METADATA_PATH = r"C:\Users\zerad\Desktop\Sujan\PassportPAL\checkpoints\custom_cnn\custom_cnn_model_metadata.json"

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

def get_transforms(img_size=224, use_augmentation=False):
    return {"test": FallbackTransforms()}

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

        transforms_dict = get_transforms(img_size=img_size, use_augmentation=False)
        inference_transform = transforms_dict['test']

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
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        seg_model, cls_model, class_names, transform = load_models()

        image_overlay = image.copy()
        h, w = image.shape[:2]

        results = seg_model(image)
        if results and results[0].masks is not None:
            r = results[0]
            mask_data = r.masks.data.cpu().numpy()  # shape: [N, H, W]
            boxes = r.boxes.data.cpu().numpy()      # shape: [N, 6] => x1,y1,x2,y2,conf,class

            if len(boxes) > 0:
                # Pick best detection by confidence
                best_idx = int(np.argmax(boxes[:, 4]))
                x1, y1, x2, y2, conf, cls_id = boxes[best_idx]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Convert predicted class index to name
                cls_id = int(cls_id)
                if isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                    class_name = class_names[cls_id]
                else:
                    class_name = f"Class_{cls_id}"

                # If best_idx is valid in mask_data
                if best_idx < len(mask_data):
                    mask = mask_data[best_idx]
                    mask = cv2.resize(mask, (w, h))
                    mask_bin = (mask > 0.5).astype(np.uint8) * 255
                else:
                    mask_bin = np.zeros((h, w), np.uint8)
                    cv2.rectangle(mask_bin, (x1, y1), (x2, y2), 255, -1)

                # Draw segmentation overlay in blue (B-channel) with higher alpha
                color_mask = np.zeros_like(image_overlay, dtype=np.uint8)

                # Fill mask area with red [B=0, G=0, R=255]
                color_mask[mask_bin == 255] = [0, 0, 255]

                # # Purple (magenta): B=255, G=0, R=255
                # color_mask[mask_bin == 255] = [255, 0, 255]

                alpha = 1  # more opaque
                image_overlay = cv2.addWeighted(image_overlay, 1.0, color_mask, alpha, 0)

                # Crop for classification
                cropped = image[y1:y2, x1:x2]
                predicted_class = class_name
                predicted_conf = float(conf)
                top3_predictions = [{"class": predicted_class, "confidence": predicted_conf}]
                
                # Try classification if we have a valid crop
                if cropped.size > 0:
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    inp = transform(cropped_rgb)["image"].unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = cls_model(inp)
                        probs = torch.nn.functional.softmax(out, dim=1)
                        values, indices = torch.topk(probs, 3)
                    best_cls = indices[0][0].item()
                    best_conf = values[0][0].item()
                    
                    # Convert top 3
                    top3_vals = [values[0][i].item() for i in range(values.shape[1])]
                    top3_idxs = [indices[0][i].item() for i in range(indices.shape[1])]

                    if isinstance(class_names, list):
                        top3_predictions = []
                        for i, idx_ in enumerate(top3_idxs):
                            label_ = class_names[idx_] if 0 <= idx_ < len(class_names) else f"Class_{idx_}"
                            top3_predictions.append({
                                "class": label_,
                                "confidence": float(top3_vals[i])
                            })
                        predicted_class = class_names[best_cls] if 0 <= best_cls < len(class_names) else f"Class_{best_cls}"
                        predicted_conf = best_conf
                    else:
                        predicted_class = f"Class_{best_cls}"
                        predicted_conf = best_conf
                        top3_predictions = [{"class": predicted_class, "confidence": best_conf}]
                
                # Draw bounding box with bigger text
                draw_thick_bbox(image_overlay, x1, y1, x2, y2, predicted_class)

            else:
                # No detections => fallback
                logger.warning("No bounding boxes found. Using fallback segmentation.")
                mask_bin, image_overlay, cropped = fallback_segmentation(image)
                predicted_class = "Document"
                predicted_conf = 0.0
                top3_predictions = [{"class": predicted_class, "confidence": 0.0}]
                # bounding box is drawn in fallback_segmentation

        else:
            # No masks => fallback
            logger.warning("No segmentation masks found. Using fallback segmentation.")
            mask_bin, image_overlay, cropped = fallback_segmentation(image)
            predicted_class = "Document"
            predicted_conf = 0.0
            top3_predictions = [{"class": predicted_class, "confidence": 0.0}]
            # bounding box is drawn in fallback_segmentation

        # Convert final overlay to base64
        segmentation_base64 = image_to_base64(image_overlay)

        return {
            "class": predicted_class,
            "confidence": float(predicted_conf),
            "top3_predictions": top3_predictions,
            "segmentation": segmentation_base64
        }
    except Exception as e:
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
        # Replace with your local samples folder as needed
        samples_dir = os.path.normpath("C:/Users/zerad/Desktop/Sujan/PassportPAL/data/samples")
        requested_path = os.path.normpath(path)
        
        if not requested_path.startswith(samples_dir):
            raise HTTPException(status_code=400, detail="Invalid sample path")
        if not os.path.exists(requested_path):
            raise HTTPException(status_code=404, detail="Sample file not found")

        response = FileResponse(requested_path)
        # For dev CORS
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
        return response
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
