import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models

MODELS_DIR = "models"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Healthy", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

app = FastAPI(title="RetinaGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(MODELS_DIR, "dr_model_best.keras")
model = None

def build_architecture():
    # 1. We rebuild your exact EfficientNetB3 architecture directly in memory.
    # weights=None prevents the server from trying to download data from the internet.
    base_model = EfficientNetB3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None 
    )
    
    recreated_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    return recreated_model

@app.on_event("startup")
async def load_model():
    global model
    try:
        print("1. Constructing clean model architecture in memory...")
        model = build_architecture()
        
        print(f"2. Pouring trained weights from {MODEL_PATH} into architecture...")
        # 3. THE BYPASS: load_weights() extracts only the numbers and ignores the broken blueprint!
        model.load_weights(MODEL_PATH)
        
        print("✅ Model weights loaded successfully into memory! Server is ALIVE.")
    except Exception as e:
        print(f"🚨 CRITICAL ERROR loading model: {e}")

@app.get("/")
async def health_check():
    return {"status": "Online", "message": "RetinaGuard AI Backend is running perfectly!"}

@app.post("/predict")
async def predict_dr(
    file: UploadFile = File(...),
    age: str = Form("45"),
    years_diabetic: str = Form("5"),
    hba1c: str = Form("6.5")
):
    if model is None:
        raise HTTPException(status_code=500, detail="Server Error: Model failed to load.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        diagnosis = CLASS_NAMES[class_idx]
        
        triage_recommendation = f"Patient Profile (Age: {age}, HbA1c: {hba1c}%). "
        if class_idx == 0:
            triage_recommendation += "Routine annual retinal screening recommended."
        elif class_idx in [1, 2]:
            triage_recommendation += "Schedule follow-up with ophthalmologist within 3-6 months. Strict glycemic control advised."
        else:
            triage_recommendation += "URGENT referral to a retina specialist. Extremely high risk of severe vision complications."

        # Add explainability import here so it doesn't crash if optional 
        heatmap_base64 = None
        try:
            import sys
            # ensure src is in path
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from src.explainability import get_gradcam_heatmap, overlay_heatmap
            import cv2
            
            heatmap = get_gradcam_heatmap(img_array, model)
            superimposed_img = overlay_heatmap(image_bytes, heatmap, alpha=0.5)
            
            # Convert superimposed img to base64
            # superimposed_img is RGB, convert to BGR for cv2 imencode
            import base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
            heatmap_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception as expl_err:
            print(f"Heatmap generation failed (non-fatal): {expl_err}")

        return {
            "diagnosis": diagnosis,
            "confidence": f"{confidence * 100:.2f}%",
            "triage_recommendation": triage_recommendation,
            "heatmap": heatmap_base64 
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))