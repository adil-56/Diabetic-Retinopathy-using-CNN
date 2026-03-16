import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tensorflow as tf
from src.config import MODELS_DIR, IMG_SIZE, CLASS_NAMES

app = FastAPI(title="RetinaGuard Clinical API", version="2.0")

print("Loading model into memory...")
# Point to the newly converted .h5 file
model_path = os.path.join(MODELS_DIR, "dr_model_best.h5")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model successfully loaded!")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    model = None

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE[0], IMG_SIZE[1]))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)

def generate_clinical_triage(diagnosis, confidence, hba1c, years_diabetic):
    """Generates actionable medical advice based on AI prediction and patient data."""
    if confidence < 60.0:
        return "WARNING: AI confidence is low. Image quality may be poor. Please retake the scan or proceed with a manual clinical evaluation."
    
    advice = ""
    if diagnosis == "Healthy":
        advice = "No signs of diabetic retinopathy detected. Maintain routine annual eye exams."
        if hba1c > 7.0:
            advice += " However, your HbA1c is elevated. Strict glycemic control is advised to prevent future vascular damage."
            
    elif diagnosis in ["Mild DR", "Moderate DR"]:
        advice = "Early-to-moderate vascular damage detected. Schedule a comprehensive exam with an ophthalmologist within 3 months."
        if years_diabetic > 10:
            advice += " Given your long history of diabetes, close monitoring is critical to prevent progression."
            
    elif diagnosis in ["Proliferate DR", "Severe DR"]:
        advice = "URGENT: High risk of severe vision loss. Immediate referral to a retinal specialist for potential intervention (laser therapy/injections) is strongly recommended."
        
    return advice

@app.post("/predict")
async def predict_diabetic_retinopathy(
    file: UploadFile = File(...),
    age: int = Form(None),
    years_diabetic: int = Form(None),
    hba1c: float = Form(None)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        image_bytes = await file.read()
        processed_image = prepare_image(image_bytes)
        
        # 1. Run standard prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        diagnosis = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0])) * 100
        
        # 2. TEMPORARY MEMORY FIX: Disable Grad-CAM for Render Free Tier
        # We skip the heavy calculus that causes the Out-of-Memory crash.
        # Instead, we just convert the ORIGINAL image to base64 so the UI doesn't break.
        buffered = io.BytesIO()
        Image.open(io.BytesIO(image_bytes)).convert("RGB").save(buffered, format="JPEG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 3. Generate Clinical Triage
        triage_message = generate_clinical_triage(
            diagnosis, 
            confidence, 
            hba1c if hba1c else 6.5, 
            years_diabetic if years_diabetic else 5
        )
        
        return {
            "status": "success",
            "diagnosis": diagnosis,
            "confidence": f"{confidence:.2f}%",
            "triage_recommendation": triage_message,
            "heatmap": heatmap_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
