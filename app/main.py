# # import os
# # import io
# # import numpy as np
# # from PIL import Image
# # from fastapi import FastAPI, UploadFile, File, HTTPException
# # import tensorflow as tf
# # from src.config import MODELS_DIR, IMG_SIZE, CLASS_NAMES

# # # 1. Initialize the FastAPI Application
# # app = FastAPI(
# #     title="RetinaGuard Diagnostic API",
# #     description="An AI-powered API for Diabetic Retinopathy screening.",
# #     version="1.0"
# # )

# # # 2. Load the Pre-Trained Model Globally
# # # This ensures the heavy model is loaded into memory exactly ONCE when the server starts.
# # print("Loading model into memory...")
# # model_path = os.path.join(MODELS_DIR, "dr_model_best.keras")

# # try:
# #     model = tf.keras.models.load_model(model_path)
# #     print("Model successfully loaded and ready for inference!")
# # except Exception as e:
# #     print(f"CRITICAL ERROR: Could not load the model from {model_path}. Details: {e}")
# #     model = None

# # # 3. Helper Function: Image Preprocessing
# # def prepare_image(image_bytes):
# #     """
# #     Takes raw uploaded bytes, converts them to an image, resizes to our model's 
# #     expected input size, and transforms it into a TensorFlow batch.
# #     """
# #     # Open the image from bytes and ensure it is RGB
# #     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
# #     # Resize to exactly 224x224 (as defined in our config)
# #     img = img.resize((IMG_SIZE[0], IMG_SIZE[1]))
    
# #     # Convert the image to a numpy array for the neural network
# #     img_array = tf.keras.preprocessing.image.img_to_array(img)
    
# #     # The model expects a batch of images (e.g., [1, 224, 224, 3])
# #     # We expand the dimensions to create a batch of 1
# #     img_array = np.expand_dims(img_array, axis=0)
    
# #     return img_array

# # # 4. The Core Prediction Endpoint
# # @app.post("/predict")
# # async def predict_diabetic_retinopathy(file: UploadFile = File(...)):
# #     """
# #     Accepts an uploaded retinal image, passes it through the EfficientNet model, 
# #     and returns a JSON response containing the diagnosis.
# #     """
# #     if model is None:
# #         raise HTTPException(status_code=500, detail="The Machine Learning model is not loaded.")
    
# #     if not file.content_type.startswith("image/"):
# #         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

# #     try:
# #         # Read the uploaded file into memory
# #         image_bytes = await file.read()
        
# #         # Preprocess the image
# #         processed_image = prepare_image(image_bytes)
        
# #         # Run the neural network inference
# #         predictions = model.predict(processed_image)
        
# #         # Find the highest probability class
# #         predicted_class_idx = np.argmax(predictions[0])
# #         predicted_class_name = CLASS_NAMES[predicted_class_idx]
        
# #         # Calculate the confidence percentage
# #         confidence = float(np.max(predictions[0])) * 100
        
# #         # Return the professional JSON response
# #         return {
# #             "status": "success",
# #             "filename": file.filename,
# #             "diagnosis": predicted_class_name,
# #             "confidence": f"{confidence:.2f}%"
# #         }
        
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")



# import os
# import io
# import base64
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, UploadFile, File, HTTPException
# import tensorflow as tf
# from src.config import MODELS_DIR, IMG_SIZE, CLASS_NAMES
# from src.explainability import get_gradcam_heatmap, overlay_heatmap

# app = FastAPI(title="RetinaGuard Diagnostic API", version="1.0")

# print("Loading model into memory...")
# model_path = os.path.join(MODELS_DIR, "dr_model_best.keras")
# try:
#     model = tf.keras.models.load_model(model_path)
#     print("Model successfully loaded!")
# except Exception as e:
#     print(f"CRITICAL ERROR: {e}")
#     model = None

# def prepare_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize((IMG_SIZE[0], IMG_SIZE[1]))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     return np.expand_dims(img_array, axis=0)

# @app.post("/predict")
# async def predict_diabetic_retinopathy(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded.")
    
#     try:
#         image_bytes = await file.read()
#         processed_image = prepare_image(image_bytes)
        
#         # 1. Run standard prediction
#         predictions = model.predict(processed_image)
#         predicted_class_idx = np.argmax(predictions[0])
#         diagnosis = CLASS_NAMES[predicted_class_idx]
#         confidence = float(np.max(predictions[0])) * 100
        
#         # 2. Generate Explainable AI Heatmap
#         heatmap = get_gradcam_heatmap(processed_image, model)
#         overlaid_img_array = overlay_heatmap(image_bytes, heatmap)
        
#         # 3. Convert the generated image to a secure base64 string
#         overlaid_img_pil = Image.fromarray(overlaid_img_array)
#         buffered = io.BytesIO()
#         overlaid_img_pil.save(buffered, format="JPEG")
#         heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
#         return {
#             "status": "success",
#             "diagnosis": diagnosis,
#             "confidence": f"{confidence:.2f}%",
#             "heatmap": heatmap_base64
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tensorflow as tf
from src.config import MODELS_DIR, IMG_SIZE, CLASS_NAMES
from src.explainability import get_gradcam_heatmap, overlay_heatmap

app = FastAPI(title="RetinaGuard Clinical API", version="2.0")

print("Loading model into memory...")
model_path = os.path.join(MODELS_DIR, "dr_model_best.keras")
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
        
        # 2. Generate Explainable AI Heatmap
        heatmap = get_gradcam_heatmap(processed_image, model)
        overlaid_img_array = overlay_heatmap(image_bytes, heatmap)
        
        # Convert the generated image to a secure base64 string
        overlaid_img_pil = Image.fromarray(overlaid_img_array)
        buffered = io.BytesIO()
        overlaid_img_pil.save(buffered, format="JPEG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 3. Generate Clinical Triage
        # Default to safe values if the patient didn't fill out the form
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