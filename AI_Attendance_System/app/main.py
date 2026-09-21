
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import io

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(BASE_DIR.parent / "model" / "attendance_model.keras")
CLASS_NAMES_PATH = str(BASE_DIR.parent / "model" / "class_names.json")

IMG_SIZE = (160, 160)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# ------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------

app = FastAPI(
    title="AI Attendance Monitoring System",
    description="Scalable AI-powered face recognition attendance API",
    version="1.0.0"
)

# In-memory attendance records
attendance_records = []

# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "AI Attendance Monitoring System",
        "status": "running",
        "version": "1.0.0"
    }

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": len(class_names)
    }

# ------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Read uploaded image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Resize
        image = image.resize(IMG_SIZE)

        # Convert to numpy
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # MobileNetV2 preprocessing
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        # Prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            predictions[0][predicted_index]
        )

        person_name = class_names[predicted_index]

        # Attendance record
        record = {
            "person": person_name,
            "status": "Present",
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().isoformat()
        }

        attendance_records.append(record)

        return JSONResponse({
            "success": True,
            "prediction": person_name,
            "confidence": round(confidence, 4),
            "attendance": "Present",
            "timestamp": record["timestamp"]
        })

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

# ------------------------------------------------------------
# Attendance records
# ------------------------------------------------------------

@app.get("/attendance")
def get_attendance():

    return {
        "total_records": len(attendance_records),
        "records": attendance_records
    }

# ------------------------------------------------------------
# Clear attendance records
# ------------------------------------------------------------

@app.delete("/attendance")
def clear_attendance():

    attendance_records.clear()

    return {
        "success": True,
        "message": "Attendance records cleared"
    }
