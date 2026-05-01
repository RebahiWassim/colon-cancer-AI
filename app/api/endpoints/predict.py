import io, base64
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from app.models.model_loader import load_model, predict
from app.utils.preprocessing import preprocess_image
from app.core.config import CLASS_NAMES

router = APIRouter()
# ❌ Remove: model = load_model() at module level
# ✅ Load lazily on first request — server starts in <1s

COLON_CLASS_INDICES = {0, 1}
HIGH_RISK_CLASS_INDEX = 0
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/jpg"}

def encode_image_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def compute_uncertainty_score(probs: list) -> float:
    colon_probs = np.array([probs[i] for i in sorted(COLON_CLASS_INDICES)], dtype=float)
    colon_probs /= colon_probs.sum()
    colon_probs = np.clip(colon_probs, 1e-9, 1.0)
    entropy = -np.sum(colon_probs * np.log(colon_probs))
    return round(float(entropy / np.log(len(colon_probs))), 4)

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    try:
        model = load_model()  # lazy — cached after first call
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed = preprocess_image(image)
        class_id, confidence, all_probs = predict(model, processed)
        pred_label = CLASS_NAMES[class_id]
        return JSONResponse(content={
            "status": "success",
            "type": "1D",
            "prediction": pred_label,
            "class_index": class_id,
            "confidence": round(confidence, 4),
            "diagnostics": {
                "uncertainty_score": compute_uncertainty_score(all_probs),
                "is_high_risk": class_id == HIGH_RISK_CLASS_INDEX,
                "all_probabilities": {CLASS_NAMES[i]: round(float(all_probs[i]), 4) for i in sorted(COLON_CLASS_INDICES)},
            },
            "original_image": encode_image_base64(image),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "healthy"}
