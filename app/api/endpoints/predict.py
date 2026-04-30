import io
import base64
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from app.models.model_loader import load_model, predict
from app.utils.preprocessing import preprocess_image
from app.core.config import CLASS_NAMES

router = APIRouter()
model = load_model()

PREDICTION_TYPE = "1D"


COLON_CLASS_INDICES = {0, 1}     
HIGH_RISK_CLASS_INDEX = 0     

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/jpg"}


def encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 PNG data URI."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_uncertainty_score(probs: list) -> float:
    """
    Entropy-based uncertainty score normalized to [0, 1].
    Computed only over the two colon classes so lung outputs don't distort it.
    0.0 = fully confident, 1.0 = maximum uncertainty.
    """
    colon_probs = np.array([probs[i] for i in sorted(COLON_CLASS_INDICES)], dtype=float)
    colon_probs /= colon_probs.sum()    
    colon_probs = np.clip(colon_probs, 1e-9, 1.0)
    entropy = -np.sum(colon_probs * np.log(colon_probs))
    max_entropy = np.log(len(colon_probs))
    return round(float(entropy / max_entropy), 4)


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        processed = preprocess_image(image)

        class_id, confidence, all_probs = predict(model, processed)

        pred_label = CLASS_NAMES[class_id]
        uncertainty_score = compute_uncertainty_score(all_probs)
        is_high_risk = class_id == HIGH_RISK_CLASS_INDEX

        original_image_b64 = encode_image_base64(image)

        colon_probabilities = {
            CLASS_NAMES[i]: round(float(all_probs[i]), 4)
            for i in sorted(COLON_CLASS_INDICES)
        }

        result = {
            "status": "success",
            "type": PREDICTION_TYPE,
            "prediction": pred_label,
            "class_index": class_id,
            "confidence": round(confidence, 4),
            "diagnostics": {
                "uncertainty_score": uncertainty_score,
                "is_high_risk": is_high_risk,
                "all_probabilities": colon_probabilities,
            },
            "original_image": original_image_b64,
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "healthy"}