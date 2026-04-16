from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io

from app.models.model_loader import load_model, predict
from app.utils.preprocessing import preprocess_image
from app.core.config import CLASS_NAMES, ORGAN_NAME

router = APIRouter()
model = load_model()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prétraitement
        processed = preprocess_image(image)
        
        # Prédiction
        class_id, confidence = predict(model, processed)
        
        # Formater la réponse
        result = {
            "organe_detecte": ORGAN_NAME,
            "diagnostic": CLASS_NAMES[class_id],
            "confiance": f"{confidence * 100:.2f}%",
            "confidence_score": confidence,
            "class_id": class_id,
            "statut": "MALIN" if class_id == 1 else "BENIN"
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "healthy"}