import torch
import os
import timm
from torch import nn
from app.core.config import MODEL_PATH, DEVICE

_model = None  # lazy — don't load at import time

def load_model():
    global _model
    if _model is not None:
        return _model

    # EfficientNet-B0: 5.3M params, ~20MB weights vs ViT-Base's 86M / ~330MB
    base = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
    in_features = base.num_features  # 1280 for b0

    model = nn.Sequential(
        base,
        nn.LayerNorm(in_features),
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 5),
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # FP16 on CPU cuts memory by half with minimal accuracy loss
    model = model.half()

    _model = model
    print(f"✅ EfficientNet-B0 loaded on CPU (FP16)")
    return _model


def predict(model, image_tensor):
    with torch.no_grad():
        # Cast input to FP16 to match model
        outputs = model(image_tensor.half())
        probabilities = torch.nn.functional.softmax(outputs.float(), dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        all_probs = probabilities[0].cpu().tolist()
        return predicted_class.item(), confidence.item(), all_probs
