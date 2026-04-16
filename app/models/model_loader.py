import torch
import os
import timm
from torch import nn

from app.core.config import MODEL_PATH, DEVICE

model = None

def load_model():
    global model
    if model is None:
        # Créer le modèle ViT avec timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        
        # Remplacer la tête par une tête à 3 couches (car le modèle sauvegardé a head.0, head.2, head.5)
        hidden_dim = 768

        model.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), 
            nn.Dropout(0.3),         
            nn.Linear(hidden_dim, 512),
            nn.GELU(),            
            nn.Dropout(0.2),        
            nn.Linear(512, 5)
        )
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        # Charger les poids
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Enlever 'module.' si présent
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Charger avec strict=False pour ignorer les différences
        model.load_state_dict(state_dict, strict=True)
        model = model.to(DEVICE)
        model.eval()
        
        print(f"✅ Model loaded successfully on {DEVICE}")
    return model

def predict(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        return predicted_class.item(), confidence.item()