import os
import torch

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_vit_lc25000.pth")
IMAGE_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

extra = "ignore"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    'Colon Adenocarcinoma',
    'Colon Benign',
    'Lung Adenocarcinoma',
    'Lung Benign',
    'Lung Squamous Cell'
] 
ORGAN_NAME = "COLON"

# Configuration API
API_TITLE = "Colon Cancer Detection API"
API_VERSION = "1.0.0"