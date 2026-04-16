import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Union

def preprocess_image(image: Union[Image.Image, np.ndarray], target_size: tuple = (224, 224)) -> torch.Tensor:
    """
    Prétraite une image pour l'inférence du modèle
    
    Args:
        image: Image PIL ou array numpy
        target_size: Taille de redimensionnement (hauteur, largeur)
    
    Returns:
        Tensor normalisé prêt pour le modèle [1, 3, H, W]
    """
    
    # Convertir numpy array en PIL Image si nécessaire
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Pipeline de transformation
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Moyennes ImageNet
            std=[0.229, 0.224, 0.225]     # Écarts-types ImageNet
        )
    ])
    
    # Appliquer les transformations
    image_tensor = transform(image)
    
    # Ajouter dimension batch [C, H, W] -> [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def preprocess_image_for_vit(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    """
    Prétraitement spécifique pour Vision Transformer (ViT)
    À utiliser si votre modèle est un ViT
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def postprocess_prediction(prediction: torch.Tensor, class_names: list = None) -> dict:
    """
    Convertit la sortie du modèle en résultat exploitable
    
    Args:
        prediction: Sortie du modèle (logits ou probabilités)
        class_names: Liste des noms de classes
    
    Returns:
        Dictionnaire avec la classe prédite et les scores
    """
    if class_names is None:
        class_names = ['Colon', 'Lung']  # Ajustez selon votre modèle
    
    # Appliquer softmax si logits
    if len(prediction.shape) > 1:
        probabilities = torch.nn.functional.softmax(prediction, dim=1)
    else:
        probabilities = torch.nn.functional.softmax(prediction, dim=0)
    
    # Obtenir la classe prédite
    predicted_class_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_idx].item()
    
    return {
        'predicted_class': class_names[predicted_class_idx],
        'confidence': confidence,
        'probabilities': {
            class_names[i]: probabilities[i].item() 
            for i in range(len(class_names))
        }
    }


def validate_image(image: Image.Image) -> bool:
    """
    Valide que l'image est dans un format correct
    """
    try:
        # Vérifier les modes couleur
        if image.mode not in ['RGB', 'L', 'RGBA']:
            return False
        
        # Convertir RGBA en RGB
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        
        # Vérifier les dimensions
        if min(image.size) < 32:
            return False
            
        return True
    
    except Exception:
        return False