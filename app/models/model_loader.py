import torch
import os
import timm
import gc
from torch import nn
from app.core.config import MODEL_PATH, DEVICE

_model = None

def _log_mem(stage: str):
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mb = proc.memory_info().rss / 1024 / 1024
        print(f"[MEM] {stage}: {mb:.1f} MB RSS")
    except ImportError:
        pass  # psutil optional

def load_model():
    global _model
    if _model is not None:
        return _model

    _log_mem("before model load")

    if not os.path.exists(MODEL_PATH):
        # Log clearly instead of crashing — lets /health still respond
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print(f"Files in model dir: {os.listdir(os.path.dirname(MODEL_PATH))}")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    print(f"Loading EfficientNet-B0 from {MODEL_PATH} ...")

    base = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
    in_features = base.num_features  # 1280

    model = nn.Sequential(
        base,
        nn.LayerNorm(in_features),
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 5),
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    # Strip DataParallel prefix if trained with multi-GPU
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.half()  # FP16 — halves RAM

    # Free anything torch cached during load
    gc.collect()
    torch.set_num_threads(1)  # Render free tier = 1 vCPU

    _model = model
    _log_mem("after model load")
    print("Model ready.")
    return _model


def predict(model, image_tensor):
    with torch.no_grad():
        out = model(image_tensor.half())
        probs = torch.nn.functional.softmax(out.float(), dim=1)
        conf, cls = torch.max(probs, 1)
        return cls.item(), conf.item(), probs[0].cpu().tolist()
