# zoom-frame-server/model_inference.py
import time
import numpy as np

# Try to use a real model (ResNet18) via torchvision.
# If torch or torchvision is missing, we fall back to a dummy scorer.
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# Global model + device
_model = None
_model_device = "cpu"
_transform = None


def _init_model():
    """
    Initialise a pretrained ResNet18 once and keep it in memory.
    """
    global _model, _model_device, _transform

    if not _TORCH_AVAILABLE:
        return

    if _model is not None:
        return

    # Pick device
    _model_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained ResNet18
    # This will download weights the first time it runs.
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)

    # Replace the final FC layer with a single logit output
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, 1)

    resnet = resnet.to(_model_device)
    resnet.eval()
    _model = resnet

    # Standard ImageNet preprocessing
    _transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) [0,255] -> (C, H, W) [0,1]
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=weights.meta["mean"],
            std=weights.meta["std"]
        ),
    ])

    print("[DF] Loaded ResNet18 pretrained model for inference.")


def _predict_with_dummy(frame_np: np.ndarray) -> float:
    """
    Very simple placeholder "score":
      - Compute mean intensity and normalise to [0,1].
    """
    mean_intensity = float(frame_np.mean())
    return max(0.0, min(1.0, mean_intensity / 255.0))


def _predict_with_model(frame_np: np.ndarray) -> float:
    """
    Run the frame through the ResNet18 binary head.

    Args:
        frame_np: grayscale uint8 array (H, W)
    Returns:
        float score in [0, 1]
    """
    assert frame_np.ndim == 2, "Expected grayscale frame (H, W)"

    # Convert (H, W) -> (H, W, 3) by stacking the grayscale into 3 channels
    h, w = frame_np.shape
    frame_rgb = np.stack([frame_np] * 3, axis=-1)  # (H, W, 3), uint8

    # To PIL-like tensor + resize + normalise
    img = frame_rgb.astype("uint8")
    img = _transform(img)  # (3, 224, 224)

    # Add batch dim
    img = img.unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        tensor = img.to(_model_device)
        logits = _model(tensor)  # (1, 1)
        prob = torch.sigmoid(logits).squeeze()
        score = float(prob.clamp(0.0, 1.0).item())
    return score


def predict_deepfake(frame_np: np.ndarray) -> dict:
    """
    Live deepfake prediction hook.

    Args:
        frame_np: numpy array image.
                  For now this will be grayscale (H, W) uint8.

    Returns:
        dict with at least:
        {
            "score": float in [0, 1], where 1 = very likely deepfake,
            "latency_ms": float,
            "mode": "dummy" | "resnet"
        }
    """
    start = time.time()

    # Initialise model on first call (if possible)
    _init_model()

    if _TORCH_AVAILABLE and _model is not None:
        score = _predict_with_model(frame_np)
        mode = "resnet"
    else:
        score = _predict_with_dummy(frame_np)
        mode = "dummy"

    latency_ms = (time.time() - start) * 1000.0
    return {
        "score": float(score),
        "latency_ms": float(latency_ms),
        "mode": mode,
    }