"""
DeepfakeBench EfficientNet-B4 inference wrapper.

This matches the checkpoint structure used in:
https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1
asset: effnb4_best.pth
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None


EFFNB4_IMG_SIZE = 256
EFFNB4_MEAN = [0.5, 0.5, 0.5]
EFFNB4_STD = [0.5, 0.5, 0.5]


@dataclass
class EffNB4LoadReport:
    loaded_ratio: float
    missing_count: int
    unexpected_count: int


class DeepfakeBenchEfficientNetB4(nn.Module):
    def __init__(self, num_classes: int = 2, in_chans: int = 3):
        super().__init__()
        if EfficientNet is None:
            raise RuntimeError(
                "efficientnet_pytorch is required for EfficientNet-B4 "
                "(pip install efficientnet_pytorch)."
            )

        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        # DeepfakeBench redefines stem explicitly.
        self.efficientnet._conv_stem = nn.Conv2d(
            in_chans, 48, kernel_size=3, stride=2, bias=False
        )
        # Classifier head is external in DeepfakeBench.
        self.efficientnet._fc = nn.Identity()
        self.last_layer = nn.Linear(1792, num_classes)

    def forward(self, x):
        feat = self.efficientnet.extract_features(x)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(feat.size(0), -1)
        return self.last_layer(feat)


def _clean_state_dict(raw_state: Dict[str, torch.Tensor]):
    state = raw_state.get("state_dict", raw_state)
    cleaned = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        cleaned[key] = v
    return cleaned


def load_dfb_efficientnet_b4_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, EffNB4LoadReport]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cleaned = _clean_state_dict(raw)

    model = DeepfakeBenchEfficientNetB4(num_classes=2, in_chans=3)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    total = max(1, len(model.state_dict()))
    loaded = total - len(missing)
    loaded_ratio = loaded / total

    report = EffNB4LoadReport(
        loaded_ratio=loaded_ratio,
        missing_count=len(missing),
        unexpected_count=len(unexpected),
    )

    model = model.to(device)
    model.eval()
    return model, report

