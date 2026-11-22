"""
VideoMAE temporal detector loader.

Supports two modes:
1) DeepfakeBench-style custom checkpoint on top of VideoMAE backbone.
2) Hugging Face end-to-end VideoMAE classifier model_id (recommended fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVideoClassification,
        VideoMAEModel,
    )
except Exception:
    AutoConfig = None
    AutoImageProcessor = None
    AutoModelForVideoClassification = None
    VideoMAEModel = None


VIDEOMAE_IMG_SIZE = 224
VIDEOMAE_MEAN = [0.5, 0.5, 0.5]
VIDEOMAE_STD = [0.5, 0.5, 0.5]
VIDEOMAE_DEFAULT_CLIP_SIZE = 16
# Public finetuned HF checkpoint with binary fake/real labels.
VIDEOMAE_DEFAULT_MODEL_ID = "Ammar2k/videomae-base-finetuned-deepfake-subset"


@dataclass
class VideoMAELoadReport:
    mode: str
    loaded_ratio: float
    missing_count: int
    unexpected_count: int
    fake_index: int
    img_size: int
    mean: list
    std: list


class VideoMAEDFB(nn.Module):
    """
    Expects clip tensor shape [B, T, C, H, W].
    """

    def __init__(self, model_id: str):
        super().__init__()
        if VideoMAEModel is None:
            raise RuntimeError(
                "transformers with VideoMAEModel is required for videomae mode."
            )
        self.backbone = VideoMAEModel.from_pretrained(model_id)
        hidden = int(self.backbone.config.hidden_size)
        self.fc_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, clip_btchw: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(clip_btchw, output_hidden_states=True)
        sequence_output = outputs[0]
        video_feat = self.fc_norm(sequence_output.mean(1))
        logits = self.head(video_feat)
        return logits


class VideoMAEHFClassifier(nn.Module):
    """
    Wrap HF VideoMAEForVideoClassification to keep a unified forward signature:
      input:  [B, T, C, H, W]
      output: [B, num_labels] logits
    """

    def __init__(self, model_id: str):
        super().__init__()
        if AutoModelForVideoClassification is None:
            raise RuntimeError(
                "transformers AutoModelForVideoClassification is required for videomae mode."
            )
        self.model = AutoModelForVideoClassification.from_pretrained(model_id)

    def forward(self, clip_btchw: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=clip_btchw)
        return out.logits


def _clean_state_dict(raw_state: Dict[str, torch.Tensor]):
    state = raw_state.get("state_dict", raw_state)
    cleaned = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = v
    return cleaned


def _infer_fake_index(cfg) -> int:
    label2id = getattr(cfg, "label2id", None) or {}
    norm = {}
    for k, v in label2id.items():
        try:
            norm[str(k).strip().lower()] = int(v)
        except Exception:
            continue

    for k in ("fake", "deepfake", "forged", "spoof", "manipulated"):
        if k in norm:
            return norm[k]

    num_labels = int(getattr(cfg, "num_labels", 2) or 2)
    if "real" in norm and num_labels == 2:
        return 1 - norm["real"]
    if num_labels == 2:
        return 1
    return 0


def _processor_stats(model_id: str):
    size = VIDEOMAE_IMG_SIZE
    mean = list(VIDEOMAE_MEAN)
    std = list(VIDEOMAE_STD)
    if AutoImageProcessor is None:
        return size, mean, std
    try:
        proc = AutoImageProcessor.from_pretrained(model_id)
        proc_mean = getattr(proc, "image_mean", None)
        proc_std = getattr(proc, "image_std", None)
        if isinstance(proc_mean, (list, tuple)) and len(proc_mean) >= 3:
            mean = [float(proc_mean[0]), float(proc_mean[1]), float(proc_mean[2])]
        if isinstance(proc_std, (list, tuple)) and len(proc_std) >= 3:
            std = [float(proc_std[0]), float(proc_std[1]), float(proc_std[2])]
        proc_size = getattr(proc, "size", None)
        if isinstance(proc_size, dict):
            if "height" in proc_size and "width" in proc_size:
                size = int(min(proc_size["height"], proc_size["width"]))
            elif "shortest_edge" in proc_size:
                size = int(proc_size["shortest_edge"])
        elif isinstance(proc_size, int):
            size = int(proc_size)
    except Exception:
        pass
    return size, mean, std


def load_dfb_videomae_model(
    model_id: str,
    checkpoint_path: str | None,
    device: torch.device,
) -> Tuple[nn.Module, VideoMAELoadReport]:
    if checkpoint_path:
        model = VideoMAEDFB(model_id=model_id)
        raw = torch.load(checkpoint_path, map_location="cpu")
        cleaned = _clean_state_dict(raw)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        total = max(1, len(model.state_dict()))
        loaded = total - len(missing)
        loaded_ratio = loaded / total
        report = VideoMAELoadReport(
            mode="deepfake_ckpt",
            loaded_ratio=loaded_ratio,
            missing_count=len(missing),
            unexpected_count=len(unexpected),
            fake_index=1,
            img_size=VIDEOMAE_IMG_SIZE,
            mean=list(VIDEOMAE_MEAN),
            std=list(VIDEOMAE_STD),
        )
    else:
        if AutoConfig is None:
            raise RuntimeError("transformers AutoConfig is required for videomae mode.")
        cfg = AutoConfig.from_pretrained(model_id)
        model = VideoMAEHFClassifier(model_id=model_id)
        fake_index = _infer_fake_index(cfg)
        img_size, mean, std = _processor_stats(model_id)
        report = VideoMAELoadReport(
            mode="hf_classifier",
            loaded_ratio=1.0,
            missing_count=0,
            unexpected_count=0,
            fake_index=fake_index,
            img_size=img_size,
            mean=mean,
            std=std,
        )

    model = model.to(device)
    model.eval()
    return model, report
