"""
DeepfakeBench-compatible I3D (3D ResNet) loader.

Checkpoint examples:
- models/train_on_df40/i3d.pth
- DeepfakeBench release v1.0.3: I3D_8x8_R50.pth
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from slowfast.config.defaults import get_cfg
    from slowfast.models.video_model_builder import ResNet as SlowFastResNet
except Exception:
    get_cfg = None
    SlowFastResNet = None


I3D_IMG_SIZE = 224
I3D_MEAN = [0.5, 0.5, 0.5]
I3D_STD = [0.5, 0.5, 0.5]
I3D_DEFAULT_CLIP_SIZE = 16


I3D_CONFIG_TEXT = """
TRAIN:
  ENABLE: True
DATA:
  NUM_FRAMES: 16
  SAMPLING_RATE: 8
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3]
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50
  TRANS_FUNC: bottleneck_transform
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3], [4], [6], [3]]
NONLOCAL:
  LOCATION: [[[]], [[]], [[]], [[]]]
  GROUP: [[1], [1], [1], [1]]
  INSTANTIATION: softmax
BN:
  USE_PRECISE_STATS: True
  NUM_BATCHES_PRECISE: 200
SOLVER:
  BASE_LR: 0.1
  LR_POLICY: cosine
  MAX_EPOCH: 196
  MOMENTUM: 0.9
  WEIGHT_DECAY: 1e-4
  WARMUP_EPOCHS: 34.0
  WARMUP_START_LR: 0.01
  OPTIMIZING_METHOD: sgd
MODEL:
  NUM_CLASSES: 1
  ARCH: i3d
  MODEL_NAME: ResNet
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
  HEAD_ACT: sigmoid
TEST:
  ENABLE: True
  BATCH_SIZE: 64
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
NUM_GPUS: 8
NUM_SHARDS: 1
RNG_SEED: 0
OUTPUT_DIR: .
"""


@dataclass
class I3DLoadReport:
    loaded_ratio: float
    missing_count: int
    unexpected_count: int


class I3DDFB(nn.Module):
    """
    Wrapper accepting clip tensor shape [B, T, C, H, W].
    """

    def __init__(self, clip_size: int = I3D_DEFAULT_CLIP_SIZE):
        super().__init__()
        if get_cfg is None or SlowFastResNet is None:
            raise RuntimeError(
                "I3D requires local slowfast modules + fvcore/yacs dependencies."
            )
        cfg = get_cfg()
        cfg.merge_from_str(I3D_CONFIG_TEXT)
        cfg.NUM_GPUS = 1
        cfg.TRAIN.BATCH_SIZE = 1
        cfg.TEST.BATCH_SIZE = 1
        cfg.DATA.NUM_FRAMES = int(clip_size)
        self.resnet = SlowFastResNet(cfg)

    def forward(self, clip_btchw: torch.Tensor) -> torch.Tensor:
        # slowfast I3D path expects [B, C, T, H, W] wrapped in list.
        x = clip_btchw.permute(0, 2, 1, 3, 4).contiguous()
        y = self.resnet([x])
        return y


def _clean_state_dict(raw_state: Dict[str, torch.Tensor]):
    state = raw_state.get("state_dict", raw_state)
    cleaned = {}
    for k, v in state.items():
        key = k
        if key.startswith("module.resnet."):
            key = key[len("module.resnet."):]
        elif key.startswith("resnet."):
            key = key[len("resnet."):]
        elif key.startswith("module."):
            key = key[len("module."):]

        # Pretrained I3D_8x8_R50 can have 400-way head; trim to binary head.
        if key == "head.projection.weight" and v.ndim == 2 and v.shape[0] > 1:
            v = v[:1, :]
        if key == "head.projection.bias" and v.ndim == 1 and v.shape[0] > 1:
            v = v[:1]
        cleaned[key] = v
    return cleaned


def load_dfb_i3d_model(
    checkpoint_path: str,
    device: torch.device,
    clip_size: int = I3D_DEFAULT_CLIP_SIZE,
) -> Tuple[nn.Module, I3DLoadReport]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cleaned = _clean_state_dict(raw)

    model = I3DDFB(clip_size=clip_size)
    missing, unexpected = model.resnet.load_state_dict(cleaned, strict=False)

    total = max(1, len(model.resnet.state_dict()))
    loaded = total - len(missing)
    loaded_ratio = loaded / total

    report = I3DLoadReport(
        loaded_ratio=loaded_ratio,
        missing_count=len(missing),
        unexpected_count=len(unexpected),
    )

    model = model.to(device)
    model.eval()
    return model, report

