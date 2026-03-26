"""
Effort CLIP-L14 inference wrapper (modular backend use).

This module supports loading checkpoints from:
- Effort (ICML 2025) style state dicts with SVD-residual self-attn linear layers.
- Plain CLIP-L14 DF40/DeepfakeBench style state dicts (no SVD-residual keys).

Expected checkpoint keys typically include:
- module.backbone.*
- module.head.*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionConfig, CLIPVisionModel
except Exception:
    CLIPVisionConfig = None
    CLIPVisionModel = None


EFFORT_IMG_SIZE = 224
EFFORT_MEAN = [0.48145466, 0.4578275, 0.40821073]
EFFORT_STD = [0.26862954, 0.26130258, 0.27577711]

# Effort uses ViT-L/14 hidden size (1024) and keeps top r=1023 singular components.
EFFORT_HIDDEN_DIM = 1024
EFFORT_SVD_R = 1023


@dataclass
class EffortLoadReport:
    mode: str
    loaded_ratio: float
    missing_count: int
    unexpected_count: int
    head_weight_in_checkpoint: bool
    head_bias_in_checkpoint: bool
    head_weight_loaded: bool
    head_bias_loaded: bool


class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        self.weight_main = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False,
        )
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=2 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.S_residual = None
        self.U_residual = None
        self.V_residual = None

    def forward(self, x):
        if (
            hasattr(self, "U_residual")
            and hasattr(self, "V_residual")
            and self.S_residual is not None
        ):
            residual_weight = (
                self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            )
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)


def _replace_with_svd_residual(module: nn.Module, r: int):
    if not isinstance(module, nn.Linear):
        return module

    new_module = SVDResidualLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        r=r,
        bias=(module.bias is not None),
        init_weight=module.weight.data.clone(),
    )

    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)

    # Same decomposition strategy as Effort implementation.
    U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
    keep_r = min(r, len(S))
    U_r = U[:, :keep_r]
    S_r = S[:keep_r]
    Vh_r = Vh[:keep_r, :]

    weight_main = U_r @ torch.diag(S_r) @ Vh_r
    new_module.weight_main.data.copy_(weight_main)

    U_residual = U[:, keep_r:]
    S_residual = S[keep_r:]
    Vh_residual = Vh[keep_r:, :]

    if len(S_residual) > 0:
        new_module.S_residual = nn.Parameter(S_residual.clone())
        new_module.U_residual = nn.Parameter(U_residual.clone())
        new_module.V_residual = nn.Parameter(Vh_residual.clone())
        new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
        new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
        new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)

    return new_module


def apply_svd_residual_to_self_attn(model: nn.Module, r: int):
    for name, module in model.named_children():
        if "self_attn" in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent = module
                    parts = sub_name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], _replace_with_svd_residual(sub_module, r))
        else:
            apply_svd_residual_to_self_attn(module, r)
    return model


class EffortClipL14(nn.Module):
    def __init__(self, use_svd_residual: bool):
        super().__init__()
        if CLIPVisionConfig is None or CLIPVisionModel is None:
            raise RuntimeError(
                "transformers is required for Effort CLIP-L14 "
                "(pip install transformers)."
            )

        cfg = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
        )

        # Effort detector builds from CLIPModel.vision_model, whose keys are
        # "backbone.embeddings.*", "backbone.encoder.*", ...
        backbone = CLIPVisionModel(cfg).vision_model
        if use_svd_residual:
            backbone = apply_svd_residual_to_self_attn(backbone, r=EFFORT_SVD_R)
        self.backbone = backbone
        self.head = nn.Linear(EFFORT_HIDDEN_DIM, 2)

    def forward(self, x):
        outputs = self.backbone(x)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            feat = outputs[1]
        else:
            feat = outputs[0][:, 0]
        return self.head(feat)


def _clean_state_dict(raw_state: Dict[str, torch.Tensor]):
    state = raw_state.get("state_dict", raw_state)
    cleaned = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("backbone.vision_model."):
            key = "backbone." + key[len("backbone.vision_model."):]
        if key.startswith("vision_model."):
            key = "backbone." + key[len("vision_model."):]
        if key.startswith(("embeddings.", "encoder.", "pre_layrnorm.", "post_layernorm.")):
            key = "backbone." + key
        cleaned[key] = v
    return cleaned


def _has_effort_svd_keys(cleaned_state: Dict[str, torch.Tensor]):
    return any(
        ("weight_main" in k) or ("S_residual" in k) or ("U_residual" in k) or ("V_residual" in k)
        for k in cleaned_state.keys()
    )


def load_effort_clip_l14_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, EffortLoadReport]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cleaned = _clean_state_dict(raw)
    use_svd = _has_effort_svd_keys(cleaned)
    head_weight_in_checkpoint = "head.weight" in cleaned
    head_bias_in_checkpoint = "head.bias" in cleaned

    model = EffortClipL14(use_svd_residual=use_svd)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    missing_set = set(missing)

    total = max(1, len(model.state_dict()))
    loaded = total - len(missing)
    loaded_ratio = loaded / total

    report = EffortLoadReport(
        mode=("effort_svd" if use_svd else "plain_clip"),
        loaded_ratio=loaded_ratio,
        missing_count=len(missing),
        unexpected_count=len(unexpected),
        head_weight_in_checkpoint=head_weight_in_checkpoint,
        head_bias_in_checkpoint=head_bias_in_checkpoint,
        head_weight_loaded=head_weight_in_checkpoint and ("head.weight" not in missing_set),
        head_bias_loaded=head_bias_in_checkpoint and ("head.bias" not in missing_set),
    )

    model = model.to(device)
    model.eval()
    return model, report
