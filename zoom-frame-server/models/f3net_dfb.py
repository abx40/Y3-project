"""
DeepfakeBench F3Net inference wrapper.

Reference checkpoint:
- https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/f3net_best.pth
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


F3NET_IMG_SIZE = 256
F3NET_MEAN = [0.5, 0.5, 0.5]
F3NET_STD = [0.5, 0.5, 0.5]


@dataclass
class F3NetLoadReport:
    loaded_ratio: float
    missing_count: int
    unexpected_count: int


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class XceptionBackbone(nn.Module):
    """
    DeepfakeBench xception backbone variant used by F3Net.
    """

    def __init__(self, num_classes=2, in_chans=12, dropout=0.5, mode="original"):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode

        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        final_channel = 2048
        self.last_linear = nn.Linear(final_channel, num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(final_channel, num_classes),
            )

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def features(self, input_):
        x = self.conv1(input_)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def classifier(self, features):
        x = self.relu(features)
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        return self.last_linear(x)

    def forward(self, input_):
        x = self.features(input_)
        out = self.classifier(x)
        return out


def DCT_mat(size):
    return [
        [
            (np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size))
            * np.cos((j + 0.5) * np.pi * i / size)
            for j in range(size)
        ]
        for i in range(size)
    ]


def generate_filter(start, end, size):
    return [[0.0 if i + j > end or i + j < start else 1.0 for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2.0 * torch.sigmoid(x) - 1.0


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super().__init__()
        self.use_learnable = use_learnable
        self.base = nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, size)),
            requires_grad=False,
        )
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0.0, 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                requires_grad=False,
            )

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            return x * filt / self.ft_num
        return x * filt


class FAD_Head(nn.Module):
    def __init__(self, size):
        super().__init__()
        dct = torch.tensor(DCT_mat(size)).float()
        self._DCT_all = nn.Parameter(dct, requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(dct, 0, 1), requires_grad=False)

        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)
        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)
            y = self._DCT_all_T @ x_pass @ self._DCT_all
            y_list.append(y)
        return torch.cat(y_list, dim=1)


class F3NetDFB(nn.Module):
    def __init__(self, img_size=F3NET_IMG_SIZE):
        super().__init__()
        self.FAD_head = FAD_Head(img_size)
        self.backbone = XceptionBackbone(num_classes=2, in_chans=12, dropout=0.5, mode="original")

    def forward(self, x):
        fea_fad = self.FAD_head(x)
        features = self.backbone.features(fea_fad)
        logits = self.backbone.classifier(features)
        return logits


def _clean_state_dict(raw_state: Dict[str, torch.Tensor]):
    state = raw_state.get("state_dict", raw_state)
    cleaned = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = v
    return cleaned


def load_dfb_f3net_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, F3NetLoadReport]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cleaned = _clean_state_dict(raw)

    model = F3NetDFB(img_size=F3NET_IMG_SIZE)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    total = max(1, len(model.state_dict()))
    loaded = total - len(missing)
    loaded_ratio = loaded / total

    report = F3NetLoadReport(
        loaded_ratio=loaded_ratio,
        missing_count=len(missing),
        unexpected_count=len(unexpected),
    )

    model = model.to(device)
    model.eval()
    return model, report

