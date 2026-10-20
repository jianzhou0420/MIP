# Copyright (c) Meta Platforms, Inc. and its affiliates (habitat-lab); Copyright (c) 2020 Jacob Krantz et al.
# (VLN-CE). MIT (LICENSE-habitat-lab, LICENSE-vlnce).
"""The DD-PPO depth encoder SmartWay uses: habitat-lab's GroupNorm ResNet-50 ``ResNetEncoder``
(github.com/facebookresearch/habitat-lab, MIT) with the gibson-2plus weights, loaded the way VLN-CE's
``VlnResnetDepthEncoder`` loads them (github.com/jacobkrantz/VLN-CE, MIT). Inference path only; the
attribute names follow habitat-lab's so the checkpoint loads by key.

Input (N, 256, 256, 1) depth in [0, 1] -> (N, 128, 4, 4): 2 x 2 average pooling, ResNet-50 with
32 base planes and 16 groups, then a 3 x 3 compression conv to 128 channels.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DEPTH_SIZE = 256  # the encoder is built for 256 x 256 depth; the compression conv assumes it
BASE_PLANES = 32
NGROUPS = BASE_PLANES // 2
EXPANSION = 4
LAYERS = (3, 4, 6, 3)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            conv1x1(inplanes, planes),
            nn.GroupNorm(NGROUPS, planes),
            nn.ReLU(True),
            conv3x3(planes, planes, stride),
            nn.GroupNorm(NGROUPS, planes),
            nn.ReLU(True),
            conv1x1(planes, planes * EXPANSION),
            nn.GroupNorm(NGROUPS, planes * EXPANSION),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.convs(x) + identity)


class ResNet50(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, BASE_PLANES, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(NGROUPS, BASE_PLANES),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = BASE_PLANES
        self.layer1 = self._make_layer(BASE_PLANES, LAYERS[0])
        self.layer2 = self._make_layer(BASE_PLANES * 2, LAYERS[1], stride=2)
        self.layer3 = self._make_layer(BASE_PLANES * 4, LAYERS[2], stride=2)
        self.layer4 = self._make_layer(BASE_PLANES * 8, LAYERS[3], stride=2)
        self.final_channels = self.inplanes

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * EXPANSION:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * EXPANSION, stride), nn.GroupNorm(NGROUPS, planes * EXPANSION)
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * EXPANSION
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class DepthEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet50(in_channels=1)
        final_spatial = int((DEPTH_SIZE // 2) * (1.0 / 2**5))  # 4
        compression_channels = int(round(2048 / final_spatial**2))  # 128
        self.compression = nn.Sequential(
            nn.Conv2d(self.backbone.final_channels, compression_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, compression_channels),
            nn.ReLU(True),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """(N, 256, 256, 1) depth in [0, 1] -> (N, 128, 4, 4)."""
        if tuple(depth.shape[1:3]) != (DEPTH_SIZE, DEPTH_SIZE):
            raise ValueError(f"depth views must be {DEPTH_SIZE} x {DEPTH_SIZE}, got {tuple(depth.shape[1:3])}")
        x = F.avg_pool2d(depth.permute(0, 3, 1, 2), 2)
        return self.compression(self.backbone(x))

    def load_ddppo(self, checkpoint: Path) -> None:
        """The visual encoder's weights out of a DD-PPO checkpoint (``actor_critic.net.visual_encoder.*``)."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = {}
        for k, v in ckpt["state_dict"].items():
            parts = k.split(".")[2:]
            if parts[0] == "visual_encoder":
                weights[".".join(parts[1:])] = v
        self.load_state_dict(weights, strict=True)
