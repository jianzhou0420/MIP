# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the Apache License 2.0 (LICENSE-dinov2).
"""DINOv2 ViT-S/14 with 4 registers (``dinov2_vits14_reg``), the RGB encoder SmartWay's predictor was
trained on: github.com/facebookresearch/dinov2, Apache-2.0 (LICENSE-dinov2 here). The inference path
of its ``DinoVisionTransformer`` without xFormers (plain attention), returning the normalised CLS token
as ``torch.hub``'s model does; attribute names follow upstream so the released weights load by key.

Input (N, 3, 224, 224) normalised RGB -> (N, 384).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

PATCH = 14
EMBED_DIM = 384
DEPTH = 12
HEADS = 6
MLP_RATIO = 4
REGISTERS = 4
PRETRAIN_IMG = 518  # the positional embedding is stored for 518 x 518 (37 x 37 patches)
LAYER_SCALE_INIT = 1.0


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH, stride=PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = (EMBED_DIM // HEADS) ** -0.5
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3, bias=True)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, HEADS, C // HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM, EMBED_DIM * MLP_RATIO)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(EMBED_DIM * MLP_RATIO, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(LAYER_SCALE_INIT * torch.ones(EMBED_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.mlp = Mlp()
        self.ls2 = LayerScale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_patches = (PRETRAIN_IMG // PATCH) ** 2
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, EMBED_DIM))
        self.register_tokens = nn.Parameter(torch.zeros(1, REGISTERS, EMBED_DIM))
        self.mask_token = nn.Parameter(torch.zeros(1, EMBED_DIM))  # training only; in the weights
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)

    def _pos_embed(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """The 37 x 37 positional embedding resized (bicubic, antialiased) to the input's patch grid."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        M = int(math.sqrt(N))
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=True,
            size=(w // PATCH, h // PATCH),
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self._pos_embed(x, w, h)
        x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]

    def load_weights(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
