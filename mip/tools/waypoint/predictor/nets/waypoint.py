# Copyright (c) 2026 Xiangyu Shi (SmartWay), MIT (LICENSE-smartway). BERT layers: Copyright 2018 The Google AI
# Language Team Authors and The HuggingFace Inc. team, Apache-2.0 (LICENSE-dinov2); WaypointBert: Copyright (c)
# 2020 Microsoft Corporation, MIT, modified in Recurrent VLN-BERT (Yicong Hong).
"""SmartWay's waypoint network (github.com/sxyxs/SmartWay-Code @ daa2dd8, MIT): 12 view tokens -> a
120-angle x 12-distance heatmap, and the non-maximum suppression that picks the candidates.

The inference path of upstream's ``BinaryDistPredictor_TRM`` in its ``cross_attn`` mode, op for op:
each view's RGB embedding queries the depth embeddings of its own and its two neighbouring views
(``ID_CrossAttention``), a two-layer BERT encoder with the same neighbour mask relates the 12 tokens
(``WaypointBert``, after Recurrent VLN-BERT, MIT, on HuggingFace pytorch-transformers' BERT layers,
Apache-2.0), and a classifier gives 10 angles x 12 distances per view. Attribute names follow
upstream so ``best.pth`` loads by key; upstream's training-only parts (dropout, unused heads) are
left out — see ``load_state``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

NUM_IMGS = 12
NUM_ANGLES = 120
NUM_DISTANCES = 12
HEATMAP_OFFSET = 5  # each view's 10 angle bins are centred on the view direction
HIDDEN = 768
RGB_DIM = 384
HEADS = 12
BERT_LAYERS = 2
INTERMEDIATE = 3072
LAYER_NORM_EPS = 1e-12

# heads of upstream's module that the cross_attn forward never calls; their weights are in best.pth
UNUSED_PREFIXES = ("visual_fc_rgb.", "visual_merge.", "attn.", "ffn.")


def neighbour_mask(num_imgs: int = NUM_IMGS, neighbor: int = 1) -> torch.Tensor:
    """(1, 1, n, n) long: 1 where view j is within ``neighbor`` views of view i (wrapping around)."""
    mask = np.zeros((num_imgs, num_imgs))
    t = np.zeros(num_imgs)
    t[: neighbor + 1] = np.ones(neighbor + 1)
    if neighbor != 0:
        t[-neighbor:] = np.ones(neighbor)
    for ri in range(num_imgs):
        mask[ri] = t
        t = np.roll(t, 1)
    return torch.from_numpy(mask).reshape(1, 1, num_imgs, num_imgs).long()


class BertLayerNorm(nn.Module):
    """TF-style layer norm (epsilon inside the square root), as pytorch-transformers 1.x without apex."""

    def __init__(self, hidden_size: int, eps: float = LAYER_NORM_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def gelu(x: torch.Tensor) -> torch.Tensor:
    """The erf form of GELU (pytorch-transformers' ``gelu``)."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_size = HIDDEN // HEADS
        self.query = nn.Linear(HIDDEN, HIDDEN)
        self.key = nn.Linear(HIDDEN, HIDDEN)
        self.value = nn.Linear(HIDDEN, HIDDEN)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.size()[:-1], HEADS, self.head_size).permute(0, 2, 1, 3)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q = self._heads(self.query(hidden))
        k = self._heads(self.key(hidden))
        v = self._heads(self.value(hidden))
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_size)
        scores = scores + attention_mask
        probs = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        return context.view(*context.size()[:-2], HIDDEN)


class BertResidual(nn.Module):
    """dense -> + input -> layer norm (BertSelfOutput / BertOutput)."""

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.dense = nn.Linear(d_in, HIDDEN)
        self.LayerNorm = BertLayerNorm(HIDDEN)

    def forward(self, hidden: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden) + input_tensor)


class BertAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertResidual(HIDDEN)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.output(self.self(hidden, attention_mask), hidden)


class BertIntermediate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(HIDDEN, INTERMEDIATE)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return gelu(self.dense(hidden))


class BertLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertResidual(INTERMEDIATE)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attended = self.attention(hidden, attention_mask)
        return self.output(self.intermediate(attended), attended)


class BertEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayer() for _ in range(BERT_LAYERS)])

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden = layer(hidden, attention_mask)
        return hidden


class BertImgModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = BertEncoder()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        extended = attention_mask.to(dtype=next(self.parameters()).dtype)
        extended = (1.0 - extended) * -10000.0
        return self.encoder(x, extended)


class WaypointBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertImgModel()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.bert(x, attention_mask)


class ID_CrossAttention(nn.Module):  # noqa: N801 — upstream's name, kept for the checkpoint keys
    """Each view's image feature (query) attends to the depth features (key, value) of itself and
    its two neighbours; add & norm, feed-forward, add & norm."""

    def __init__(self) -> None:
        super().__init__()
        self.query_linear = nn.Linear(RGB_DIM, HIDDEN)
        self.key_linear = nn.Linear(HIDDEN, HIDDEN)
        self.value_linear = nn.Linear(HIDDEN, HIDDEN)
        self.multihead_attn = nn.MultiheadAttention(HIDDEN, HEADS)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.ffn = nn.Sequential(nn.Linear(HIDDEN, HIDDEN * 4), nn.ReLU(), nn.Linear(HIDDEN * 4, HIDDEN))
        self.norm2 = nn.LayerNorm(HIDDEN)

    def forward(self, img_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        n = NUM_IMGS
        attn_mask = torch.zeros(n, n).to(img_feat.device)
        for i in range(n):
            attn_mask[i, (i - 1) % n] = 1
            attn_mask[i, i] = 1
            attn_mask[i, (i + 1) % n] = 1
        attn_mask = ~attn_mask.bool()  # True = may not attend

        query = self.query_linear(img_feat).permute(1, 0, 2)
        key = self.key_linear(depth_feat).permute(1, 0, 2)
        value = self.value_linear(depth_feat).permute(1, 0, 2)
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        attn_output = self.norm1(attn_output.permute(1, 0, 2) + depth_feat)
        return self.norm2(self.ffn(attn_output) + attn_output)


class WaypointPredictor(nn.Module):
    """(12, 384) RGB embeddings + (12, 128, 4, 4) depth embeddings, in clockwise view order ->
    (1, 120, 12) heatmap logits (angle bin 0 = the first view's direction)."""

    def __init__(self) -> None:
        super().__init__()
        self.visual_fc_depth = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, HIDDEN), nn.ReLU(True))
        self.pre_fuse = ID_CrossAttention()
        self.waypoint_TRM = WaypointBert()  # noqa: N815 — upstream's name, kept for the checkpoint keys
        self.vis_classifier = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, NUM_DISTANCES * (NUM_ANGLES // NUM_IMGS)),
        )
        self.register_buffer("mask", neighbour_mask(), persistent=False)

    def forward(self, rgb_feats: torch.Tensor, depth_feats: torch.Tensor) -> torch.Tensor:
        b = rgb_feats.size(0) // NUM_IMGS
        rgb_x = rgb_feats.reshape(b, NUM_IMGS, -1)
        depth_x = self.visual_fc_depth(depth_feats).reshape(b, NUM_IMGS, -1)
        vis_x = self.pre_fuse(rgb_x, depth_x)
        vis_rel_x = self.waypoint_TRM(vis_x, self.mask.repeat(b, 1, 1, 1))
        logits = self.vis_classifier(vis_rel_x).reshape(b, NUM_ANGLES, NUM_DISTANCES)
        return torch.cat((logits[:, HEATMAP_OFFSET:, :], logits[:, :HEATMAP_OFFSET, :]), dim=1)

    def load_state(self, state: dict) -> None:
        """Load best.pth's weights: every key of this module must be there, and the only extra keys
        allowed are the unused heads'."""
        kept = {k: v for k, v in state.items() if not k.startswith(UNUSED_PREFIXES)}
        missing, unexpected = self.load_state_dict(kept, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"waypoint checkpoint: missing {missing}, unexpected {unexpected}")


def neighborhoods(mu: torch.Tensor, x_range: int, y_range: int, sigma: tuple[float, float]) -> torch.Tensor:
    """(N, y_range, x_range) 0/1 boxes of half-size ``sigma`` around each ``mu`` = (x, y), x wrapping."""
    x_mu = mu[:, 0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:, 1].unsqueeze(1).unsqueeze(1)
    x = torch.arange(start=0, end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0, end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)
    y_diff = y - y_mu
    x_diff = torch.min(torch.abs(x - x_mu), torch.abs(x - x_mu + x_range))
    return torch.logical_and(torch.abs(x_diff) <= sigma[0], torch.abs(y_diff) <= sigma[1]).type(mu.dtype)


def nms(pred: torch.Tensor, max_predictions: int, sigma: tuple[float, float]) -> torch.Tensor:
    """(B, 1, H, W) -> the same shape with only ``max_predictions`` peaks kept, each suppressing a box
    of half-size ``sigma`` around it. As upstream, the peak's row is ``ix / W`` with true division (a
    fractional row under current torch), which is what the box is centred on."""
    shape = pred.shape
    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0], -1))
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0], -1))
    for _ in range(max_predictions):
        flat_supp_pred = supp_pred.reshape((shape[0], -1))
        _, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0, shape[0])
        flat_output[indices, ix] = flat_pred[indices, ix]
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x, y], dim=1).float()
        g = neighborhoods(mu, shape[-1], shape[-2], sigma)
        supp_pred *= 1 - g.unsqueeze(1)
    output[output < 0] = 0
    return output
