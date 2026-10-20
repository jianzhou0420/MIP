"""The predictor's forward pass: a 12-view RGB-D panorama -> at most five candidate waypoints, each an
angle and a distance.

The network is SmartWay's (smartway/): a DD-PPO ResNet-50 encodes each depth view, SmartWay's
image-depth cross-attention and TRM transformer turn the 12 views into a 120-angle x 12-distance
heatmap, and non-maximum suppression keeps the five strongest peaks.

The RGB input of the network (a DINOv2 embedding per view) is all zeros: the paper's Waypoint, Hybrid
and Hybrid + SLAM runs called the predictor with the views only, and SmartWay's predictor then fills
the RGB embedding with zeros. This module reproduces exactly that, so the candidates depend on depth
alone.

Both checkpoints are checked against the sha256 of the files the paper used before they are loaded.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("waypoint_predictor")

N_VIEWS = 12
NUM_ANGLES = 120
NUM_DISTANCES = 12
RGB_DIM = 384  # the DINOv2 ViT-S/14 embedding width the network was trained with

# The two checkpoints (INSTALL.md § 6 says where to download them).
WAYPOINT_SHA256 = "839440d984b3ad4f26f8f5c9ab7373cf8c7b639e50aa8b9296424819d2be7db0"  # best.pth
DDPPO_SHA256 = "a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad"  # gibson-2plus-resnet50.pth


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_checkpoint(path: Path, expected: str) -> None:
    """Refuse a checkpoint that is missing or is not the file the paper used."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    got = _sha256(path)
    if got != expected:
        raise ValueError(f"{path}: sha256 {got}, expected {expected}")


def _decode_rgb(b64: str) -> np.ndarray:
    """A base64 image -> (H, W, 3) uint8."""
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _decode_depth(b64: str) -> np.ndarray:
    """A base64 depth image -> (H, W) float32 in [0, 1] (8-bit images are divided by 255)."""
    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))), dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def decode_views(views: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``[{rgb_base64, depth_base64}, ...]`` -> the RGB and the depth images, in the order given.
    A view that does not decode is skipped (with a warning)."""
    rgb: list[np.ndarray] = []
    depth: list[np.ndarray] = []
    for v in views:
        if not isinstance(v, dict):
            continue
        if v.get("rgb_base64"):
            try:
                rgb.append(_decode_rgb(v["rgb_base64"]))
            except Exception as exc:
                log.warning("RGB view does not decode: %s", exc)
        if v.get("depth_base64"):
            try:
                depth.append(_decode_depth(v["depth_base64"]))
            except Exception as exc:
                log.warning("depth view does not decode: %s", exc)
    return rgb, depth


class WaypointEngine:
    """The loaded network. Needs a CUDA GPU (SmartWay's forward moves the model to ``cuda``)."""

    def __init__(self, waypoint_ckpt: Path, ddppo_ckpt: Path) -> None:
        import torch
        from gym import spaces

        from smartway.depth.resnet_encoders import VlnResnetDepthEncoder
        from smartway.waypoint_predictor.TRM_net import BinaryDistPredictor_TRM

        check_checkpoint(waypoint_ckpt, WAYPOINT_SHA256)
        check_checkpoint(ddppo_ckpt, DDPPO_SHA256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        predictor = BinaryDistPredictor_TRM(hidden_dim=768, n_classes=NUM_DISTANCES).to(self.device)
        state = torch.load(waypoint_ckpt, map_location=self.device, weights_only=False)
        # best.pth nests the weights as ckpt["predictor"]["state_dict"]
        while isinstance(state, dict) and not any(isinstance(v, torch.Tensor) for v in state.values()):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "predictor" in state:
                state = state["predictor"]
            else:
                break
        missing, unexpected = predictor.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{waypoint_ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        for p in predictor.parameters():
            p.requires_grad = False
        predictor.eval()
        self.predictor = predictor

        # the depth encoder takes 256 x 256 single-channel depth, as in SmartWay's VLN-CE task config
        obs_space = spaces.Dict(
            {"depth": spaces.Box(low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32)}
        )
        depth_net = VlnResnetDepthEncoder(
            observation_space=obs_space,
            output_size=128,
            checkpoint=str(ddppo_ckpt),
            backbone="resnet50",
            trainable=False,
        ).to(self.device)
        depth_net.eval()
        self.depth_encoder = depth_net
        log.info("waypoint predictor ready on %s", self.device)

    def predict(self, rgb_views: list[np.ndarray], depth_views: list[np.ndarray]) -> dict[int, dict]:
        """``{k: {"angle", "distance"}}`` for k = 0..K-1 (K <= 5), in ascending angle-index order.

        The views come in the env's order (view a at a * 30 degrees counter-clockwise); SmartWay reads
        them clockwise, so view a goes to slot (12 - a) % 12. ``angle`` is counter-clockwise radians
        (2 pi - i / 120 * 2 pi for heatmap angle i), the convention of the env's ``step_hightolow``;
        ``distance`` is (j + 1) * 0.25 m for heatmap distance j. A missing view is filled with the
        first one present.
        """
        import torch

        from smartway.waypoint_predictor.utils import nms

        if not rgb_views:
            return {}

        n = min(N_VIEWS, len(rgb_views))
        depth_cw: list[np.ndarray | None] = [None] * N_VIEWS
        for a in range(n):
            if a < len(depth_views):
                depth_cw[(N_VIEWS - a) % N_VIEWS] = depth_views[a]
        first_depth = next(
            (v for v in depth_cw if v is not None), np.zeros_like(rgb_views[0][:, :, 0], dtype=np.float32)
        )
        depth_batch = np.stack([first_depth if v is None else v for v in depth_cw], axis=0)  # (12, H, W)

        with self._lock, torch.no_grad():
            depth_tensor = torch.from_numpy(depth_batch).unsqueeze(-1).to(self.device).float()
            depth_embedding = self.depth_encoder({"depth": depth_tensor})
            rgb_embedding = torch.zeros(N_VIEWS, RGB_DIM, device=self.device)
            logits = self.predictor(
                rgb_embedding, depth_embedding, None, pre_fuse=False, cross_attn=True
            )  # (1, 120, 12)

            # softmax over the whole heatmap, then nms with the angle axis wrapped around
            probs = torch.softmax(logits.reshape(1, NUM_ANGLES * NUM_DISTANCES), dim=1).reshape(
                1, NUM_ANGLES, NUM_DISTANCES
            )
            wrapped = torch.cat((probs[:, -1:, :], probs, probs[:, :1, :]), dim=1)
            peaks = nms(wrapped.unsqueeze(1), max_predictions=5, sigma=(7.0, 5.0))
            peaks = peaks.squeeze(1)[:, 1:-1, :]  # (1, 120, 12)
            nonzero = peaks[0].nonzero()

        if nonzero.numel() == 0:
            return {}
        angle_idx = nonzero[:, 0].cpu().numpy()
        dist_idx = nonzero[:, 1].cpu().numpy()
        angles = (2 * math.pi) - angle_idx.astype(np.float32) / NUM_ANGLES * (2 * math.pi)
        distances = (dist_idx.astype(np.float32) + 1) * 0.25
        return {
            k: {"angle": float(angles[k]), "distance": float(distances[k])} for k in range(len(angle_idx))
        }
