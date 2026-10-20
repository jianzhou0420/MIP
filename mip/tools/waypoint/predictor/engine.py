"""The predictor's forward pass: a 12-view RGB-D panorama -> at most five candidate waypoints, each an
angle and a distance — SmartWay's waypoint predictor (arXiv 2503.10069, github.com/sxyxs/SmartWay-Code).

    RGB    each view -> DINOv2 ViT-S/14-reg (nets/dinov2.py) -> 384-d     (preprocessed as SmartWay does:
           shortest edge 256 bicubic, centre crop 224, ImageNet mean / std)
    depth  each view (256 x 256) -> DD-PPO ResNet-50 (nets/depth.py) -> 128 x 4 x 4
    both   -> SmartWay's cross-attention + transformer (nets/waypoint.py) -> 120 angles x 12 distances
           -> softmax over the heatmap, wrap-around NMS -> <= 5 peaks

The checkpoints (INSTALL.md § 6) are checked against their sha256 before they are loaded.
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
import torch
from PIL import Image

from nets.depth import DepthEncoder
from nets.dinov2 import DinoV2
from nets.waypoint import NUM_ANGLES, NUM_DISTANCES, NUM_IMGS, WaypointPredictor, nms

log = logging.getLogger("waypoint_predictor")

CHECKPOINTS = {  # file name under the checkpoint directory -> sha256
    "best.pth": "839440d984b3ad4f26f8f5c9ab7373cf8c7b639e50aa8b9296424819d2be7db0",  # SmartWay's waypoint network
    "gibson-2plus-resnet50.pth": "a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad",  # DD-PPO depth
    "dinov2_vits14_reg4_pretrain.pth": "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",  # DINOv2
}

# DINOv2's preprocessing (the facebook/dinov2-small image-processor config SmartWay calls)
RESIZE_SHORTEST = 256
CROP = 224
RESCALE = 0.00392156862745098  # 1 / 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAX_CANDIDATES = 5
NMS_SIGMA = (7.0, 5.0)
DISTANCE_STEP = 0.25  # heatmap distance j = (j + 1) * 0.25 m


def check_checkpoint(path: Path, expected: str) -> None:
    """Refuse a checkpoint that is missing or is not the released file."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError(f"{path}: sha256 {h.hexdigest()}, expected {expected}")


def decode_views(views: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``[{rgb_base64, depth_base64}, ...]`` -> the RGB images (H, W, 3) uint8 and the depth images
    (H, W) float32 in [0, 1] (8-bit depth is divided by 255), in the order given. A view that does
    not decode is skipped with a warning."""
    rgb: list[np.ndarray] = []
    depth: list[np.ndarray] = []
    for v in views:
        if not isinstance(v, dict):
            continue
        if v.get("rgb_base64"):
            try:
                img = Image.open(io.BytesIO(base64.b64decode(v["rgb_base64"]))).convert("RGB")
                rgb.append(np.asarray(img, dtype=np.uint8))
            except Exception as exc:
                log.warning("RGB view does not decode: %s", exc)
        if v.get("depth_base64"):
            try:
                arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(v["depth_base64"]))), dtype=np.float32)
                depth.append(arr / 255.0 if arr.max() > 1.0 else arr)
            except Exception as exc:
                log.warning("depth view does not decode: %s", exc)
    return rgb, depth


def dinov2_input(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, 224, 224) float32, as the image processor's PIL path computes it."""
    h, w = img.shape[:2]
    short, long = (h, w) if h <= w else (w, h)
    new_short, new_long = RESIZE_SHORTEST, int(RESIZE_SHORTEST * long / short)
    new_h, new_w = (new_short, new_long) if h <= w else (new_long, new_short)
    resized = np.array(Image.fromarray(img).resize((new_w, new_h), resample=Image.BICUBIC))
    top, left = (new_h - CROP) // 2, (new_w - CROP) // 2
    cropped = resized[top : top + CROP, left : left + CROP]
    x = (cropped.astype(np.float64) * RESCALE).astype(np.float32)
    # contiguous CHW, as the image processor hands it over: a strided batch makes cuDNN pick another
    # convolution algorithm for the patch embedding, and the features drift in the last bits
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1))


def clockwise(views: list, fill) -> list:
    """The env's views (view a at a * 30 degrees counter-clockwise) in SmartWay's clockwise slots:
    view a -> slot (12 - a) % 12; empty slots get ``fill``."""
    slots = [None] * NUM_IMGS
    for a, v in enumerate(views[:NUM_IMGS]):
        slots[(NUM_IMGS - a) % NUM_IMGS] = v
    return [fill if s is None else s for s in slots]


class WaypointEngine:
    def __init__(self, ckpt_dir: Path) -> None:
        for name, sha in CHECKPOINTS.items():
            check_checkpoint(ckpt_dir / name, sha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

        self.rgb_encoder = DinoV2()
        self.rgb_encoder.load_weights(ckpt_dir / "dinov2_vits14_reg4_pretrain.pth")
        self.depth_encoder = DepthEncoder()
        self.depth_encoder.load_ddppo(ckpt_dir / "gibson-2plus-resnet50.pth")
        self.predictor = WaypointPredictor()
        state = torch.load(ckpt_dir / "best.pth", map_location="cpu", weights_only=False)
        self.predictor.load_state(state["predictor"]["state_dict"])
        for net in (self.rgb_encoder, self.depth_encoder, self.predictor):
            net.to(self.device).eval().requires_grad_(False)
        log.info("waypoint predictor ready on %s", self.device)

    def predict(self, rgb_views: list[np.ndarray], depth_views: list[np.ndarray]) -> dict[int, dict]:
        """``{k: {"angle", "distance"}}`` for k = 0..K-1 (K <= 5), in ascending angle-index order;
        ``angle`` in counter-clockwise radians (2 pi - i / 120 * 2 pi for heatmap angle i, the
        convention of the env's ``step_hightolow``), ``distance`` in metres. A missing view is
        filled with the first one present."""
        if not rgb_views:
            return {}
        with self._lock, torch.no_grad():
            pixels = torch.from_numpy(np.stack([dinov2_input(v) for v in rgb_views[:NUM_IMGS]])).to(self.device)
            features = list(self.rgb_encoder(pixels))  # per view, in the env's order
            rgb_embedding = torch.stack(clockwise(features, features[0]))
            no_depth = np.zeros_like(rgb_views[0][:, :, 0], dtype=np.float32)
            depth_cw = clockwise(depth_views[: len(rgb_views)], depth_views[0] if depth_views else no_depth)
            depth = torch.from_numpy(np.stack(depth_cw)).unsqueeze(-1).to(self.device).float()
            logits = self.predictor(rgb_embedding, self.depth_encoder(depth))  # (1, 120, 12)

            probs = torch.softmax(logits.reshape(1, NUM_ANGLES * NUM_DISTANCES), dim=1).reshape(
                1, NUM_ANGLES, NUM_DISTANCES
            )
            wrapped = torch.cat((probs[:, -1:, :], probs, probs[:, :1, :]), dim=1)
            peaks = nms(wrapped.unsqueeze(1), max_predictions=MAX_CANDIDATES, sigma=NMS_SIGMA)
            nonzero = peaks.squeeze(1)[0, 1:-1, :].nonzero()
        if nonzero.numel() == 0:
            return {}
        angle_idx = nonzero[:, 0].cpu().numpy()
        dist_idx = nonzero[:, 1].cpu().numpy()
        angles = (2 * math.pi) - angle_idx.astype(np.float32) / NUM_ANGLES * (2 * math.pi)
        distances = (dist_idx.astype(np.float32) + 1) * DISTANCE_STEP
        return {k: {"angle": float(angles[k]), "distance": float(distances[k])} for k in range(len(angle_idx))}
