"""Image preprocessing used at capture and inference time on the laptop."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def blur_score(img):
    """Variance of the Laplacian; low values mean a blurry frame."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


_clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_GRID)


def enhance_low_light(img, force=False):
    """Apply CLAHE to the L channel (LAB) when the frame is dark.

    Returns (image, applied_flag).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if not force and l.mean() >= C.LOW_LIGHT_MEAN:
        return img, False
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), True


def letterbox(img, size=C.IMG_SIZE, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to a size x size square
    (same letterbox Ultralytics uses during training).

    Returns (padded_image, scale, (pad_x, pad_y)) so boxes can be mapped back.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, (pad_x, pad_y)


def to_tensor(boxed):
    """Letterboxed BGR image -> (1, 3, H, W) float32 RGB tensor in [0, 1]."""
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)


def to_model_input(img, size=C.IMG_SIZE):
    """BGR frame -> (1, 3, size, size) float32 RGB tensor in [0, 1] plus letterbox info."""
    boxed, scale, pad = letterbox(img, size)
    return to_tensor(boxed), scale, pad
