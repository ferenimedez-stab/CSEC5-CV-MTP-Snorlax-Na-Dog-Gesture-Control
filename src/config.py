"""Shared configuration for the local (laptop) side of the Gesture Control system.

Dataset download, splitting, training and evaluation run on Google Colab
(src/training/train_colab.ipynb) and use the same class order and
preprocessing values as defined here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Class order defines YOLO class ids (0..4). Must match the notebook.
CLASSES = ["fist", "like", "palm", "rock", "call"]

DISPLAY_NAMES = {
    "fist": "Fist",
    "like": "Thumbs Up",
    "palm": "Open Palm",
    "rock": "Rock Sign",
    "call": "Call Sign",
}
# Robot command each gesture will drive in the Final project.
COMMANDS = {
    "fist": "STOP",
    "like": "FORWARD",
    "palm": "BACKWARD",
    "rock": "LEFT",
    "call": "RIGHT",
}

# ---- Paths -------------------------------------------------------------------
WEBCAM_DIR = ROOT / "dataset" / "webcam"         # self-collected images: <class>/*.jpg
MODEL_PATH = ROOT / "models" / "best.onnx"       # exported from Colab
RESULTS_DIR = ROOT / "results" / "midterm"
SCREENSHOT_DIR = RESULTS_DIR / "screenshots"

# ---- Preprocessing -----------------------------------------------------------
IMG_SIZE = 640                # YOLO input size (letterboxed square)
CLAHE_CLIP = 2.0              # CLAHE clip limit (low-light enhancement)
CLAHE_GRID = (8, 8)
LOW_LIGHT_MEAN = 70           # mean L-channel below this => apply CLAHE
BLUR_THRESHOLD = 60.0         # variance of Laplacian below this => blurry (capture warning)

# ---- Inference ---------------------------------------------------------------
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
SMOOTH_WINDOW = 5             # frames used for majority-vote label smoothing
CAMERA_INDEX = 0
