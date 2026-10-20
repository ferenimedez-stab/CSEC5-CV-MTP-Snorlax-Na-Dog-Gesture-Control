"""YOLO gesture detector running the Colab-exported ONNX model with onnxruntime.

PyTorch cannot load on this laptop (blocked by Windows Application Control), so
the trained YOLO11n model is exported to ONNX on Colab and decoded here.

YOLO11 ONNX output: (1, 4 + num_classes, N) where rows 0-3 are box cx, cy, w, h
in letterboxed-input pixels and the remaining rows are per-class scores.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from vision.preprocess import enhance_low_light, letterbox, to_tensor  # noqa: E402


@dataclass
class Detection:
    class_id: int
    label: str
    conf: float
    box: tuple  # x1, y1, x2, y2 in original-frame pixels

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


class GestureDetector:
    def __init__(self, model_path=C.MODEL_PATH, conf=C.CONF_THRESHOLD, iou=C.IOU_THRESHOLD,
                 low_light=True):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Export best.onnx from the Colab notebook into models/.")
        opts = ort.SessionOptions()
        # One thread per physical core: hyper-threads made inference ~2x slower on the i3-1115G4.
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = self.session.get_inputs()[0].shape[2] or C.IMG_SIZE
        self.conf, self.iou, self.low_light = conf, iou, low_light
        self.last_enhanced = False
        self.last_model_input = None   # letterboxed BGR image actually fed to the model (for display)
        self.last_class_scores = np.zeros(len(C.CLASSES), np.float32)

    def __call__(self, frame):
        self.last_enhanced = False
        if self.low_light:
            frame, self.last_enhanced = enhance_low_light(frame)
        boxed, scale, (pad_x, pad_y) = letterbox(frame, self.size)
        self.last_model_input = boxed
        tensor = to_tensor(boxed)
        out = self.session.run(None, {self.input_name: tensor})[0][0].T  # (N, 4 + nc)

        scores = out[:, 4:]
        self.last_class_scores = scores.max(0)   # best score per class anywhere in the frame (for display)
        class_ids = scores.argmax(1)
        confs = scores[np.arange(len(scores)), class_ids]
        keep = confs >= self.conf
        if not keep.any():
            return []
        boxes, confs, class_ids = out[keep, :4], confs[keep], class_ids[keep]

        # cx, cy, w, h (letterboxed) -> x, y, w, h (original frame)
        xywh = np.empty_like(boxes)
        xywh[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale
        xywh[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale
        xywh[:, 2] = boxes[:, 2] / scale
        xywh[:, 3] = boxes[:, 3] / scale

        # Class-agnostic NMS: one hand should get one label.
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(), self.conf, self.iou)
        h, w = frame.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            box = (int(max(0, x)), int(max(0, y)), int(min(w - 1, x + bw)), int(min(h - 1, y + bh)))
            cid = int(class_ids[i])
            dets.append(Detection(cid, C.CLASSES[cid], float(confs[i]), box))
        return sorted(dets, key=lambda d: d.conf, reverse=True)


# Pastel class colors (BGR): coral, mint, sky, lavender, peach
PALETTE = [(128, 128, 240), (165, 194, 102), (230, 168, 110), (220, 130, 160), (80, 170, 240)]
LABEL_TEXT = (58, 44, 74)   # dark plum, readable on the pastel label backgrounds


def draw_detections(frame, dets):
    """Draw bounding box, class label, confidence and center point for each detection."""
    for d in dets:
        color = PALETTE[d.class_id % len(PALETTE)]
        x1, y1, x2, y2 = d.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{C.DISPLAY_NAMES[d.label]} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + 8)
        lx = max(0, min(x1, frame.shape[1] - tw - 6))   # keep the label inside the image
        cv2.rectangle(frame, (lx, ty - th - 8), (lx + tw + 6, ty), color, -1)
        cv2.putText(frame, text, (lx + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_TEXT, 2)
        cv2.circle(frame, d.center, 4, color, -1)
    return frame
