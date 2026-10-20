"""Gesture Control demo app (Midterm: SEE -> RECOGNIZE).

One window with two input modes and an optional pipeline view:
  LIVE   laptop webcam; detection runs in a background thread so the video stays smooth,
         and the gesture is smoothed with a 5-result majority vote
  IMAGE  any images or folder (e.g. instructor-provided), stepped with the arrow keys
  p      pipeline view: Input -> Preprocessed model input (640x640) -> Detection output

Hover the (i) icon (or press h) to see all keyboard shortcuts.

Usage:
    python src/vision/demo_app.py [--camera 0] [--model models/best.onnx] [--images <file-or-folder>]
"""
import argparse
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from vision.detector import PALETTE, GestureDetector, draw_detections  # noqa: E402

WINDOW = "Gesture Control Demo - SnorlaxTheDog"
W, H = 1280, 720
BANNER_H, SIDEBAR_W = 100, 330
VIDEO = (0, BANNER_H, W - SIDEBAR_W, H - BANNER_H)          # x, y, w, h of the video area
# Pastel pink theme (BGR)
BG = (231, 225, 248)        # base pastel pink
PANEL = (242, 238, 252)     # banner / sidebar, lighter pink
PANEL2 = (220, 208, 244)    # highlighted row, deeper pink
VIDEO_BG = (216, 205, 240)  # behind the video / image panels
LINE = (203, 190, 230)      # dividers, empty bars
TEXT = (58, 44, 74)         # dark plum
MUTED = (125, 110, 150)     # muted plum
ACCENT = (131, 93, 222)     # rose
WARN = (40, 100, 200)       # deep orange for notes
CHIP_TEXT = (255, 255, 255)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEY_LEFT, KEY_RIGHT = {2424832, 65361, ord("a")}, {2555904, 65363, ord("d")}
FONT = cv2.FONT_HERSHEY_SIMPLEX
SHORTCUTS = [("p", "Pipeline view on / off"), ("o", "Open image(s)"), ("f", "Open a folder of images"),
             ("c", "Back to live camera"), ("<- ->", "Previous / next image"), ("[  ]", "Threshold - / + 0.05"),
             ("s", "Save screenshot"),
             ("m", "Mirror on / off"), ("l", "Low-light enhancement on / off"), ("h", "Pin this help"),
             ("q", "Quit")]


# ------------------------------------------------------------------ drawing helpers
def text(img, s, org, scale=0.6, color=TEXT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_w(s, scale=0.6, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def text_right(img, s, right_x, y, scale=0.6, color=TEXT, thick=1):
    text(img, s, (right_x - text_w(s, scale, thick), y), scale, color, thick)


def blend_rect(img, p1, p2, color, alpha):
    x1, y1 = max(p1[0], 0), max(p1[1], 0)
    x2, y2 = min(p2[0], img.shape[1]), min(p2[1], img.shape[0])
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def pill(img, s, x, y, fg, bg, scale=0.5):
    """Rounded label; (x, y) is the top-left. Returns its width."""
    tw = text_w(s, scale, 1)
    w, h = tw + 20, 24
    cv2.rectangle(img, (x + 6, y), (x + w - 6, y + h), bg, -1, cv2.LINE_AA)
    cv2.circle(img, (x + 6, y + h // 2), h // 2, bg, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w - 6, y + h // 2), h // 2, bg, -1, cv2.LINE_AA)
    text(img, s, (x + 10, y + 17), scale, fg, 1)
    return w


def fit_into(canvas, img, x, y, w, h, fill=PANEL):
    """Draw img inside the (x, y, w, h) region, keeping aspect ratio. Returns the placed rect."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), fill, -1)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return ox, oy, nw, nh


def limit_size(img, max_side=1280):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def pick_files(folder=False):
    """Native file dialog (tkinter). Returns a list of image paths."""
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if folder:
        d = filedialog.askdirectory(title="Select a folder of test images")
        files = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS) if d else []
    else:
        sel = filedialog.askopenfilenames(title="Select test image(s)",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        files = [Path(p) for p in sel]
    root.destroy()
    return files


def collect_images(path):
    p = Path(path)
    return [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.suffix.lower() in EXTS)


# ------------------------------------------------------------------ detection
@dataclass
class Result:
    frame: np.ndarray                 # frame the detection was computed on
    dets: list
    model_input: np.ndarray           # letterboxed 640x640 image fed to the model
    enhanced: bool                    # CLAHE applied
    ms: float                         # inference time
    scores: np.ndarray = field(default_factory=lambda: np.zeros(len(C.CLASSES)))
    seq: int = 0


def run_detection(det, lock, frame):
    with lock:
        t = time.perf_counter()
        dets = det(frame)
        ms = (time.perf_counter() - t) * 1000
        return Result(frame, dets, det.last_model_input, det.last_enhanced, ms, det.last_class_scores.copy())


class AsyncDetector(threading.Thread):
    """Runs the detector on the newest camera frame in the background; older frames are skipped."""

    def __init__(self, det, lock):
        super().__init__(daemon=True)
        self.det, self.lock = det, lock
        self.cv = threading.Condition()
        self.pending, self.result, self.running, self.seq = None, None, True, 0

    def submit(self, frame):
        with self.cv:
            self.pending = frame
            self.cv.notify()

    def run(self):
        while self.running:
            with self.cv:
                while self.pending is None and self.running:
                    self.cv.wait(0.1)
                frame, self.pending = self.pending, None
            if frame is None:
                continue
            res = run_detection(self.det, self.lock, frame)
            self.seq += 1
            res.seq = self.seq
            self.result = res

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify()


# ------------------------------------------------------------------ app
class DemoApp:
    def __init__(self, args):
        self.det = GestureDetector(args.model, conf=args.conf)
        self.lock = threading.Lock()
        self.worker = AsyncDetector(self.det, self.lock)
        self.worker.start()
        self.camera_index = args.camera
        self.cap = None
        self.mode = "LIVE"
        self.pipeline = False
        self.mirror = True
        self.images, self.idx = [], 0
        self.history = deque(maxlen=C.SMOOTH_WINDOW)
        self.last_seq = 0
        self.cam_fps = self.det_rate = 0.0
        self.prev_frame_t = self.prev_det_t = time.perf_counter()
        self.message, self.message_until = "", 0.0
        self.mouse = (-1, -1)
        self.help_pinned = False
        self.slider = None          # (x1, x2, y) of the threshold slider track, set while drawing
        self.dragging = False
        self.image_cache = None
        if args.images:
            self.load_images(collect_images(args.images))
        else:
            self.open_camera()

    # ---------------------------------------------------------------- input
    def open_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if self.cap.isOpened():
            self.mode = "LIVE"
            self.history.clear()
            self.worker.result = None
        else:
            self.flash("Camera not available - press o to open images")
            self.mode = "IMAGE"

    def load_images(self, files):
        if not files:
            self.flash("No images selected")
            return
        self.images, self.idx, self.mode = list(files), 0, "IMAGE"
        self.image_cache = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def flash(self, msg, seconds=3.0):
        self.message, self.message_until = msg, time.time() + seconds

    def next_input(self):
        """Return (frame, source_label) for the current mode."""
        if self.mode == "LIVE":
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if not ok:
                return None, "Camera read failed"
            if self.mirror:   # selfie view; all 5 gestures look the same mirrored
                frame = cv2.flip(frame, 1)
            return frame, f"Camera {self.camera_index}"
        if not self.images:
            return None, "No images loaded"
        path = self.images[self.idx]
        frame = cv2.imread(str(path))
        if frame is None:
            return None, f"Cannot read {path.name}"
        return limit_size(frame), path.name

    # ---------------------------------------------------------------- decision
    def smoothed_label(self, res):
        """Live: majority vote over the last results. Image: top detection."""
        if self.mode == "IMAGE":
            return res.dets[0].label if res and res.dets else None
        if not self.history:
            return None
        label, votes = Counter(self.history).most_common(1)[0]
        return label if votes > len(self.history) // 2 else None

    # ---------------------------------------------------------------- drawing
    def draw_banner(self, cv_, label, res, source):
        cv2.rectangle(cv_, (0, 0), (W, BANNER_H), PANEL, -1)
        cv2.line(cv_, (0, BANNER_H - 1), (W, BANNER_H - 1), LINE, 1)
        color = PALETTE[C.CLASSES.index(label) % len(PALETTE)] if label else ACCENT
        cv2.rectangle(cv_, (0, 0), (8, BANNER_H), color, -1)

        # Gesture and action: same size, values aligned in one column
        scale, lx, vx = 1.0, 28, 190
        conf = next((d.conf for d in res.dets if d.label == label), 0.0) if (res and label) else 0.0
        gesture = f"{C.DISPLAY_NAMES[label]}  ({conf:.0%})" if label else "-"
        action = C.COMMANDS[label] if label else "-"
        text(cv_, "Gesture", (lx, 44), scale, MUTED, 1)
        text(cv_, gesture, (vx, 44), scale, ACCENT if label else MUTED, 2)
        text(cv_, "Action", (lx, 84), scale, MUTED, 1)
        text(cv_, action, (vx, 84), scale, TEXT if label else MUTED, 2)

        # Mode chips + source, right-aligned
        x = W - 24
        chips = [(self.mode, CHIP_TEXT, ACCENT if self.mode == "LIVE" else PALETTE[3])]
        if self.pipeline:
            chips.append(("PIPELINE", CHIP_TEXT, PALETTE[2]))
        for s, fg, bg in reversed(chips):
            x -= text_w(s, 0.5) + 20
            pill(cv_, s, x, 22, fg, bg)
            x -= 22   # the pill's rounded ends extend 6 px past its box
        text_right(cv_, source[:48], W - 24, 78, 0.5, MUTED)

    def draw_sidebar(self, cv_, label, res):
        x0, y0 = W - SIDEBAR_W, BANNER_H
        cv2.rectangle(cv_, (x0, y0), (W, H), PANEL, -1)
        cv2.line(cv_, (x0, y0), (x0, H), LINE, 1)
        pad, right = x0 + 20, W - 20

        # --- gesture confidence bars (all classes)
        y = y0 + 34
        text(cv_, "GESTURE CONFIDENCE", (pad, y), 0.5, MUTED, 1)
        scores = res.scores if res is not None else np.zeros(len(C.CLASSES))
        detected = {d.label for d in res.dets} if res else set()
        bar_x, bar_w = pad + 130, right - (pad + 130) - 52
        y += 16
        for i, c in enumerate(C.CLASSES):
            ry = y + i * 38
            active = c in detected
            if c == label:
                cv2.rectangle(cv_, (x0 + 8, ry), (W - 8, ry + 32), PANEL2, -1)
            col = PALETTE[i % len(PALETTE)]
            cv2.rectangle(cv_, (pad, ry + 10), (pad + 12, ry + 22), col if active else LINE, -1)
            text(cv_, C.DISPLAY_NAMES[c], (pad + 22, ry + 22), 0.55, TEXT if active else MUTED, 2 if c == label else 1)
            s = float(scores[i])
            cv2.rectangle(cv_, (bar_x, ry + 12), (bar_x + bar_w, ry + 20), LINE, -1)
            cv2.rectangle(cv_, (bar_x, ry + 12), (bar_x + int(bar_w * min(s, 1)), ry + 20), col if active else MUTED, -1)
            tx = bar_x + int(bar_w * self.det.conf)       # threshold marker
            cv2.line(cv_, (tx, ry + 8), (tx, ry + 24), TEXT, 1)
            text_right(cv_, f"{s:.0%}", right, ry + 22, 0.5, TEXT if active else MUTED)
        y += 5 * 38 + 6
        distinct = len(detected)
        if distinct >= 2:
            text(cv_, f"{distinct} different gestures - using the most confident", (pad, y + 4), 0.45, WARN)
        elif distinct == 0:
            text(cv_, "Show one hand gesture to the camera", (pad, y + 4), 0.45, MUTED)

        # --- confidence threshold slider (drag, or [ / ])
        y += 34
        text(cv_, "Confidence threshold", (pad, y), 0.5, MUTED)
        text_right(cv_, f"{self.det.conf:.2f}", right, y, 0.55, TEXT, 2)
        ty, tx1, tx2 = y + 20, pad + 6, right - 6
        self.slider = (tx1, tx2, ty)
        kx = tx1 + int((tx2 - tx1) * self.det.conf)
        cv2.line(cv_, (tx1, ty), (tx2, ty), LINE, 6, cv2.LINE_AA)
        cv2.line(cv_, (tx1, ty), (kx, ty), ACCENT, 6, cv2.LINE_AA)
        cv2.circle(cv_, (kx, ty), 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(cv_, (kx, ty), 10, ACCENT, 2, cv2.LINE_AA)

        # --- stats: aligned label / value rows
        y = ty + 52
        cv2.line(cv_, (pad, y - 26), (right, y - 26), LINE, 1)
        text(cv_, "LIVE STATS" if self.mode == "LIVE" else "IMAGE STATS", (pad, y), 0.5, MUTED, 1)
        if self.mode == "LIVE":
            rows = [("Camera", f"{self.cam_fps:.1f} FPS"),
                    ("Detection", f"{self.det_rate:.1f} / s"),
                    ("Inference", f"{res.ms:.0f} ms" if res else "-"),
                    ("Smoothing", f"{C.SMOOTH_WINDOW} results"),
                    ("Low-light", ("Active" if res and res.enhanced else "Auto") if self.det.low_light else "Off"),
                    ("Mirror", "On" if self.mirror else "Off")]
        else:
            rows = [("Image", f"{self.idx + 1} / {len(self.images)}" if self.images else "-"),
                    ("Inference", f"{res.ms:.0f} ms" if res else "-"),
                    ("Detections", str(len(res.dets)) if res else "-"),
                    ("Low-light", ("Active" if res and res.enhanced else "Auto") if self.det.low_light else "Off")]
        for i, (k, v) in enumerate(rows):
            ry = y + 28 + i * 25
            text(cv_, k, (pad, ry), 0.55, MUTED)
            text_right(cv_, v, right, ry, 0.55, TEXT)

        # --- info icon (hover for shortcuts)
        cx, cy = W - 34, H - 30
        hover = (cx - 16 <= self.mouse[0] <= cx + 16 and cy - 16 <= self.mouse[1] <= cy + 16)
        cv2.circle(cv_, (cx, cy), 14, ACCENT if (hover or self.help_pinned) else MUTED, 2, cv2.LINE_AA)
        text(cv_, "i", (cx - 4, cy + 7), 0.6, ACCENT if (hover or self.help_pinned) else MUTED, 2)
        text_right(cv_, "Shortcuts", cx - 24, cy + 6, 0.5, MUTED)
        return hover

    def draw_help(self, cv_):
        w, h = 400, 44 + len(SHORTCUTS) * 30
        x, y = W - SIDEBAR_W - w - 16 + (SIDEBAR_W if self.pipeline else 0), H - h - 16
        blend_rect(cv_, (x, y), (x + w, y + h), PANEL, 0.95)
        cv2.rectangle(cv_, (x, y), (x + w, y + h), ACCENT, 1)
        text(cv_, "KEYBOARD SHORTCUTS", (x + 18, y + 30), 0.5, MUTED)
        for i, (k, desc) in enumerate(SHORTCUTS):
            ry = y + 50 + i * 30
            pill(cv_, k, x + 16, ry - 4, TEXT, PANEL2, 0.45)
            text(cv_, desc, (x + 110, ry + 13), 0.5, TEXT)

    def draw_toast(self, cv_):
        if not self.message or time.time() > self.message_until:
            return
        vx, vy, vw, vh = VIDEO if not self.pipeline else (0, BANNER_H, W, H - BANNER_H)
        tw = text_w(self.message, 0.55)
        x, y = vx + (vw - tw) // 2 - 16, vy + vh - 56
        blend_rect(cv_, (x, y), (x + tw + 32, y + 36), PANEL, 0.95)
        cv2.rectangle(cv_, (x, y), (x + tw + 32, y + 36), ACCENT, 1)
        text(cv_, self.message, (x + 16, y + 24), 0.55, TEXT)

    def draw_pipeline(self, cv_, res):
        top, bottom = BANNER_H + 56, H - 64
        n, gap = 3, 44
        pw = (W - 2 * 24 - (n - 1) * gap) // n
        ph = bottom - top
        pre_note = ("CLAHE on, " if res.enhanced else "") + f"letterbox {C.IMG_SIZE}x{C.IMG_SIZE}, RGB, /255"
        panels = [("1. Input", f"{res.frame.shape[1]}x{res.frame.shape[0]} BGR frame", res.frame),
                  ("2. Preprocessing", pre_note, res.model_input),
                  ("3. YOLO11n -> Detection", f"conf >= {self.det.conf:.2f}, NMS IoU {C.IOU_THRESHOLD}",
                   draw_detections(res.frame.copy(), res.dets))]
        for i, (title, note, img) in enumerate(panels):
            x = 24 + i * (pw + gap)
            text(cv_, title, (x, top - 18), 0.65, TEXT, 2)
            fit_into(cv_, img, x, top, pw, ph, fill=VIDEO_BG)
            text(cv_, note, (x, bottom + 28), 0.47, MUTED)
            if i < n - 1:
                ax, ay = x + pw + 8, top + ph // 2
                cv2.arrowedLine(cv_, (ax, ay), (ax + gap - 16, ay), ACCENT, 3, cv2.LINE_AA, tipLength=0.35)

    def render(self, frame, res, label, source):
        canvas = np.full((H, W, 3), BG, np.uint8)
        self.draw_banner(canvas, label, res, source)
        if frame is None:
            vx, vy, vw, vh = VIDEO
            cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), VIDEO_BG, -1)
            text(canvas, source, (vx + 40, vy + vh // 2 - 10), 0.8, TEXT, 2)
            text(canvas, "Press o to open images, f for a folder, or c to retry the camera.",
                 (vx + 40, vy + vh // 2 + 26), 0.55, MUTED)
            hover = self.draw_sidebar(canvas, label, None)
        elif self.pipeline and res is not None:
            self.draw_pipeline(canvas, res)
            # info icon still available in pipeline view
            cx, cy = W - 34, H - 30
            hover = (cx - 16 <= self.mouse[0] <= cx + 16 and cy - 16 <= self.mouse[1] <= cy + 16)
            cv2.circle(canvas, (cx, cy), 14, ACCENT if (hover or self.help_pinned) else MUTED, 2, cv2.LINE_AA)
            text(canvas, "i", (cx - 4, cy + 7), 0.6, ACCENT if (hover or self.help_pinned) else MUTED, 2)
        else:
            # newest camera frame with the newest detection drawn on it
            shown = draw_detections(frame.copy(), res.dets) if res else frame
            fit_into(canvas, shown, *VIDEO, fill=VIDEO_BG)
            hover = self.draw_sidebar(canvas, label, res)
        if hover or self.help_pinned:
            self.draw_help(canvas)
        self.draw_toast(canvas)
        return canvas

    # ---------------------------------------------------------------- events
    def set_conf(self, value):
        self.det.conf = round(min(max(value, 0.05), 0.95), 2)

    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if self.slider is None or self.pipeline:
            return
        x1, x2, ty = self.slider
        if event == cv2.EVENT_LBUTTONDOWN and x1 - 12 <= x <= x2 + 12 and abs(y - ty) <= 14:
            self.dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        if self.dragging and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            self.set_conf((x - x1) / (x2 - x1))

    def handle_key(self, key, canvas):
        k = key & 0xFF
        if k in (ord("q"), 27):
            return False
        if k == ord("p"):
            self.pipeline = not self.pipeline
        elif k == ord("h"):
            self.help_pinned = not self.help_pinned
        elif k == ord("["):
            self.set_conf(self.det.conf - 0.05)
        elif k == ord("]"):
            self.set_conf(self.det.conf + 0.05)
        elif k == ord("m"):
            self.mirror = not self.mirror
        elif k == ord("l"):
            self.det.low_light = not self.det.low_light
            self.flash(f"Low-light enhancement {'on (auto)' if self.det.low_light else 'off'}")
        elif k == ord("c"):
            self.open_camera()
        elif k == ord("o"):
            self.load_images(pick_files(folder=False))
        elif k == ord("f"):
            self.load_images(pick_files(folder=True))
        elif key in KEY_RIGHT and self.mode == "IMAGE" and self.images:
            self.idx = (self.idx + 1) % len(self.images)
        elif key in KEY_LEFT and self.mode == "IMAGE" and self.images:
            self.idx = (self.idx - 1) % len(self.images)
        elif k == ord("s") and canvas is not None:
            name = f"demo_{self.mode.lower()}{'_pipeline' if self.pipeline else ''}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(C.SCREENSHOT_DIR / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.flash(f"Screenshot saved: results/midterm/screenshots/{name}")
        return True

    # ---------------------------------------------------------------- main loop
    def step(self):
        """Produce one canvas for the current mode."""
        frame, source = self.next_input()
        if frame is None:
            return self.render(None, None, None, source)
        if self.mode == "IMAGE":
            key = (self.idx, self.det.conf, self.det.low_light)
            if self.image_cache is None or self.image_cache[0] != key:   # re-run only when something changed
                self.image_cache = (key, run_detection(self.det, self.lock, frame))
            res = self.image_cache[1]
            return self.render(frame, res, self.smoothed_label(res), source)

        self.worker.submit(frame)
        now = time.perf_counter()
        self.cam_fps = 0.9 * self.cam_fps + 0.1 / max(now - self.prev_frame_t, 1e-6)
        self.prev_frame_t = now
        res = self.worker.result
        if res is not None and res.seq != self.last_seq:     # a new detection result arrived
            self.last_seq = res.seq
            self.history.append(res.dets[0].label if res.dets else None)
            self.det_rate = 0.8 * self.det_rate + 0.2 / max(now - self.prev_det_t, 1e-6)
            self.prev_det_t = now
        return self.render(frame, res, self.smoothed_label(res), source)

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        C.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            while True:
                canvas = self.step()
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKeyEx(1 if self.mode == "LIVE" else 30)
                if key == -1:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue
                if not self.handle_key(key, canvas):
                    break
        finally:
            self.worker.stop()
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=C.CAMERA_INDEX)
    ap.add_argument("--model", default=str(C.MODEL_PATH))
    ap.add_argument("--conf", type=float, default=C.CONF_THRESHOLD)
    ap.add_argument("--images", help="start in image mode with this file or folder")
    DemoApp(ap.parse_args()).run()


if __name__ == "__main__":
    main()
