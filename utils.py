"""
utils.py
--------
Drawing utilities (bounding boxes, labels, overlays) and a smooth FPS counter.
"""

import time
import cv2
import numpy as np
from collections import deque


# ── Colour palette ─────────────────────────────────────────────────────────────

_CLASS_COLORS: dict[int, tuple[int, int, int]] = {}


def _make_color(class_id: int) -> tuple[int, int, int]:
    """Generate a vivid, consistent BGR colour for a class ID using golden-angle hue spacing."""
    hue   = int((class_id * 137.508) % 180)      # golden-angle hue (HSV hue 0–179)
    hsv   = np.uint8([[[hue, 210, 230]]])
    bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def get_class_color(class_id: int) -> tuple[int, int, int]:
    """Return a cached BGR colour tuple for the given class ID."""
    if class_id not in _CLASS_COLORS:
        _CLASS_COLORS[class_id] = _make_color(class_id)
    return _CLASS_COLORS[class_id]


# ── Detection drawing ───────────────────────────────────────────────────────────

def draw_detection(frame: np.ndarray, detection: dict) -> None:
    """
    Draw one detection's bounding box + label onto *frame* in-place.

    Label format:  #<track_id>  <class_name>  <conf%>
    """
    h, w      = frame.shape[:2]
    x1, y1, x2, y2 = detection["bbox"]

    # Clamp coords to frame boundaries
    x1 = max(0, int(x1));  y1 = max(0, int(y1))
    x2 = min(w, int(x2));  y2 = min(h, int(y2))

    conf       = detection["conf"]
    class_name = detection["class_name"]
    class_id   = detection["class_id"]
    track_id   = detection["track_id"]
    color      = get_class_color(class_id)

    # ── Bounding box ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ── Build label string ────────────────────────────────────────────────────
    if track_id >= 0:
        label = f"#{track_id}  {class_name}  {conf:.0%}"
    else:
        label = f"{class_name}  {conf:.0%}"

    # ── Measure text ──────────────────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_DUPLEX
    scale      = 0.55
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    PAD = 5

    # Place label above the box; fall inside if too close to top edge
    if y1 - th - PAD * 2 >= 0:
        bg_y1 = y1 - th - PAD * 2
        bg_y2 = y1
        txt_y = y1 - PAD
    else:
        bg_y1 = y1
        bg_y2 = y1 + th + PAD * 2
        txt_y = y1 + th + PAD

    bg_x2 = min(x1 + tw + PAD * 2, w)

    # ── Filled label background ───────────────────────────────────────────────
    cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, -1)

    # ── Label text (white) ────────────────────────────────────────────────────
    cv2.putText(frame, label, (x1 + PAD, txt_y),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ── HUD overlays ────────────────────────────────────────────────────────────────

def _put_text_with_shadow(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float = 0.8,
    thickness: int = 2,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw text with a dark shadow for readability on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    # shadow
    cv2.putText(frame, text, (x + 2, y + 2), font, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # main
    cv2.putText(frame, text, (x, y), font, font_scale,
                color, thickness, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Render live FPS in the top-left corner (green)."""
    _put_text_with_shadow(frame, f"FPS: {fps:5.1f}", (12, 38),
                          font_scale=0.85, thickness=2, color=(0, 230, 0))


def draw_info_bar(
    frame: np.ndarray,
    num_objects: int,
    source_label: str,
    frame_num: int,
) -> None:
    """Render object count + source label below the FPS counter."""
    text = f"Objects: {num_objects:3d}  |  Source: {source_label}  |  Frame: {frame_num}"
    _put_text_with_shadow(frame, text, (12, 70),
                          font_scale=0.55, thickness=1, color=(200, 200, 200))


# ── FPS counter ─────────────────────────────────────────────────────────────────

class FPSCounter:
    """
    Sliding-window FPS counter.

    Usage:
        fps_ctr = FPSCounter(window=30)
        while processing:
            fps_ctr.tick()
            print(fps_ctr.fps)
    """

    def __init__(self, window: int = 30):
        self._window = window
        self._times: deque[float] = deque(maxlen=window)
        self._last: float = time.perf_counter()

    def tick(self) -> None:
        """Call once per processed frame."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now

    @property
    def fps(self) -> float:
        """Current smoothed FPS estimate."""
        if len(self._times) < 2:
            return 0.0
        total = sum(self._times)
        return len(self._times) / total if total > 0 else 0.0
