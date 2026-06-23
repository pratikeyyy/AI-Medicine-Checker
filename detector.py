"""
detector.py
-----------
YOLOv8 object detection + ByteTrack multi-object tracking wrapper.
ByteTrack is built into the Ultralytics library — no extra install needed.
"""

import torch
from ultralytics import YOLO


class ObjectDetector:
    """
    Wraps YOLOv8 with ByteTrack for persistent multi-object tracking.

    Usage:
        detector = ObjectDetector()
        detections = detector.track(frame)   # call per frame
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.30,
        iou: float = 0.50,
        device: str = None,
    ):
        """
        Args:
            weights : YOLOv8 model file (auto-downloaded on first run).
            conf    : Minimum detection confidence (0–1).
            iou     : IoU threshold for Non-Maximum Suppression.
            device  : 'cuda', 'cpu', or None for auto-detect.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Detector] Loading {weights} on {device} …")
        self.model = YOLO(weights)
        self.model.fuse()
        self.model.to(device)

        self.conf   = conf
        self.iou    = iou
        self.device = device
        self.names  = self.model.names          # {0: 'person', 1: 'bicycle', …}

        print(f"[Detector] Ready — {len(self.names)} classes detected.")

    # ------------------------------------------------------------------
    def track(self, frame) -> list[dict]:
        """
        Run detection + ByteTrack on a single BGR frame.

        Returns:
            List of dicts:
                bbox       : [x1, y1, x2, y2]  (int pixel coords)
                conf       : float confidence (0–1)
                class_id   : int class index
                class_name : str  class label
                track_id   : int  persistent ID (-1 if unassigned)
        """
        results = self.model.track(
            source=frame,
            persist=True,   
            tracker="bytetrack.yaml",
            conf=self.conf,
            iou=self.iou,
            imgsz=640,
            device=self.device,
            half=True,
            stream=False,
            agnostic_nms=True,
            verbose=False,
        )

        detections: list[dict] = []

        if not results or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes

        # Track IDs may be None on the very first frame before tracks form
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)

        bboxes     = boxes.xyxy.cpu().numpy().astype(int)
        confs      = boxes.conf.cpu().numpy()
        class_ids  = boxes.cls.int().cpu().tolist()

        for bbox, conf, class_id, track_id in zip(bboxes, confs, class_ids, track_ids):
            detections.append(
                {
                    "bbox":       bbox,                    # np.array [x1,y1,x2,y2]
                    "conf":       float(conf),
                    "class_id":   int(class_id),
                    "class_name": self.names[class_id],
                    "track_id":   int(track_id),
                }
            )

        return detections
