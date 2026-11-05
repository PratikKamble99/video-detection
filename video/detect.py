"""P1 — Object detection (ARCHITECTURE.md §5).

Correctness first (plan.md Phase 1): this starts with eager PyTorch YOLO11 to get
boxes on screen. The TensorRT FP16 export is a Phase 6 optimization and sits
behind the same ``.detect()`` interface, so swapping the backend never touches a
caller — that interface stability is the whole point (ARCHITECTURE.md §14).

The detector is *stateless per frame*: it returns raw detections with no
identity. Turning that flickering per-frame output into stable object identities
is the tracker's job (``video.track``), and it is not optional — a downstream
reasoner fed raw detections hallucinates events ("a knife appeared and vanished
6 times").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection in one frame. No identity — that comes from the tracker.

    ``bbox`` is normalized xyxy in [0, 1], so everything downstream is
    resolution-independent.
    """

    cls: str
    conf: float
    bbox: tuple[float, float, float, float]   # xyxy, normalized


class Detector:
    """YOLO11 wrapper with a stable ``detect(frame_bgr) -> list[Detection]``.

    ``model`` may be a bundled weight name (``yolo11n.pt`` … ``yolo11x.pt``,
    downloaded on first use) or a local path. ``classes`` optionally restricts
    output to a subset of COCO class ids. Open-vocab (YOLO-World) is a later
    swap behind this same interface.
    """

    def __init__(self, model: str = "yolo11n.pt", *, conf: float = 0.35,
                 imgsz: int = 640, device: str = "cpu",
                 classes: list[int] | None = None):
        # Imported lazily so Phase 0 / headless CI never pays the torch import.
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.classes = classes
        self.names: dict[int, str] = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run detection on one BGR frame. Pure inference, no tracking state."""
        res = self.model.predict(
            frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )[0]

        h, w = frame_bgr.shape[:2]
        out: list[Detection] = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            out.append(Detection(
                cls=self.names.get(int(k), str(int(k))),
                conf=float(c),
                bbox=(float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h),
            ))
        return out

    def set_classes(self, names: list[str]) -> None:
        """Open-vocabulary prompt update (YOLO-World models only). Precomputes
        and caches the text embeddings so per-frame cost stays a single matmul
        (ARCHITECTURE.md §7.1)."""
        if hasattr(self.model, "set_classes"):
            self.model.set_classes(names)
            self.names = {i: n for i, n in enumerate(names)}
        else:
            raise NotImplementedError(
                f"{self.model.model_name} is not open-vocabulary; use a "
                "YOLO-World checkpoint to set classes at runtime"
            )
