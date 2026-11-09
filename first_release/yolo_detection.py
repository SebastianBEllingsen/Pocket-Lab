"""
Find blood cells with the trained YOLO model (models/blood_cells.pt).
"""
from pathlib import Path

import cv2
from ultralytics import YOLO

from postprocessing import refine_red_cells

MODEL_PATH = Path(__file__).resolve().parent / "models" / "blood_cells.pt"
REFINE = True      # correct YOLO's red cells with the post-processing (postprocessing.py)
CONFIDENCE = 0.35  # best setting from evaluate.py (lowest RBC count error)

WBC_TYPES = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil"]

COLOURS = {  # BGR
    "rbc": (0, 0, 255),         # red
    "platelet": (255, 0, 255),  # pink
}
WBC_COLOUR = (0, 170, 0)        # green

_model = None


def get_model():
    """Load the model once, the first time it is needed."""
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def detect_cells(img, refine=REFINE):
    """
    Return a list of cells found in a BGR image, e.g.
    [{"type": "rbc", "confidence": 0.91, "box": (x1, y1, x2, y2)}, ...]
    """
    model = get_model()
    # agnostic_nms: one box per cell, even if two classes both match it
    result = model.predict(img, imgsz=352, conf=CONFIDENCE, agnostic_nms=True, verbose=False)[0]

    cells = []
    for box, class_id, confidence in zip(
        result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
    ):
        cells.append({
            "type": model.names[int(class_id)],
            "confidence": confidence,
            "box": tuple(int(v) for v in box),
        })
    if refine:
        cells = refine_red_cells(img, cells)
    return cells


def draw_cells(img, cells, show_red=True, show_white=True, show_platelets=False):
    """Return a copy of img with boxes drawn around the chosen cell types."""
    output = img.copy()

    for cell in cells:
        kind = cell["type"]
        if kind == "rbc" and not show_red:
            continue
        if kind in WBC_TYPES and not show_white:
            continue
        if kind == "platelet" and not show_platelets:
            continue

        x1, y1, x2, y2 = cell["box"]
        colour = COLOURS.get(kind, WBC_COLOUR)
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)

        # Write the type above white cells, so you can see which one it is
        if kind in WBC_TYPES:
            cv2.putText(output, kind, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    return output
