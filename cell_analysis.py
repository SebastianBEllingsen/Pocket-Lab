"""
Blood-cell analysis for the web app: counts and boxes for red cells, white cells,
platelets and smudge cells, made with the classical counter (count_cells.py), plus
the type of every white cell.

Why the classical counter and not YOLO26: on our hand-counted test squares it gave the
most accurate red-cell counts (preliminary evaluation, see evaluate.py), it needs no
trained model file, and it runs on any CPU in a few seconds.

White-cell types come from the first-release YOLO11 model (models/blood_cells.pt),
trained on PBC images: 360 px crops with one white cell in the centre. Each white cell
found by the classical counter is cut out so that it looks like a PBC image (same
red-cell size), and the model names the type of the cell in the centre.
"""
import hashlib, os, tempfile
import cv2
from ultralytics import YOLO
from count_cells import process, R_CANON

CLASSES = ["rbc", "wbc", "platelet", "smudge"]
COLORS = {"rbc": (0, 200, 0), "wbc": (255, 60, 0), "platelet": (0, 140, 255),
          "smudge": (200, 0, 200)}

TYPE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "blood_cells.pt")
WBC_TYPES = {"basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte",
             "neutrophil"}
PBC_SIZE = 360              # side of a PBC image (px)
PBC_RBC_RADIUS = 35.4       # red-cell radius in PBC images (median of 300 images)
MIN_RBC_RADIUS = 20         # below this (40x) the crop would be enlarged too much to type
# Calibration for phone photos, learned on the train + val photos only (evaluate.py):
# the PBC-trained model calls lymphocytes "erythroblast". On the test photos: 11/14
# correct with it, 4/14 without it, 7/14 for always answering "lymphocyte". It can only
# tell lymphocytes from neutrophils; rarer types are not recognised on phone photos.
TYPE_CALIBRATION = {"erythroblast": "lymphocyte"}

_cache = {}
_type_model = None


def pbc_crop(image, box, rbc_radius):
    """Square crop around the cell in box (image px), scaled so that a red cell has the
    same radius as in PBC images: what the type model sees."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = int(round(PBC_SIZE * rbc_radius / PBC_RBC_RADIUS))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    big = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_REPLICATE)
    return cv2.resize(big[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side],
                      (PBC_SIZE, PBC_SIZE))


def wbc_type(image, box, rbc_radius):
    """Type of the white cell in box (image px). rbc_radius: red-cell radius in image px."""
    global _type_model
    if rbc_radius < MIN_RBC_RADIUS:
        return "unknown (needs 100x)"
    if _type_model is None:
        _type_model = YOLO(TYPE_MODEL)
    r = _type_model.predict(pbc_crop(image, box, rbc_radius), imgsz=352, conf=0.1,
                            verbose=False)[0]
    best, best_conf = "unknown", 0.0
    for b, k, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        name = _type_model.names[int(k)]
        near_centre = (abs((b[0] + b[2]) / 2 - PBC_SIZE / 2) < PBC_SIZE / 4 and
                       abs((b[1] + b[3]) / 2 - PBC_SIZE / 2) < PBC_SIZE / 4)
        if name in WBC_TYPES and near_centre and conf > best_conf:
            best, best_conf = name, conf
    return TYPE_CALIBRATION.get(best, best)


def analyse(image):
    """Counts and boxes (in the image's own pixels) for one BGR image, plus the type of
    every white cell. The result of the latest image is cached, so Count and Show do not
    run the analysis twice. Raises ValueError when the red-cell size cannot be measured."""
    key = hashlib.md5(image.tobytes()).hexdigest()
    if key in _cache:
        return _cache[key]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "image.png")             # PNG: lossless, same counts as the file
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        row, nrm, dets, review = process(path, tmp, return_dets=True)
    s = nrm["scale"]                                      # normalised px -> image px: divide by s
    boxes = {c: [] for c in CLASSES}
    for c, x0, y0, x1, y1 in dets:
        boxes[CLASSES[c]].append((x0 / s, y0 / s, x1 / s, y1 / s))
    clumps = [((x0 / s, y0 / s, x1 / s, y1 / s), n) for x0, y0, x1, y1, n in review]
    counted = None
    if nrm["has_fov"]:                                    # circle in which cells are counted
        counted = (nrm["cx"] / s, nrm["cy"] / s, (nrm["Rf"] - R_CANON) / s)
    result = dict(counts=dict(rbc=row["rbc"], wbc=row["wbc"], platelet=row["platelets"],
                              smudge=row["smudge_or_artifact"]),    
                  boxes=boxes, clumps=clumps, counted=counted,
                  rbc_radius=nrm["rbc_radius_px"],
                  wbc_types=[wbc_type(image, b, nrm["rbc_radius_px"]) for b in boxes["wbc"]])
    _cache.clear()                                        # keep only the latest image
    _cache[key] = result
    return result


def draw(image, result, classes=CLASSES):
    """Copy of the image with the boxes of the given classes and the counted circle.
    Clumps get '+n': red cells counted from the clump area that have no box of their own.
    White cells get their type."""
    vis = image.copy()
    t = max(2, round(max(image.shape[:2]) / 800))         # line width grows with the photo
    for c in classes:
        for i, (x0, y0, x1, y1) in enumerate(result["boxes"][c]):
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[c], t)
            if c == "wbc":
                cv2.putText(vis, result["wbc_types"][i], (int(x0), int(y0) - 3 * t),
                            cv2.FONT_HERSHEY_SIMPLEX, t / 2, COLORS[c], t)
    if "rbc" in classes:
        for (x0, y0, x1, y1), n in result["clumps"]:
            cv2.putText(vis, f"+{n}", (int(x0), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,
                        t / 3, (0, 0, 200), t)
    if result["counted"]:
        cx, cy, r = result["counted"]
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), t)
    return vis