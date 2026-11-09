"""Cell counts, boxes and white-cell types for the web app."""
import hashlib, os, tempfile
import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from count_cells import process, R_CANON

CLASSES = ["rbc", "wbc", "platelet", "smudge"]
COLORS = {"rbc": (0, 200, 0), "wbc": (255, 60, 0), "platelet": (0, 140, 255),
          "smudge": (200, 0, 200)}

PBC_SIZE = 360
PBC_RBC_RADIUS = 35.4
MIN_RBC_RADIUS = 20  # below this (40x) we don't type the cells
NEUTROPHIL_BELOW = 0.94  # nucleus thickness in red-cell radii

_cache = {}


def pbc_crop(image, box, rbc_radius):
    """Square crop around the cell, rescaled so a red cell matches PBC images."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = int(round(PBC_SIZE * rbc_radius / PBC_RBC_RADIUS))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    big = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_REPLICATE)
    return cv2.resize(big[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side],
                      (PBC_SIZE, PBC_SIZE))


def nucleus_thickness(crop):
    """Biggest distance from inside the nucleus to its edge, in red-cell radii."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ps = lab[..., 1] - lab[..., 2]
    score = ps - np.median(ps[lab[..., 0] > np.percentile(lab[..., 0], 85)])
    m = apply_hysteresis_threshold(score, 30, 40).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, ce = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        near = np.hypot(ce[i][0] - PBC_SIZE / 2, ce[i][1] - PBC_SIZE / 2) < 1.6 * PBC_RBC_RADIUS
        if near and st[i, cv2.CC_STAT_AREA] >= 0.02 * np.pi * PBC_RBC_RADIUS ** 2:
            keep[lbl == i] = 255
    if not keep.any():
        return None
    return float(cv2.distanceTransform(keep, cv2.DIST_L2, 5).max()) / PBC_RBC_RADIUS


def wbc_type(image, box, rbc_radius):
    if rbc_radius < MIN_RBC_RADIUS:
        return "unknown (needs 100x)"
    thickness = nucleus_thickness(pbc_crop(image, box, rbc_radius))
    if thickness is None:
        return "unknown"
    return "neutrophil" if thickness < NEUTROPHIL_BELOW else "lymphocyte"


def analyse(image):
    """Counts, boxes and white-cell types for one BGR image. Cached."""
    key = hashlib.md5(image.tobytes()).hexdigest()
    if key in _cache:
        return _cache[key]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "image.png")  # PNG keeps the counts the same as the file
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        row, nrm, dets, review = process(path, tmp, return_dets=True)
    s = nrm["scale"]
    boxes = {c: [] for c in CLASSES}
    for c, x0, y0, x1, y1 in dets:
        boxes[CLASSES[c]].append((x0 / s, y0 / s, x1 / s, y1 / s))
    clumps = [((x0 / s, y0 / s, x1 / s, y1 / s), n) for x0, y0, x1, y1, n in review]
    counted = None
    if nrm["has_fov"]:
        counted = (nrm["cx"] / s, nrm["cy"] / s, (nrm["Rf"] - R_CANON) / s)
    result = dict(counts=dict(rbc=row["rbc"], wbc=row["wbc"], platelet=row["platelets"],
                              smudge=row["smudge_or_artifact"]),
                  boxes=boxes, clumps=clumps, counted=counted,
                  rbc_radius=nrm["rbc_radius_px"],
                  wbc_types=[wbc_type(image, b, nrm["rbc_radius_px"]) for b in boxes["wbc"]])
    _cache.clear()  # only keep the latest image
    _cache[key] = result
    return result


def draw(image, result, classes=CLASSES):
    vis = image.copy()
    t = max(2, round(max(image.shape[:2]) / 800))
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
