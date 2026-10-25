"""
Blood-cell analysis for the web app: counts and boxes for red cells, white cells,
platelets and smudge cells, made with the classical counter (count_cells.py).

Why the classical counter and not YOLO26: on our hand-counted test squares it gave the
most accurate red-cell counts (preliminary evaluation, see evaluate.py), it needs no
trained model file, and it runs on any CPU in a few seconds.
"""
import hashlib, os, tempfile
import cv2
from count_cells import process, R_CANON

CLASSES = ["rbc", "wbc", "platelet", "smudge"]
COLORS = {"rbc": (0, 200, 0), "wbc": (255, 60, 0), "platelet": (0, 140, 255),
          "smudge": (200, 0, 200)}

_cache = {}


def analyse(image):
    """Counts and boxes (in the image's own pixels) for one BGR image. The result of the
    latest image is cached, so Count and Show do not run the analysis twice. Raises
    ValueError when the red-cell size cannot be measured."""
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
                  boxes=boxes, clumps=clumps, counted=counted)
    _cache.clear()                                        # keep only the latest image
    _cache[key] = result
    return result


def draw(image, result, classes=CLASSES):
    """Copy of the image with the boxes of the given classes and the counted circle.
    Clumps get '+n': red cells counted from the clump area that have no box of their own."""
    vis = image.copy()
    t = max(2, round(max(image.shape[:2]) / 800))         # line width grows with the photo
    for c in classes:
        for x0, y0, x1, y1 in result["boxes"][c]:
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[c], t)
    if "rbc" in classes:
        for (x0, y0, x1, y1), n in result["clumps"]:
            cv2.putText(vis, f"+{n}", (int(x0), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,
                        t / 2, (0, 0, 200), t)
    if result["counted"]:
        cx, cy, r = result["counted"]
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), t)
    return vis