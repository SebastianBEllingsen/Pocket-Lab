import argparse, csv, os
import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from skimage.filters import apply_hysteresis_threshold
from ultralytics import YOLO
from count_cells import normalise, count_masks, circle_mask, R_CANON, flat_background, A_RBC

TILE, STRIDE = 640, 512
COLORS = [(0, 200, 0), (255, 60, 0), (0, 140, 255), (200, 0, 200)]


def tile_origins(W, H):
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE)) or [0]
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)) or [0]
    if xs[-1] + TILE < W: xs.append(W - TILE)
    if ys[-1] + TILE < H: ys.append(H - TILE)
    return [(x, y) for y in ys for x in xs]

def stain_maps(img, field):
    """Same purple-stain measures as count_cells.process(): darkness relative to the
    glass (dL), dark-nucleus seeds (strong) and the stained-object mask (stain)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    ps = lab[..., 1].astype(np.float32) - lab[..., 2].astype(np.float32)
    dL = L - flat_background(lab[..., 0]).astype(np.float32)
    bg_sel = (L > np.percentile(L[field > 0], 85)) & (field > 0)
    score = ps - np.median(ps[bg_sel])
    strong = (score > 40) & (dL < -75) & (field > 0)
    weak = (score > 30) & (field > 0)
    stain = (apply_hysteresis_threshold(score * weak, 30, 40) & weak).astype(np.uint8) * 255
    stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    stain = cv2.morphologyEx(stain, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return dL, strong, stain


def relabel_smudge(nrm, b, c, names):
    """Stage A never saw a smudge cell, so it calls them WBC. Check each WBC box with
    the classical rule: a WBC has a dark nucleus; a large pale purple blob is smudge."""
    ids = {v: k for k, v in names.items()}
    field, _ = count_masks(nrm)
    dL, strong, stain = stain_maps(nrm["img"], field)
    c = c.copy()
    for j in np.flatnonzero(c == ids["wbc"]):
        x0, y0, x1, y1 = np.maximum(b[j], 0).astype(int)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(stain[y0:y1, x0:x1])
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))      # biggest stained blob in the box
        sel, a = lbl == i, st[i, cv2.CC_STAT_AREA]
        is_wbc = (a >= 0.8 * A_RBC and strong[y0:y1, x0:x1][sel].any()
                  and dL[y0:y1, x0:x1][sel].mean() < -78)
        if not is_wbc and a >= 2.0 * A_RBC:
            c[j] = ids["smudge"]
    return c

def count_image(model, path, out_dir, conf, has_fov=None, rbc_radius=None):
    nrm = normalise(cv2.imread(path), has_fov, rbc_radius)
    img = nrm["img"].copy()
    H, W = img.shape[:2]
    _, inner = count_masks(nrm)
    img[inner == 0] = 0
    boxes, scores, classes = [], [], []
    origins = [(x, y) for x, y in tile_origins(W, H)
               if (img[y:y + TILE, x:x + TILE].max(axis=2) > 0).mean() >= 0.1]
    crops = [img[y:y + TILE, x:x + TILE] for x, y in origins]
    for i in range(0, len(crops), 8):
        results = model.predict(crops[i:i + 8], imgsz=TILE, conf=conf,
                                max_det=1000, verbose=False)
        for (x, y), r in zip(origins[i:i + 8], results):
            b = r.boxes.xyxy.cpu().numpy()
            # A box touching an inner tile edge is a cut-off cell: the neighbouring tile
            # overlaps by 128 px (more than any cell) and sees the whole cell, so drop it.
            cut = (((b[:, 0] <= 2) & (x > 0)) | ((b[:, 1] <= 2) & (y > 0)) |
                   ((b[:, 2] >= TILE - 2) & (x + TILE < W)) | ((b[:, 3] >= TILE - 2) & (y + TILE < H)))
            boxes.append(b[~cut] + [x, y, x, y])
            scores.append(r.boxes.conf.cpu().numpy()[~cut])
            classes.append(r.boxes.cls.cpu().numpy().astype(int)[~cut])
    if boxes:
        b = torch.tensor(np.concatenate(boxes), dtype=torch.float32)
        s = torch.tensor(np.concatenate(scores), dtype=torch.float32)
        c = torch.tensor(np.concatenate(classes))
        keep = batched_nms(b, s, c, iou_threshold=0.5).numpy()
        b, c = b.numpy()[keep], c.numpy()[keep]
    else:
        b, c = np.zeros((0, 4)), np.zeros(0, int)
    # count a box only if its centre lies inside the counted area
    cxs = ((b[:, 0] + b[:, 2]) / 2).astype(int).clip(0, W - 1)
    cys = ((b[:, 1] + b[:, 3]) / 2).astype(int).clip(0, H - 1)
    ok = inner[cys, cxs] > 0
    b, c = b[ok], c[ok]
    c = relabel_smudge(nrm, b, c, model.names)
    names = model.names
    counts = {names[k]: int((c == k).sum()) for k in names}
    vis = img.copy()
    for (x0, y0, x1, y1), k in zip(b.astype(int), c):
        cv2.rectangle(vis, (x0, y0), (x1, y1), COLORS[k % 4], 2)
    cv2.putText(vis, "  ".join(f"{k.upper()} {v}" for k, v in counts.items()),
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, W / 1100, (255, 255, 255), 3)
    name = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{name}_yolo.jpg"),
                cv2.resize(vis, (1600, int(1600 * H / W))))
    return dict(image=name, **counts, rbc_radius_px=round(nrm["rbc_radius_px"], 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="results_yolo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-fov", action="store_true", help="plain crops without an eyepiece circle")
    ap.add_argument("--rbc-radius", type=float, default=None,
                    help="RBC radius in raw px, used if too few round cells are found")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.weights)
    rows = [count_image(model, p, a.out, a.conf, False if a.no_fov else None, a.rbc_radius)
            for p in a.images]
    with open(os.path.join(a.out, "counts_yolo.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)