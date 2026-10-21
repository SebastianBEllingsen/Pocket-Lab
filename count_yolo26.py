"""
Count blood cells in new smear photos with a trained YOLO26 model.

    python count_yolo26.py new_images/*.jpg --weights runs/detect/bloodcells/weights/best.pt

Steps per image
  1. normalise(): same FOV detection + rescaling as during training (RBC radius 16 px)
  2. black out the rim outside the counted area (same as training)
  3. tiled inference: 640x640 tiles, stride 512
  4. merge tiles: map boxes back to the full image, class-wise NMS removes
     duplicates from overlapping tiles
  5. count per class, write <out>/counts_yolo.csv and annotated images
"""
import argparse, csv, os
import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO
from count_cells import normalise, circle_mask, R_CANON

TILE, STRIDE = 640, 512
COLORS = [(0, 200, 0), (255, 60, 0), (0, 140, 255), (200, 0, 200)]


def tile_origins(W, H):
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE)) or [0]
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)) or [0]
    if xs[-1] + TILE < W: xs.append(W - TILE)
    if ys[-1] + TILE < H: ys.append(H - TILE)
    return [(x, y) for y in ys for x in xs]


def count_image(model, path, out_dir, conf):
    nrm = normalise(cv2.imread(path))
    img = nrm["img"].copy()
    H, W = img.shape[:2]
    inner = circle_mask(img.shape, nrm["cx"], nrm["cy"], nrm["Rf"] - R_CANON)
    img[inner == 0] = 0
    boxes, scores, classes = [], [], []
    origins = [(x, y) for x, y in tile_origins(W, H)
               if (img[y:y + TILE, x:x + TILE].max(axis=2) > 0).mean() >= 0.1]
    crops = [img[y:y + TILE, x:x + TILE] for x, y in origins]
    for i in range(0, len(crops), 8):
        results = model.predict(crops[i:i + 8], imgsz=TILE, conf=conf,
                                max_det=1000, verbose=False)
        for (x, y), r in zip(origins[i:i + 8], results):
            b = r.boxes.xyxy.cpu().numpy() + [x, y, x, y]
            boxes.append(b)
            scores.append(r.boxes.conf.cpu().numpy())
            classes.append(r.boxes.cls.cpu().numpy().astype(int))
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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.weights)
    rows = [count_image(model, p, a.out, a.conf) for p in a.images]
    with open(os.path.join(a.out, "counts_yolo.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)