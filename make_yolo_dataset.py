"""
Build a YOLO dataset from raw smear photos using the classical pipeline as an
automatic labeller ("pseudo-labels"). Correct the labels by hand afterwards
(CVAT / Label Studio), then train YOLO26 with train_yolo26.py.

What it does per image
  1. normalise(): find the eyepiece circle and rescale so an RBC has radius 16 px
     (the same step is used at inference, so YOLO always sees cells at one size)
  2. run the classical detector -> per-cell boxes for 4 classes
  3. black out everything outside the counted area (rim cells are never labelled,
     so they must not be visible either, or YOLO learns to ignore real cells)
  4. cut 640x640 tiles (stride 512) and write YOLO txt labels
  5. split train/val/test BY SOURCE IMAGE (never by tile -> no leakage)

Usage
    python make_yolo_dataset.py raw_images/*.jpg --out dataset --val 0.2 --test 0.1
Outputs
    dataset/images/{train,val,test}/*.jpg, dataset/labels/{...}/*.txt,
    dataset/data.yaml, dataset/review.csv (blobs with more cells than boxes),
    dataset/preview/*.jpg (labels drawn, for a quick visual check)
"""
import argparse, csv, hashlib, os
import cv2
import numpy as np
from count_cells import process, circle_mask, R_CANON

CLASSES = ["rbc", "wbc", "platelet", "smudge"]
COLORS = [(0, 200, 0), (255, 60, 0), (0, 140, 255), (200, 0, 200)]
TILE, STRIDE = 640, 512


def split_of(name, val, test):
    """Deterministic split from the file name, so re-runs keep images in place."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 / 1000
    return "test" if h < test else "val" if h < test + val else "train"


def tiles(W, H):
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE)) or [0]
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)) or [0]
    if xs[-1] + TILE < W: xs.append(W - TILE)
    if ys[-1] + TILE < H: ys.append(H - TILE)
    return [(x, y) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--split", nargs="*", default=None,
                    help="explicit name=split pairs, e.g. img3=val img5=test")
    a = ap.parse_args()
    forced = dict(p.split("=") for p in (a.split or []))
    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            os.makedirs(os.path.join(a.out, sub, sp), exist_ok=True)
    os.makedirs(os.path.join(a.out, "preview"), exist_ok=True)
    tmp = os.path.join(a.out, "_classical")
    os.makedirs(tmp, exist_ok=True)

    review_rows, stats = [], {sp: [0, 0] for sp in ("train", "val", "test")}
    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        slide = name.split("_")[0]          # S07_100x_f12 -> S07: whole slide in one split
        sp = forced.get(name) or forced.get(slide) or split_of(slide, a.val, a.test)
        row, nrm, dets, review = process(path, tmp, return_dets=True)
        img = nrm["img"].copy()
        H, W = img.shape[:2]
        inner = circle_mask(img.shape, nrm["cx"], nrm["cy"], nrm["Rf"] - R_CANON)
        img[inner == 0] = 0                                  # hide unlabelled rim
        for x0, y0, x1, y1, missing in review:
            review_rows.append(dict(image=name, x1=x0, y1=y0, x2=x1, y2=y1,
                                    missing_rbc_boxes=missing, split=sp))
        prev = img.copy()
        for c, x0, y0, x1, y1 in dets:
            cv2.rectangle(prev, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[c], 2)
        for x0, y0, x1, y1, m in review:
            cv2.rectangle(prev, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(a.out, "preview", f"{name}.jpg"),
                    cv2.resize(prev, (1600, int(1600 * H / W))))

        d = np.array([b[1:] for b in dets], float) if dets else np.zeros((0, 4))
        cls = np.array([b[0] for b in dets], int)
        for tx, ty in tiles(W, H):
            tile = img[ty:ty + TILE, tx:tx + TILE]
            if (tile.max(axis=2) > 0).mean() < 0.25:         # mostly outside FOV
                continue
            lines = []
            for c, (x0, y0, x1, y1) in zip(cls, d):
                cx0, cy0 = max(x0, tx), max(y0, ty)
                cx1, cy1 = min(x1, tx + TILE), min(y1, ty + TILE)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                if (cx1 - cx0) * (cy1 - cy0) < 0.6 * (x1 - x0) * (y1 - y0):
                    continue                                 # mostly cut off -> skip
                bx, by = (cx0 + cx1) / 2 - tx, (cy0 + cy1) / 2 - ty
                bw, bh = cx1 - cx0, cy1 - cy0
                lines.append(f"{c} {bx/TILE:.6f} {by/TILE:.6f} {bw/TILE:.6f} {bh/TILE:.6f}")
            stem = f"{name}_{tx}_{ty}"
            cv2.imwrite(os.path.join(a.out, "images", sp, stem + ".jpg"), tile)
            with open(os.path.join(a.out, "labels", sp, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            stats[sp][0] += 1
            stats[sp][1] += len(lines)
        print(f"{name}: split={sp}  counts={row['rbc']} RBC / {row['wbc']} WBC  "
              f"boxes={len(dets)}  review-blobs={len(review)}")

    with open(os.path.join(a.out, "review.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "x1", "y1", "x2", "y2",
                                          "missing_rbc_boxes", "split"])
        w.writeheader()
        w.writerows(review_rows)
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\nnames:\n" +
                "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    for sp, (nt, nb) in stats.items():
        print(f"{sp:5s}: {nt} tiles, {nb} boxes")


if __name__ == "__main__":
    main()