"""
Can the shape of the nucleus tell lymphocytes from neutrophils, at 100x and at 40x detail?

    python scripts/wbc_features.py

Reads data/wbc_types/crops and ground_truth/wbc_types.csv. Test photos are skipped.
Each rule is scored on photos it was not fitted on. Prints only, writes nothing.
"""
import csv, os
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import apply_hysteresis_threshold

CROPS = "data/wbc_types/crops"
LABELS = "ground_truth/wbc_types.csv"
SIZE = 360                  # side of a crop (px)
R = 35.4                    # red-cell radius in a crop (px)
A_RBC = np.pi * R ** 2
R_40X = 11.2                # red-cell radius in our 40x photos (px)
REACH = 1.6 * R             # nucleus parts must lie this close to the crop centre
FEATURES = ["area", "solidity", "lobes", "thickness"]


def as_40x(crop):
    """The same crop with the detail of a 40x photo, back at crop size."""
    f = R_40X / R
    small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)


def nucleus_mask(crop):
    """Purple pixels near the crop centre, with the same stain score as the counter."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    ps = lab[..., 1] - lab[..., 2]
    score = ps - np.median(ps[lab[..., 0] > np.percentile(lab[..., 0], 85)])
    m = apply_hysteresis_threshold(score, 30, 40).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, ce = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        near = np.hypot(ce[i][0] - SIZE / 2, ce[i][1] - SIZE / 2) < REACH
        if near and st[i, cv2.CC_STAT_AREA] >= 0.02 * A_RBC:
            keep[lbl == i] = 255
    return keep


def features(mask):
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    area = cv2.countNonZero(mask)
    hull = cv2.contourArea(cv2.convexHull(pts))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peaks = peak_local_max(dist, min_distance=int(0.25 * R), threshold_abs=0.15 * R,
                           exclude_border=False)
    return dict(area=area / A_RBC, solidity=area / max(hull, 1.0),
                lobes=float(len(peaks)), thickness=float(dist.max()) / R)


def balanced(y, pred):
    return (pred[y].mean() + (~pred[~y]).mean()) / 2


def best_rule(x, y):
    """Threshold and direction with the best balanced accuracy."""
    best = (-1.0, 0.0, 1)
    for t in np.unique(x):
        for sign in (1, -1):
            b = balanced(y, sign * x > sign * t)
            if b > best[0]:
                best = (b, t, sign)
    return best


def leave_one_photo_out(x, y, photos):
    pred = np.zeros(len(y), bool)
    for p in np.unique(photos):
        test = photos == p
        _, t, sign = best_rule(x[~test], y[~test])
        pred[test] = sign * x[test] > sign * t
    return balanced(y, pred)


def report(name, rows):
    y = np.array([r["type"] == "neutrophil" for r in rows])
    photos = np.array([r["photo"] for r in rows])
    print(f"\n{name}: {(~y).sum()} lymphocytes, {y.sum()} neutrophils, "
          f"{len(set(photos))} photos")
    print(f"{'feature':<10} {'lymphocyte':>11} {'neutrophil':>11} {'unseen photos':>14}")
    for f in FEATURES:
        x = np.array([r[f] for r in rows])
        print(f"{f:<10} {np.median(x[~y]):>11.2f} {np.median(x[y]):>11.2f} "
              f"{leave_one_photo_out(x, y, photos):>14.2f}")


def show_values(name, rows):
    """Thickness of every cell, to place the threshold in the gap between the types."""
    print(f"\n{name}: thickness per type")
    for t in sorted({r["type"] for r in rows}):
        print(f"  {t:<11}", " ".join(f"{r['thickness']:.2f}"
                                     for r in sorted(rows, key=lambda c: c["thickness"])
                                     if r["type"] == t))


def main():
    all_cells = [r for r in csv.DictReader(open(LABELS))
                 if r["split"] != "test" and r["type"] not in ("not_wbc", "unsure", "smudge")]
    rows_100, rows_40, others = [], [], []
    for r in all_cells:
        crop = cv2.imread(os.path.join(CROPS, r["crop"] + ".jpg"))
        if crop is None:
            print("missing crop:", r["crop"])
            continue
        main_type = r["type"] in ("lymphocyte", "neutrophil")
        targets = ((rows_100, crop), (rows_40, as_40x(crop))) if main_type else ((others, crop),)
        for rows, image in targets:
            f = features(nucleus_mask(image))
            if f is None:
                print("no nucleus found:", r["crop"])
            else:
                rows.append({**r, **f})
    report("100x detail", rows_100)
    report("40x detail (simulated)", rows_40)
    print("\nBalanced accuracy on unseen photos. Always answering lymphocyte = 0.50, "
          "current model on 100x = 0.64, needed = 0.75")
    show_values("100x detail", rows_100 + others)
    show_values("40x detail (simulated)", rows_40)


if __name__ == "__main__":
    main()