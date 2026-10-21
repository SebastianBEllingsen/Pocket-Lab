"""
Blood smear cell counter - classical machine vision, scale- and colour-invariant.

Designed for eyepiece/phone photos of Wright/Giemsa smears taken at DIFFERENT
magnifications and white balances (yellow, pink or white background).

Pipeline
  1. Field of view  : find the bright eyepiece circle, ignore the dark rim.
  2. Scale          : measure the red-cell radius automatically from isolated,
                      round blobs, then rescale every image so an RBC has
                      radius R_CANON px. All later parameters are in "RBC units",
                      so the same code works at 40x and 100x.
  3. Colour         : work RELATIVE to the local empty-glass background
                      (Lab lightness and a*-b* "purple score"), so a colour
                      cast or vignetting does not matter.
  4. Stained objects: hysteresis segmentation of the purple stain, then
                        WBC      = large + dark nucleus
                        platelet = small purple dot
                        smudge   = large but pale purple blob (broken cell /
                                   stain debris) -> reported, NOT counted as WBC
  5. RBCs           : foreground mask (darker than glass, central pallor filled)
                      -> distance transform -> one seed per cell.
                      Overlapping cells that give only one seed are recovered
                      from blob area: count = max(seeds, round(area / A_single)).

Usage
    python count_cells.py img1.jpg img2.jpg ... [--out results]
Writes <out>/<name>_annotated.jpg and <out>/counts.csv
"""
import argparse, csv, os
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import apply_hysteresis_threshold
from skimage.segmentation import watershed
from scipy import ndimage as ndi

R_CANON = 16.0                      # RBC radius after rescaling (px)
A_RBC = np.pi * R_CANON ** 2        # one RBC area in canonical px


# ----------------------------------------------------------------- helpers
def find_fov(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, f = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), R = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return cx, cy, R


def circle_mask(shape, cx, cy, r):
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (int(cx), int(cy)), int(r), 255, -1)
    return m


def flat_background(ch, k=15):
    """Empty-glass brightness. Cells may cover >50 % of the field, so use a
    local MAXIMUM (grey dilation) larger than a cell instead of a median."""
    H, W = ch.shape
    small = cv2.resize(ch, (W // 10, H // 10), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 3)
    small = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    small = cv2.GaussianBlur(small, (0, 0), k / 2)
    return cv2.resize(small, (W, H))


def foreground(img, fov):
    """Binary mask of cells (darker than glass) with the RBC central pallor filled."""
    g = img[:, :, 1]                                   # green = best RBC contrast
    d = cv2.subtract(flat_background(g), g)
    d[fov == 0] = 0
    t, _ = cv2.threshold(d[fov > 0].reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((d > 0.6 * t) & (fov > 0)).astype(np.uint8) * 255   # 0.6: keep pale rims
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    H, W = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 255)
    m |= cv2.bitwise_not(ff)                           # fill holes (pallor)
    return m


def round_blob_radii(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rs = []
    for k in cnts:
        a = cv2.contourArea(k)
        if a < 30:
            continue
        hull = cv2.contourArea(cv2.convexHull(k))
        per = cv2.arcLength(k, True)
        if a / hull > 0.93 and 4 * np.pi * a / per ** 2 > 0.8:
            rs.append(np.sqrt(a / np.pi))
    return np.array(rs)


# ----------------------------------------------------------------- main step
def normalise(raw):
    """Find the FOV, measure the RBC radius and rescale so RBC radius == R_CANON.
    Shared by the classical counter, the YOLO dataset builder and YOLO inference,
    so the network always sees cells at the same size."""
    cx, cy, R = find_fov(raw)
    s1 = 800.0 / R
    im1 = cv2.resize(raw, None, fx=s1, fy=s1, interpolation=cv2.INTER_AREA)
    fov1 = circle_mask(im1.shape, cx * s1, cy * s1, 0.93 * 800)
    m1 = cv2.morphologyEx(foreground(im1, fov1), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    r_est = float(np.median(round_blob_radii(m1)))
    s = s1 * R_CANON / r_est
    img = cv2.resize(raw, None, fx=s, fy=s,
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    cx, cy, Rf = cx * s, cy * s, 0.93 * R * s
    return dict(img=img, scale=s, cx=cx, cy=cy, Rf=Rf, rbc_radius_px=r_est / s1)


def process(path, out_dir, return_dets=False):
    raw = cv2.imread(path)
    nrm = normalise(raw)
    img, s, cx, cy, Rf = nrm["img"], nrm["scale"], nrm["cx"], nrm["cy"], nrm["Rf"]
    H, W = img.shape[:2]
    fov = circle_mask(img.shape, cx, cy, Rf)
    inner = circle_mask(img.shape, cx, cy, Rf - R_CANON)   # count only fully visible cells

    # 3. colour relative to background
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    ps = lab[..., 1].astype(np.float32) - lab[..., 2].astype(np.float32)
    dL = L - flat_background(lab[..., 0]).astype(np.float32)        # <0 = darker
    bg_sel = (L > np.percentile(L[fov > 0], 85)) & (fov > 0)
    score = ps - np.median(ps[bg_sel])                                 # 0 = glass

    # 4. stained objects
    strong = (score > 40) & (dL < -75) & (fov > 0)
    weak = (score > 30) & (fov > 0)
    stain = (apply_hysteresis_threshold(score * weak, 30, 40) & weak).astype(np.uint8) * 255
    stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    stain = cv2.morphologyEx(stain, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))  # join nuclear lobes
    n, lbl, st, ce = cv2.connectedComponentsWithStats(stain)
    wbc, plt, smudge = [], [], []
    exclude = np.zeros((H, W), np.uint8)          # area that must not count as RBC
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        sel = lbl == i
        x, y = ce[i]
        if not inner[int(y), int(x)]:
            continue
        has_seed = strong[sel].any()
        if a >= 0.8 * A_RBC and has_seed and dL[sel].mean() < -78:
            wbc.append((x, y, a))
            cv2.circle(exclude, (int(x), int(y)), int(np.sqrt(a / np.pi) * 2.0), 255, -1)
        elif a >= 2.0 * A_RBC:
            smudge.append((x, y, a, st[i]))
            bx, by, bw, bh = st[i][:4]
            cv2.rectangle(exclude, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8), 255, -1)
        elif 0.04 * A_RBC <= a < 0.6 * A_RBC and has_seed:
            plt.append((x, y, a))

    # 5. RBCs: mask -> distance transform seeds -> area correction for overlaps
    fg = foreground(img, fov)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fg[exclude > 0] = 0
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    peaks = peak_local_max(dist, min_distance=int(0.6 * R_CANON),
                           threshold_abs=0.45 * R_CANON, exclude_border=False)
    ncomp, comp, cst, _ = cv2.connectedComponentsWithStats(fg)
    seeds_per = np.zeros(ncomp, int)
    for y, x in peaks:
        seeds_per[comp[y, x]] += 1
    # single-cell area measured on THIS image (round blobs with one seed)
    singles = [cst[i, cv2.CC_STAT_AREA] for i in range(1, ncomp) if seeds_per[i] == 1]
    a_single = float(np.median(singles)) if singles else A_RBC
    rbc_total, rbc_overlap_extra, comp_counts = 0, 0, {}
    for i in range(1, ncomp):
        x0, y0, w, h, a = cst[i]
        cyy, cxx = y0 + h / 2, x0 + w / 2
        if not inner[int(cyy), int(cxx)]:
            continue
        if a < 0.3 * a_single:                    # debris / platelet-sized
            continue
        k_area = int(round(a / (0.85 * a_single)))   # 0.85: overlaps hide area
        k = max(seeds_per[i], k_area, 1)
        rbc_total += k
        rbc_overlap_extra += k - max(seeds_per[i], 1)
        comp_counts[i] = k

    # per-cell boxes (for YOLO labels): watershed from the seeds inside counted blobs
    markers = np.zeros((H, W), np.int32)
    n_m = 0
    for y, x in peaks:
        if comp[y, x] in comp_counts:
            n_m += 1
            markers[y, x] = n_m
    ws = watershed(-dist, markers, mask=fg > 0)
    dets = []                                   # (class, x1, y1, x2, y2) canonical px
    for lab_, sl in enumerate(ndi.find_objects(ws), start=1):
        if sl is None:
            continue
        y0_, y1_, x0_, x1_ = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        dets.append((0, x0_ - 2, y0_ - 2, x1_ + 2, y1_ + 2))
    review = []                                 # blobs with more cells than seeds
    for i, k in comp_counts.items():
        if k > max(seeds_per[i], 1):
            x0, y0, w, h, _ = cst[i]
            review.append((x0, y0, x0 + w, y0 + h, int(k - max(seeds_per[i], 1))))
    for x, y, a in wbc:
        r_ = np.sqrt(a / np.pi) * 2.0
        dets.append((1, x - r_, y - r_, x + r_, y + r_))
    for x, y, a in plt:
        r_ = max(np.sqrt(a / np.pi) * 1.6, 5)
        dets.append((2, x - r_, y - r_, x + r_, y + r_))
    for x, y, a, b in smudge:
        dets.append((3, b[0], b[1], b[0] + b[2], b[1] + b[3]))

    # ------------------------------------------------------------ output
    vis = img.copy()
    cs, _ = cv2.findContours(np.isin(comp, list(comp_counts)).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs, -1, (0, 200, 0), 2)
    for y, x in peaks:
        if comp[y, x] in comp_counts:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 120, 0), -1)
    for i, k in comp_counts.items():                      # mark clumps with their count
        if k >= 2:
            x0, y0, w, h, _ = cst[i]
            cv2.putText(vis, str(k), (x0 + w // 2 - 8, y0 + h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    for x, y, a in wbc:
        cv2.circle(vis, (int(x), int(y)), int(np.sqrt(a / np.pi) * 2.0), (255, 60, 0), 6)
    for x, y, a in plt:
        cv2.circle(vis, (int(x), int(y)), int(R_CANON * 0.6), (0, 140, 255), 3)
    for x, y, a, b in smudge:
        cv2.rectangle(vis, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (200, 0, 200), 5)
    cv2.circle(vis, (int(cx), int(cy)), int(Rf - R_CANON), (0, 0, 255), 2)
    fs = W / 1100
    cv2.putText(vis, f"RBC {rbc_total}  WBC {len(wbc)}  PLT {len(plt)}  smudge {len(smudge)}",
                (int(W * 0.02), int(H * 0.06)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), max(2, int(fs * 2.5)))
    name = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{name}_annotated.jpg"),
                cv2.resize(vis, (2000, int(2000 * H / W)), interpolation=cv2.INTER_AREA))
    r_px = nrm["rbc_radius_px"]
    row = dict(image=name, rbc=int(rbc_total), rbc_from_overlaps=int(rbc_overlap_extra),
                wbc=len(wbc), platelets=len(plt), smudge_or_artifact=len(smudge),
                rbc_to_wbc=round(rbc_total / len(wbc), 1) if wbc else None,
                rbc_radius_px=round(r_px, 1),
                approx_um_per_px=round(3.75 / r_px, 3))   # assumes 7.5 µm RBC
    if return_dets:
        return row, nrm, dets, review
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [process(p, a.out) for p in a.images]
    with open(os.path.join(a.out, "counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    tot = {k: sum(r[k] for r in rows) for k in ("rbc", "wbc", "platelets")}
    for r in rows:
        print(r)
    print("TOTAL", tot, "RBC:WBC =", round(tot["rbc"] / max(tot["wbc"], 1), 1))