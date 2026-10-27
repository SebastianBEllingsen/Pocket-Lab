"""Cut-out bank: single cells with soft alpha masks, stored at a fixed reference scale.

A cut-out is a small BGR patch plus an alpha mask. It is stored at BANK_R, a bank scale in
which a red cell has radius 32 px (twice the detector's canonical 16 px, so the generator can
shrink without blurring). The colour of the glass around the cell in its source image is kept
(``bg_bgr``) so the generator can turn the patch into optical density (how much light the cell
absorbs) and blend overlaps the way light really adds up.

Public API
    BANK_R                    reference RBC radius of stored cut-outs (px)
    Cutout                    one cell: class, subtype, patch, alpha, source glass colour, origin
    save_bank(cutouts, dir)   write PNG (BGRA) files and index.json
    load_bank(dir)            read them back -> list[Cutout]
    isolated_radii(mask)      equivalent radii of round, border-free blobs in a mask
    cutout_from_mask(...)     patch + soft alpha for one blob, rescaled to the bank scale
    extract_rbcs(sample, ...)  isolated red cells from a dataset image with a reference mask (Data_Set_1)
    grow_core(core, limit, ...)  grow a dark core inside a wider mask until the shape starts leaking
    extract_central(sample, ...) the one cell in the centre of a single-cell image (PBC)
    extract_neighbour_rbcs(...)  isolated red cells around the centre of a PBC image (they have the pale centre)

Classes: rbc, wbc (subtype in ``subtype``), platelet. Smudges are made by the generator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

BANK_R = 32.0
PBC_RBC_R = 36.0          # measured 2026-10-27: median radius of isolated RBCs in PBC (px); all PBC images share one zoom
EDGE_ERODE = 1            # px (source resolution) removed from the mask so no glass-coloured fringe is pasted
EDGE_FEATHER = 0.8        # Gaussian sigma (bank px) of the soft alpha edge


@dataclass
class Cutout:
    cls: str                       # rbc | wbc | platelet
    subtype: str | None            # WBC type (neutrophil, ...) or None
    bgr: np.ndarray                # HxWx3 uint8 patch at bank scale
    alpha: np.ndarray              # HxW uint8
    bg_bgr: tuple[float, float, float]   # glass colour around the cell in its source image
    source: str                    # dataset:name of the origin image


def isolated_radii(mask: np.ndarray, min_solidity: float = 0.93) -> list[tuple[int, float]]:
    """(label, equivalent radius) of blobs that are not touching the border and are nearly convex."""
    n, lab, st, _ = cv.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    h, w = mask.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h or a < 30:
            continue
        cnts, _ = cv.findContours((lab[y:y + bh, x:x + bw] == i).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        hull = cv.contourArea(cv.convexHull(max(cnts, key=cv.contourArea)))
        if hull > 0 and a / hull >= min_solidity:
            out.append((i, float(np.sqrt(a / np.pi))))
    return out


def cutout_from_mask(img: np.ndarray, blob: np.ndarray, scale: float, cls: str, subtype: str | None,
                     bg_bgr, source: str, margin: int = 6) -> Cutout:
    """Crop ``img`` around the boolean ``blob``, drop a thin edge ring, rescale by ``scale`` to the
    bank scale and give it a feathered alpha."""
    ys, xs = np.nonzero(blob)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[1])
    m = blob[y0:y1, x0:x1].astype(np.uint8) * 255
    if EDGE_ERODE:
        m = cv.erode(m, np.ones((3, 3), np.uint8), iterations=EDGE_ERODE)
    patch = img[y0:y1, x0:x1]
    size = (max(1, round(patch.shape[1] * scale)), max(1, round(patch.shape[0] * scale)))
    patch = cv.resize(patch, size, interpolation=cv.INTER_AREA)
    m = cv.resize(m, size, interpolation=cv.INTER_AREA)
    alpha = np.clip(cv.GaussianBlur(m, (0, 0), EDGE_FEATHER).astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    return Cutout(cls, subtype, patch, alpha, tuple(float(v) for v in bg_bgr), source)


def extract_rbcs(sample, background_bgr: np.ndarray, max_cells: int = 25, area_band=(0.75, 1.3),
                 rng: np.random.Generator | None = None) -> list[Cutout]:
    """Isolated, round red cells of one image with a reference mask. The image's own median
    radius sets the scale (so zoom differences between images cancel); blobs far from the
    median area (WBCs, clumps) are skipped."""
    mask = sample.load_mask()
    if mask is None:
        return []
    img = sample.load_image()
    blobs = isolated_radii(mask)
    if len(blobs) < 4:
        return []
    radii = np.array([r for _, r in blobs])
    med = float(np.median(radii))
    keep = [(i, r) for i, r in blobs if area_band[0] <= (r / med) ** 2 <= area_band[1]]
    rng = rng or np.random.default_rng(0)
    if len(keep) > max_cells:
        keep = [keep[j] for j in rng.choice(len(keep), max_cells, replace=False)]
    n, lab = cv.connectedComponents((mask > 0).astype(np.uint8))
    out = []
    for i, _ in keep:
        blob = lab == i
        cy, cx = (int(v.mean()) for v in np.nonzero(blob))
        out.append(cutout_from_mask(img, blob, BANK_R / med, "rbc", None, background_bgr[cy, cx], f"{sample.dataset}:{sample.name}"))
    return out


def _solidity(blob: np.ndarray) -> float:
    cnts, _ = cv.findContours(blob.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv.contourArea)
    hull = cv.contourArea(cv.convexHull(c))
    return float(cv.contourArea(c) / hull) if hull > 0 else 0.0


def grow_core(core: np.ndarray, limit: np.ndarray, max_px: int = 40, min_solidity: float = 0.9) -> np.ndarray:
    """Grow ``core`` (dark nucleus) 2 px at a time inside ``limit`` (the wider cell-plus-neighbours
    mask) and keep the last step whose shape is still compact. Cytoplasm gets included; a red cell
    touching the white cell is only entered while the whole blob stays compact."""
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cur = core.astype(np.uint8)
    best = cur
    for _ in range(max_px // 2):
        nxt = cv.dilate(cur, k) & limit.astype(np.uint8)
        if nxt.sum() == cur.sum():
            break
        cur = nxt
        if _solidity(cur) >= min_solidity:
            best = cur
        elif _solidity(best) >= min_solidity and cur.sum() > 1.3 * best.sum():
            break
    return best > 0


def extract_central(sample, mask: np.ndarray, background_bgr: np.ndarray, cls: str, subtype: str | None,
                    max_area_frac: float = 0.30, wide: np.ndarray | None = None) -> Cutout | None:
    """The cell in the middle of a single-cell image (PBC). Rejected when the blob touches the
    border or is huge (the cell is merged with neighbours), so only clean cut-outs enter the bank."""
    h, w = mask.shape
    n, lab, st, cen = cv.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if n < 2:
        return None
    centre = (w / 2, h / 2)
    near = [i for i in range(1, n) if st[i, cv.CC_STAT_AREA] > 60 and
            np.hypot(*(cen[i] - centre)) < 0.18 * max(h, w)]
    if not near:
        return None
    i = max(near, key=lambda k: st[k, cv.CC_STAT_AREA])
    x, y, bw, bh, a = st[i]
    if x == 0 or y == 0 or x + bw >= w or y + bh >= h or a > max_area_frac * h * w:
        return None
    blob = lab == i
    if wide is not None:        # ``mask`` is the dark core; grow it into the cytoplasm inside ``wide``
        core = cv.morphologyEx(blob.astype(np.uint8), cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        grown = grow_core(core, wide).astype(np.uint8)
        d = max(25, int(0.4 * 2 * np.sqrt(grown.sum() / np.pi)) | 1)      # opening disc: 40% of the cell diameter
        opened = cv.morphologyEx(grown, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (d, d)))   # trims stubs of touching red cells
        k, lab2 = cv.connectedComponents(opened)
        keep = [j for j in range(1, k) if (core.astype(bool) & (lab2 == j)).any()]
        blob = np.isin(lab2, keep) if keep else grown > 0
    blob = cv.morphologyEx(blob.astype(np.uint8), cv.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    cy, cx = (int(v.mean()) for v in np.nonzero(blob))
    img = sample.load_image()
    return cutout_from_mask(img, blob, BANK_R / PBC_RBC_R, cls, subtype, background_bgr[cy, cx], f"{sample.dataset}:{sample.name}")


def extract_neighbour_rbcs(sample, mask: np.ndarray, background_bgr: np.ndarray, max_cells: int = 3,
                           area_band=(0.6, 1.4)) -> list[Cutout]:
    """Isolated round red cells of a PBC image (everything except the centre cell). All PBC
    images share one zoom, so PBC_RBC_R gives the scale and the band keeps only red-cell sizes."""
    h, w = mask.shape
    img = sample.load_image()
    n, lab = cv.connectedComponents((mask > 0).astype(np.uint8))
    out = []
    for i, r in isolated_radii(mask):
        if not area_band[0] <= (r / PBC_RBC_R) ** 2 <= area_band[1]:
            continue
        blob = lab == i
        cy, cx = (int(v.mean()) for v in np.nonzero(blob))
        if np.hypot(cx - w / 2, cy - h / 2) < 0.18 * max(h, w):      # the centre cell is not a red cell
            continue
        out.append(cutout_from_mask(img, blob, BANK_R / PBC_RBC_R, "rbc", None, background_bgr[cy, cx], f"{sample.dataset}:{sample.name}"))
        if len(out) >= max_cells:
            break
    return out


def save_bank(cutouts: list[Cutout], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for k, c in enumerate(cutouts):
        fn = f"{c.cls}_{k:05d}.png"
        cv.imwrite(str(out / fn), np.dstack([c.bgr, c.alpha]))
        index.append(dict(file=fn, cls=c.cls, subtype=c.subtype, bg_bgr=list(c.bg_bgr), source=c.source))
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return out


def load_bank(bank_dir: str | Path) -> list[Cutout]:
    d = Path(bank_dir)
    out = []
    for e in json.loads((d / "index.json").read_text()):
        bgra = cv.imread(str(d / e["file"]), cv.IMREAD_UNCHANGED)
        out.append(Cutout(e["cls"], e["subtype"], bgra[..., :3].copy(), bgra[..., 3].copy(), tuple(e["bg_bgr"]), e["source"]))
    return out
