"""Field-of-view finder for eyepiece/phone photos (bright circle inside a dark rim).

Without this step the dark rim wins the background vote and the whole bright circle
becomes "foreground". With it, background removal works inside the circle only and the
rim is excluded from the mask (cells cut by the circle edge are simply cut, matching the
detector's convention of never labelling the rim).

Public API
    find_fov(img, cfg) -> uint8 0/255 full-size circle mask, or None when the image has
                          no dark rim around a bright circle (e.g. BCCD, PBC, Data_Set_1)

Method: grey image at low resolution, cut at cfg.fov_cut_frac (default 0.25) of the Otsu value (low, so the shadowed or
dim side of an unevenly lit circle still counts as glass), largest bright blob, holes filled, circle
fitted by robust least squares. Accepted only when most of the image border is dark, the blob is round
(blob area / fitted circle area >= cfg.fov_min_roundness) and the circle covers a
sensible part of the frame. The circle radius is multiplied by cfg.fov_shrink to drop the
soft rim.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from .config import BgConfig
from .thresholds import otsu_of


def _fit_circle(pts: np.ndarray, rounds: int = 4) -> tuple[tuple[float, float], float]:
    """Least-squares circle through contour points; points far from the circle (a shadow or
    halo bulging out) are dropped and the fit repeated, so one bad side cannot drag it."""
    keep = np.ones(len(pts), bool)
    for _ in range(rounds):
        x, y = pts[keep, 0], pts[keep, 1]
        A = np.c_[2 * x, 2 * y, np.ones(keep.sum())]
        (cx, cy, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        r = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
        err = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = err <= max(1.5 * np.median(err[keep]), 1.0)
    return (float(cx), float(cy)), r


def find_fov(img: np.ndarray, cfg: BgConfig) -> np.ndarray | None:
    h, w = img.shape[:2]
    k = 300.0 / max(h, w)
    small = cv.resize(img, None, fx=k, fy=k, interpolation=cv.INTER_AREA)
    gray = cv.GaussianBlur(cv.cvtColor(small, cv.COLOR_BGR2GRAY), (0, 0), 2)
    cut = cfg.fov_cut_frac * otsu_of(gray.astype(np.float32), 255.0)
    bright = (gray > cut).astype(np.uint8)
    border = np.concatenate([bright[0], bright[-1], bright[:, 0], bright[:, -1]])
    if border.mean() > 0.5:                      # frame is mostly bright: no rim
        return None
    n, labels, stats, _ = cv.connectedComponentsWithStats(bright)
    if n < 2:
        return None
    blob = (labels == 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv.findContours(blob, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv.contourArea)
    (cx, cy), r = _fit_circle(cnt.reshape(-1, 2).astype(np.float64))
    if r <= 0 or cv.contourArea(cnt) / (np.pi * r * r) < cfg.fov_min_roundness:
        return None
    if not 0.1 < np.pi * r * r / (small.shape[0] * small.shape[1]) < 0.95:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv.circle(mask, (round(cx / k), round(cy / k)), int(r / k * cfg.fov_shrink), 255, -1)
    return mask
