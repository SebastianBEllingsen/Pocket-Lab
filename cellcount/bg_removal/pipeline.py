"""Background removal: image -> foreground mask, flattened image and a step-by-step trace.

Pure logic, no I/O and no plotting. The ``trace`` records every intermediate
result and the values used, so a viewer (``--explain``) or a test can inspect any step.

Public API
    Step                      one recorded stage (name, caption, kind, array, values)
    BgResult                  mask, flattened, background, distance, threshold, stats, trace
    remove_background(img, cfg=BgConfig())

Steps recorded: input -> background estimate -> flattened -> distance to background
-> histogram + Otsu threshold -> raw mask -> holes filled -> specks removed (final).

Flattening (image / background) is for display and downstream stages; the mask
itself is derived from the Lab distance to the background colour, so it does not
depend on any fixed stain colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import cv2 as cv
import numpy as np

from .config import BgConfig
from .methods import estimate


@dataclass
class Step:
    name: str
    caption: str
    kind: Literal["image", "heatmap", "hist", "mask"]
    array: np.ndarray
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class BgResult:
    mask: np.ndarray            # uint8 0/255
    flattened: np.ndarray       # BGR uint8
    background: np.ndarray      # BGR uint8
    distance: np.ndarray        # float32, Lab distance to background
    threshold: float            # Otsu threshold in distance units
    reference_lab: tuple[float, float, float]
    config: BgConfig
    trace: list[Step]
    stats: dict[str, float]


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    return cv.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv.COLOR_LAB2BGR)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes; the 1px border keeps this correct when a cell touches the corner."""
    pad = cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    inv = cv.bitwise_not(pad)
    cv.floodFill(inv, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 0)
    return cv.bitwise_or(mask, inv[1:-1, 1:-1])


def remove_small_blobs(mask: np.ndarray, min_area: float) -> np.ndarray:
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv.CC_STAT_AREA] >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def _count_blobs(mask: np.ndarray) -> int:
    return cv.connectedComponents(mask)[0] - 1


def remove_background(img: np.ndarray, cfg: BgConfig = BgConfig()) -> BgResult:
    h, w = img.shape[:2]
    trace: list[Step] = [Step("Input", f"Original image, {w}x{h}px", "image", img,
                              {"size": f"{w}x{h}", "mean BGR": np.round(img.reshape(-1, 3).mean(0), 1).tolist()})]

    est = estimate(img, cfg)
    background = _lab_to_bgr(est.lab)
    trace.append(Step("Background estimate", est.caption, "image", background,
                      {"method": cfg.method, **est.values,
                       "reference colour (Lab)": [round(v, 1) for v in est.reference_lab]}))

    flat = np.clip(img.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0) * 255.0,
                   0, 255).astype(np.uint8)
    trace.append(Step("Flattened", "Image divided by the estimated background: uneven lighting removed, "
                      "background close to white", "image", flat))

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - est.lab, axis=2)
    if cfg.blur > 0:
        dist = cv.GaussianBlur(dist, (0, 0), cfg.blur)
    dist = dist.astype(np.float32)
    dmax = float(max(dist.max(), 1e-6))
    trace.append(Step("Distance to background", f"Lab colour distance of each pixel to the background "
                      f"(blurred, sigma {cfg.blur:g}px). Bright = unlike background", "heatmap", dist,
                      {"min": round(float(dist.min()), 1), "max": round(dmax, 1), "mean": round(float(dist.mean()), 1)}))

    d8 = np.clip(dist / dmax * 255, 0, 255).astype(np.uint8)
    t8, raw = cv.threshold(d8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    threshold = float(t8) / 255.0 * dmax
    fg_raw = int((raw > 0).sum())
    trace.append(Step("Histogram + Otsu", "Distribution of the distance map. Otsu picks the threshold that best "
                      "separates background from cells, per image", "hist", dist,
                      {"threshold": round(threshold, 2)}))
    trace.append(Step("Raw mask", "Pixels with distance above the threshold", "mask", raw,
                      {"foreground px": fg_raw, "blobs": _count_blobs(raw)}))

    mask = fill_holes(raw) if cfg.fill_holes else raw
    trace.append(Step("Holes filled", "Enclosed holes (pale cell centres) filled" if cfg.fill_holes
                      else "Hole filling disabled (--no-fill-holes)", "mask", mask,
                      {"px added": int((mask > 0).sum()) - fg_raw, "blobs": _count_blobs(mask)}))

    before = _count_blobs(mask)
    mask = remove_small_blobs(mask, cfg.min_area_frac * mask.size)
    after = _count_blobs(mask)
    trace.append(Step("Specks removed", f"Blobs smaller than {cfg.min_area_frac * mask.size:.0f}px "
                      f"({cfg.min_area_frac:g} of image) removed. FINAL MASK", "mask", mask,
                      {"blobs removed": before - after, "blobs left": after}))

    stats = {"threshold": threshold, "foreground_pct": 100.0 * float((mask > 0).mean()), "blobs": float(after)}
    return BgResult(mask, flat, background, dist, threshold, est.reference_lab, cfg, trace, stats)
