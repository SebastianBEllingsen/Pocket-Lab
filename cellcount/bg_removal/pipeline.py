"""Background removal: image -> foreground mask, flattened image and a step-by-step trace.

Pure logic, no I/O and no plotting. The ``trace`` records every intermediate
result and the values used, so a viewer (``--explain``) or a test can inspect any step.

Public API
    Step                      one recorded stage (name, caption, kind, array, values)
    BgResult                  mask, flattened, background, distance, threshold, stats, trace
    remove_background(img, cfg=BgConfig())   (finds the eyepiece circle first when cfg.fov == "auto")

Steps recorded: input -> background estimate (+ colour votes) -> flattened -> distance to background
-> histogram + threshold (see thresholds.py) -> raw mask -> holes filled -> specks removed (final).

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
from .fov import find_fov
from .methods import estimate
from .thresholds import pick_threshold


@dataclass
class Step:
    name: str
    caption: str
    kind: Literal["image", "heatmap", "hist", "mask", "swatches"]
    array: np.ndarray
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class BgResult:
    mask: np.ndarray            # uint8 0/255
    flattened: np.ndarray       # BGR uint8
    background: np.ndarray      # BGR uint8
    distance: np.ndarray        # float32, Lab distance to background
    threshold: float            # chosen threshold in distance units
    reference_lab: tuple[float, float, float]
    config: BgConfig
    trace: list[Step]
    stats: dict[str, float]
    fov: np.ndarray | None = None   # uint8 0/255 field-of-view circle (full size), None when not used


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


def _remove(img: np.ndarray, cfg: BgConfig) -> BgResult:
    h, w = img.shape[:2]
    trace: list[Step] = [Step("Input", f"Original image, {w}x{h}px", "image", img,
                              {"size": f"{w}x{h}", "mean BGR": np.round(img.reshape(-1, 3).mean(0), 1).tolist()})]

    est = estimate(img, cfg)
    background = _lab_to_bgr(est.lab)
    trace.append(Step("Background estimate", est.caption, "image", background,
                      {"method": cfg.method, **est.values,
                       "reference colour (Lab)": [round(v, 1) for v in est.reference_lab]}))

    if est.proposals:
        rgb = np.array([_lab_to_bgr(np.array([[p.lab]], np.float32))[0, 0][::-1] for p in est.proposals], np.uint8)
        trace.append(Step("Background colour votes",
                          "Each proposer names the background colour. Ticked = agrees with the winner; "
                          + ("the winner's colour seeds the background field."
                             if est.confident else "NO AGREEMENT: fell back to the first proposer."),
                          "swatches", rgb,
                          {"labels": [f"{p.name}: {p.note}" for p in est.proposals],
                           "agree": list(est.agreeing), "chosen": est.chosen,
                           "lab": [tuple(round(v) for v in p.lab) for p in est.proposals]}))

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

    threshold = pick_threshold(dist, cfg)
    raw = ((dist > threshold) * 255).astype(np.uint8)
    fg_raw = int((raw > 0).sum())
    trace.append(Step(f"Histogram + threshold ({cfg.threshold})", "Distribution of the distance map and the "
                      f"cut-off chosen by the '{cfg.threshold}' picker, per image", "hist", dist,
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

    stats = {"votes_agree": float(sum(est.agreeing)), "votes_total": float(len(est.proposals)), "threshold": threshold, "foreground_pct": 100.0 * float((mask > 0).mean()), "blobs": float(after)}
    return BgResult(mask, flat, background, dist, threshold, est.reference_lab, cfg, trace, stats)


def remove_background(img: np.ndarray, cfg: BgConfig = BgConfig()) -> BgResult:
    """Foreground mask for ``img``. With cfg.fov == "auto" and a dark rim around a bright
    circle, the work happens on the circle's bounding box (corners filled with the glass
    colour) and the mask is cut to the circle; all arrays are returned at full image size."""
    fov = find_fov(img, cfg) if cfg.fov == "auto" else None
    if fov is None:
        return _remove(img, cfg)
    h, w = img.shape[:2]
    x, y, bw, bh = cv.boundingRect(fov)
    crop, inside = img[y:y + bh, x:x + bw].copy(), fov[y:y + bh, x:x + bw] > 0
    crop[~inside] = np.median(crop[inside], axis=0).astype(np.uint8)   # typical glass colour, so it cannot win the vote
    r = _remove(crop, cfg)

    def paste(a: np.ndarray, fill=0) -> np.ndarray:
        out = np.full((h, w) + a.shape[2:], fill, a.dtype)
        out[y:y + bh, x:x + bw] = a
        return out

    mask = paste(r.mask)
    mask[fov == 0] = 0
    trace = list(r.trace)
    trace.insert(1, Step("Field of view", "Bright circle found inside a dark rim; background removal works "
                         "inside it and the rim is excluded", "mask", fov,
                         {"circle px": int((fov > 0).sum()), "box": f"{bw}x{bh} at ({x},{y})"}))
    stats = dict(r.stats, foreground_pct=100.0 * float((mask > 0).sum()) / max(1, int((fov > 0).sum())),
                 blobs=float(_count_blobs(mask)))
    return BgResult(mask, paste(r.flattened), paste(r.background), paste(r.distance), r.threshold,
                    r.reference_lab, cfg, trace, stats, fov)
