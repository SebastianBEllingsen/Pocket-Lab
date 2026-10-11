"""Outline colour selection that contrasts with the image.

Public API
    pick_contrasting_colors(img_bgr, masks, n)  n RGB colours that stand out against the
                                                pixels around the given mask edges and
                                                differ from each other
    parse_color(text)                           "cyan", "#00ffff" or "0,255,255" -> RGB tuple

Method: score each candidate colour by its median CIE-Lab distance to the image
pixels in a band around the mask edges, then pick greedily, skipping candidates
too close to ones already chosen.
"""
from __future__ import annotations

from typing import Sequence

import cv2 as cv
import numpy as np

RGB = tuple[int, int, int]

NAMED: dict[str, RGB] = {
    "cyan": (0, 255, 255), "yellow": (255, 255, 0), "magenta": (255, 0, 255),
    "lime": (0, 255, 0), "orange": (255, 140, 0), "red": (255, 0, 0),
    "blue": (0, 80, 255), "white": (255, 255, 255), "black": (0, 0, 0),
}
MIN_SEPARATION = 60.0  # Lab distance between two chosen colours
_MAX_SIDE = 400


def parse_color(text: str) -> RGB:
    t = text.strip().lower()
    if t in NAMED:
        return NAMED[t]
    if t.startswith("#") and len(t) == 7:
        return tuple(int(t[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    parts = t.split(",")
    if len(parts) == 3:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    raise ValueError(f"Cannot parse colour {text!r}; use a name ({', '.join(NAMED)}), #rrggbb or r,g,b")


def _lab(rgb: RGB) -> np.ndarray:
    px = np.array([[rgb[::-1]]], np.uint8)
    return cv.cvtColor(px, cv.COLOR_BGR2LAB).reshape(3).astype(np.float32)


def _edge_band(masks: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    band = np.zeros(shape, bool)
    k = np.ones((5, 5), np.uint8)
    for m in masks:
        small = cv.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv.INTER_NEAREST)
        band |= cv.morphologyEx(small, cv.MORPH_GRADIENT, k).astype(bool)
    return band


def pick_contrasting_colors(img_bgr: np.ndarray, masks: Sequence[np.ndarray], n: int) -> list[RGB]:
    scale = min(1.0, _MAX_SIDE / max(img_bgr.shape[:2]))
    small = cv.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    lab = cv.cvtColor(small, cv.COLOR_BGR2LAB).astype(np.float32)
    band = _edge_band(masks, small.shape[:2]) if masks else np.ones(small.shape[:2], bool)
    pixels = lab[band] if band.any() else lab.reshape(-1, 3)
    pixels = pixels[:: max(1, len(pixels) // 5000)]
    scored = []
    for name, rgb in NAMED.items():
        if name in ("white", "black"):
            continue  # reserved for manual choice; saturated hues read better
        scored.append((float(np.median(np.linalg.norm(pixels - _lab(rgb), axis=1))), rgb))
    scored.sort(reverse=True)
    chosen: list[RGB] = []
    for _, rgb in scored:
        if all(np.linalg.norm(_lab(rgb) - _lab(c)) >= MIN_SEPARATION for c in chosen):
            chosen.append(rgb)
        if len(chosen) == n:
            return chosen
    # not enough distinct hues: pad with the best remaining ones
    chosen += [rgb for _, rgb in scored if rgb not in chosen][: n - len(chosen)]
    return chosen
