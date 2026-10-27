"""Threshold pickers: from the Lab distance-to-background map to one cut-off value.

Why several: a single Otsu threshold is pulled up by a small dark region (a WBC nucleus),
so pale red cells fall below it and come out as open rings or are missed. The lower
variants keep pale cells; which one is best is judged by eye (``--threshold a,b,c``
compares them side by side).

Public API
    register_threshold(name)   decorator that adds a picker to the registry
    THRESHOLDS                 name -> picker(dist, cfg) -> float
    pick_threshold(dist, cfg)  run the picker chosen by cfg.threshold
    otsu_of(values, vmax)      Otsu cut-off of 1-D values in [0, vmax]

Pickers
    otsu     Otsu on the whole map (the original behaviour)
    half     cfg.threshold_scale (default 0.5) times the Otsu value
    lower    (default, chosen by Sebastian 2026-10-27) Otsu again on only the values below the first Otsu cut-off (a 3-class split
             that takes the bottom boundary: background vs. pale cells)
    clipped  Otsu after clipping the map at the median of the values above the first
             cut-off, so a few very dark pixels cannot dominate the split
"""
from __future__ import annotations

from typing import Callable

import cv2 as cv
import numpy as np

from .config import BgConfig

Picker = Callable[[np.ndarray, BgConfig], float]
THRESHOLDS: dict[str, Picker] = {}


def register_threshold(name: str):
    def deco(fn: Picker) -> Picker:
        THRESHOLDS[name] = fn
        return fn
    return deco


def otsu_of(values: np.ndarray, vmax: float) -> float:
    """Otsu cut-off of ``values`` (any shape), quantised to 8 bits over [0, vmax]."""
    if values.size == 0 or vmax <= 0:
        return 0.0
    v8 = np.clip(values / vmax * 255, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv.threshold(v8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return float(t) / 255.0 * vmax


@register_threshold("otsu")
def _otsu(dist: np.ndarray, cfg: BgConfig) -> float:
    return otsu_of(dist, float(max(dist.max(), 1e-6)))


@register_threshold("half")
def _half(dist: np.ndarray, cfg: BgConfig) -> float:
    return cfg.threshold_scale * _otsu(dist, cfg)


@register_threshold("lower")
def _lower(dist: np.ndarray, cfg: BgConfig) -> float:
    t0 = _otsu(dist, cfg)
    below = dist[dist < t0]
    return otsu_of(below, t0) if below.size else t0


@register_threshold("clipped")
def _clipped(dist: np.ndarray, cfg: BgConfig) -> float:
    t0 = _otsu(dist, cfg)
    above = dist[dist > t0]
    hi = float(np.median(above)) if above.size else float(dist.max())
    return otsu_of(np.minimum(dist, hi), hi)


def pick_threshold(dist: np.ndarray, cfg: BgConfig) -> float:
    if cfg.threshold not in THRESHOLDS:
        raise ValueError(f"unknown threshold {cfg.threshold!r}; available: {sorted(THRESHOLDS)}")
    return THRESHOLDS[cfg.threshold](dist, cfg)
