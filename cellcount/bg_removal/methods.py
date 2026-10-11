"""Background estimators. Add a new one by writing a function and decorating it.

Each estimator takes (BGR image, BgConfig) and returns a ``BackgroundEstimate``:
the per-pixel background colour in OpenCV 8-bit Lab scale, the reference colour,
a short description for the viewer, and the parameter values it actually used.

Public API
    BackgroundEstimate
    register_method(name)    decorator that adds an estimator to the registry
    METHODS                  name -> estimator
    estimate(img, cfg)       run the estimator chosen by cfg.method

Estimators
    mode    most common colour = background; weighted smoothing of the pixels that match it,
            so the field follows vignetting while cells are ignored. Robust when cells cover
            much of the frame (BCCD).
    median  large median filter of a downscaled image. Simple, but wrong when cells cover
            most of the area.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2 as cv
import numpy as np

from .config import BgConfig


@dataclass
class BackgroundEstimate:
    lab: np.ndarray                       # HxWx3 float32, OpenCV 8-bit Lab scale
    reference_lab: tuple[float, float, float]
    caption: str
    values: dict = field(default_factory=dict)


METHODS: dict[str, Callable[[np.ndarray, BgConfig], BackgroundEstimate]] = {}


def register_method(name: str):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def estimate(img: np.ndarray, cfg: BgConfig) -> BackgroundEstimate:
    if cfg.method not in METHODS:
        raise ValueError(f"Unknown method {cfg.method!r}; choose from {sorted(METHODS)}")
    return METHODS[cfg.method](img, cfg)


def _small(img: np.ndarray, scale: float) -> np.ndarray:
    return cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)


def _upsample(field_: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv.resize(field_, (shape[1], shape[0]), interpolation=cv.INTER_LINEAR)


@register_method("mode")
def _mode(img: np.ndarray, cfg: BgConfig) -> BackgroundEstimate:
    small = _small(img, cfg.scale)
    lab = cv.cvtColor(small, cv.COLOR_BGR2LAB).astype(np.float32)
    q = (lab // (256 / cfg.bins)).astype(int)
    key = q[..., 0] * cfg.bins ** 2 + q[..., 1] * cfg.bins + q[..., 2]
    sel = key == np.bincount(key.ravel()).argmax()
    ref = np.median(lab[sel], axis=0)
    d = np.linalg.norm(lab - ref, axis=2)
    d_sel = d[sel]
    spread = 1.4826 * np.median(np.abs(d_sel - np.median(d_sel))) + 1e-3
    limit = max(float(np.median(d_sel) + cfg.tol * spread), 4.0)
    support = (d < limit).astype(np.float32)
    sigma = max(small.shape[:2]) * cfg.sigma_frac
    num = cv.GaussianBlur(lab * support[..., None], (0, 0), sigma)
    den = cv.GaussianBlur(support, (0, 0), sigma)[..., None]
    field_ = np.where(den < 1e-2, ref, num / np.maximum(den, 1e-3))  # no support -> global colour
    return BackgroundEstimate(
        lab=_upsample(field_.astype(np.float32), img.shape[:2]),
        reference_lab=tuple(float(v) for v in ref),
        caption=f"Estimated background (mode): dominant colour, then smoothed over matching pixels (sigma {sigma:.0f}px)",
        values={"scale": cfg.scale, "bins": cfg.bins, "tol": cfg.tol, "match limit (Lab dist)": round(limit, 1),
                "background pixels (% of work image)": round(100 * float(support.mean()), 1)},
    )


@register_method("median")
def _median(img: np.ndarray, cfg: BgConfig) -> BackgroundEstimate:
    small = _small(img, cfg.scale)
    k = max(5, int(max(small.shape[:2]) * cfg.kernel_frac) | 1)
    k = min(k, 255)
    bg = cv.GaussianBlur(cv.medianBlur(small, k), (0, 0), k / 6)
    lab = cv.cvtColor(bg, cv.COLOR_BGR2LAB).astype(np.float32)
    return BackgroundEstimate(
        lab=_upsample(lab, img.shape[:2]),
        reference_lab=tuple(float(v) for v in np.median(lab.reshape(-1, 3), axis=0)),
        caption=f"Estimated background (median filter, kernel {k}px on a {cfg.scale:g}x copy)",
        values={"scale": cfg.scale, "kernel": k},
    )
