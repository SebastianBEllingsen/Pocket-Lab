"""Background estimators. Add a new one by writing a function and decorating it.

Each estimator takes (BGR image, BgConfig) and returns a ``BackgroundEstimate``:
the per-pixel background colour in OpenCV 8-bit Lab scale, the reference colour,
a short description for the viewer, the parameter values it actually used, and
(for voting methods) the colour proposals and which of them agreed.

Public API
    BackgroundEstimate
    register_method(name)    decorator that adds an estimator to the registry
    METHODS                  name -> estimator
    estimate(img, cfg)       run the estimator chosen by cfg.method

Estimators
    consensus  several independent proposers (see proposers.py) each name the background
               colour; the colour most of them agree on wins. If none agree, the first
               proposer in cfg.proposers is used and the result is flagged. Default.
    mode       the single "most common colour" proposer, then the same field estimate.
    median     large median filter of a downscaled image. Simple, but wrong when cells
               cover most of the area.

From a reference colour, the background *field* is built by smoothing only the pixels that
match it, so slow lighting changes (vignetting) are followed while cells are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2 as cv
import numpy as np

from .config import BgConfig
from .proposers import Proposal, propose


@dataclass
class BackgroundEstimate:
    lab: np.ndarray                       # HxWx3 float32, OpenCV 8-bit Lab scale
    reference_lab: tuple[float, float, float]
    caption: str
    values: dict = field(default_factory=dict)
    proposals: list[Proposal] = field(default_factory=list)
    agreeing: list[bool] = field(default_factory=list)   # per proposal: part of the winning group
    chosen: int | None = None                            # index of the proposal that seeded the result
    confident: bool = True


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


def _field_from_reference(img: np.ndarray, small_lab: np.ndarray, ref: np.ndarray, cfg: BgConfig):
    """Smooth background field from the pixels that match ``ref`` (see module docstring)."""
    d = np.linalg.norm(small_lab - ref, axis=2)
    near = d < cfg.agree_tol
    if near.sum() < 50:                       # reference colour barely present: take the closest 2%
        near = d <= np.quantile(d, 0.02)
    d_sel = d[near]
    spread = 1.4826 * np.median(np.abs(d_sel - np.median(d_sel))) + 1e-3
    limit = max(float(np.median(d_sel) + cfg.tol * spread), 4.0)
    support = (d < limit).astype(np.float32)
    sigma = max(small_lab.shape[:2]) * cfg.sigma_frac
    num = cv.GaussianBlur(small_lab * support[..., None], (0, 0), sigma)
    den = cv.GaussianBlur(support, (0, 0), sigma)[..., None]
    field_ = np.where(den < 1e-2, ref, num / np.maximum(den, 1e-3))  # no support -> reference colour
    values = {"scale": cfg.scale, "match limit (Lab dist)": round(limit, 1),
              "background pixels (% of work image)": round(100 * float(support.mean()), 1)}
    return _upsample(field_.astype(np.float32), img.shape[:2]), sigma, values


def _small_lab(img: np.ndarray, cfg: BgConfig) -> np.ndarray:
    return cv.cvtColor(_small(img, cfg.scale), cv.COLOR_BGR2LAB).astype(np.float32)


@register_method("mode")
def _mode(img: np.ndarray, cfg: BgConfig) -> BackgroundEstimate:
    lab = _small_lab(img, cfg)
    p = propose("mode", lab, cfg)
    field_, sigma, values = _field_from_reference(img, lab, np.array(p.lab, np.float32), cfg)
    return BackgroundEstimate(
        lab=field_, reference_lab=p.lab,
        caption=f"Estimated background (mode): most common colour, smoothed over matching pixels (sigma {sigma:.0f}px)",
        values=values, proposals=[p], agreeing=[True], chosen=0)


@register_method("consensus")
def _consensus(img: np.ndarray, cfg: BgConfig) -> BackgroundEstimate:
    lab = _small_lab(img, cfg)
    props = [propose(n, lab, cfg) for n in cfg.proposers]
    colours = np.array([p.lab for p in props], np.float32)
    close = np.linalg.norm(colours[:, None] - colours[None], axis=2) <= cfg.agree_tol
    votes = close.sum(axis=1)
    best = int(votes.argmax())                 # first proposer wins ties (cfg.proposers order)
    confident = votes[best] >= 2
    agreeing = [bool(x) for x in close[best]] if confident else [i == 0 for i in range(len(props))]
    chosen = best if confident else 0
    ref = np.median(colours[agreeing], axis=0)
    field_, sigma, values = _field_from_reference(img, lab, ref, cfg)
    note = (f"{int(votes[best])} of {len(props)} proposers agree" if confident
            else "NO AGREEMENT between proposers: fell back to the first one")
    values = {"votes for winner": f"{int(votes[best])}/{len(props)}", **values}
    return BackgroundEstimate(
        lab=field_, reference_lab=tuple(float(v) for v in ref),
        caption=f"Estimated background (consensus): {note}; field smoothed over matching pixels (sigma {sigma:.0f}px)",
        values=values, proposals=props, agreeing=agreeing, chosen=chosen, confident=bool(confident))


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
