"""Background-colour proposers: independent opinions on "what colour is the background?".

Each proposer looks at a small Lab image and returns one ``Proposal`` (a colour plus
a short note). The consensus estimator in methods.py lets them vote. Add a proposer
by writing a function and decorating it with ``@register_proposer("name")``.

Public API
    Proposal                 name, Lab colour (OpenCV 8-bit scale), note
    PROPOSERS                name -> proposer
    register_proposer(name)  decorator
    propose(name, lab, cfg)  run one proposer

Proposers
    mode     most common colour (fails when cells cover most of the frame)
    bright   lightest colour among clusters that cover enough of the image
    kmeans   lightest centre of a 3-cluster k-means on the colours
    border   median colour of the image frame (fails when cells touch the border)
    neutral  least colourful cluster among those that cover enough of the image
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2 as cv
import numpy as np

from .config import BgConfig

Lab = tuple[float, float, float]


@dataclass(frozen=True)
class Proposal:
    name: str
    lab: Lab
    note: str = ""


PROPOSERS: dict[str, Callable[[np.ndarray, BgConfig], Proposal]] = {}


def register_proposer(name: str):
    def deco(fn):
        PROPOSERS[name] = fn
        return fn
    return deco


def propose(name: str, lab: np.ndarray, cfg: BgConfig) -> Proposal:
    if name not in PROPOSERS:
        raise ValueError(f"Unknown proposer {name!r}; choose from {sorted(PROPOSERS)}")
    return PROPOSERS[name](lab, cfg)


def _tup(v) -> Lab:
    return (float(v[0]), float(v[1]), float(v[2]))


def _bins(lab: np.ndarray, cfg: BgConfig):
    """Histogram clusters: (labels per pixel, counts, ids of clusters large enough)."""
    q = (lab // (256 / cfg.bins)).astype(int)
    key = q[..., 0] * cfg.bins ** 2 + q[..., 1] * cfg.bins + q[..., 2]
    counts = np.bincount(key.ravel())
    big = np.flatnonzero(counts >= cfg.min_cluster_frac * key.size)
    return key, counts, big


@register_proposer("mode")
def _mode(lab, cfg):
    key, counts, _ = _bins(lab, cfg)
    k = counts.argmax()
    return Proposal("mode", _tup(np.median(lab[key == k], axis=0)), f"most common colour ({100 * counts[k] / key.size:.0f}% of image)")


@register_proposer("bright")
def _bright(lab, cfg):
    key, counts, big = _bins(lab, cfg)
    if len(big) == 0:
        big = np.array([counts.argmax()])
    best = max(big, key=lambda k: float(lab[key == k][:, 0].mean()))
    return Proposal("bright", _tup(np.median(lab[key == best], axis=0)), "lightest large colour cluster")


@register_proposer("neutral")
def _neutral(lab, cfg):
    key, counts, big = _bins(lab, cfg)
    if len(big) == 0:
        big = np.array([counts.argmax()])

    def chroma(k):
        c = np.median(lab[key == k], axis=0)
        return float(np.hypot(c[1] - 128, c[2] - 128))

    best = min(big, key=chroma)
    return Proposal("neutral", _tup(np.median(lab[key == best], axis=0)), "least colourful large cluster")


@register_proposer("kmeans")
def _kmeans(lab, cfg):
    px = lab.reshape(-1, 3)
    px = px[:: max(1, len(px) // 20000)].astype(np.float32)
    cv.setRNGSeed(0)  # deterministic
    _, labels, centres = cv.kmeans(px, 3, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                                   3, cv.KMEANS_PP_CENTERS)
    share = np.bincount(labels.ravel(), minlength=3) / len(px)
    ok = [i for i in range(3) if share[i] >= cfg.min_cluster_frac] or [int(share.argmax())]
    best = max(ok, key=lambda i: centres[i][0])
    return Proposal("kmeans", _tup(centres[best]), f"lightest of 3 k-means clusters ({100 * share[best]:.0f}% of image)")


@register_proposer("border")
def _border(lab, cfg):
    h, w = lab.shape[:2]
    t = max(1, int(min(h, w) * cfg.border_frac))
    frame = np.concatenate([lab[:t].reshape(-1, 3), lab[-t:].reshape(-1, 3),
                            lab[t:-t, :t].reshape(-1, 3), lab[t:-t, -t:].reshape(-1, 3)])
    return Proposal("border", _tup(np.median(frame, axis=0)), "median colour of the image frame")
