"""Parameters of the background-removal stage (one frozen dataclass, no magic numbers).

``BgConfig`` is the single source of truth for defaults; the CLI builds its flags
from these fields, so changing a default here changes it everywhere.

Fields
    method         background estimator: "consensus" (default), "mode" or "median"
    proposers      consensus: which colour proposers vote, in tie-break order
    agree_tol      consensus: two proposals agree if their Lab colours are within this distance
    min_cluster_frac  proposers: a colour cluster must cover this fraction of the image to count
    border_frac    "border" proposer: width of the image frame sampled, as a fraction of image size
    scale          working resolution as a fraction of the image (speed vs. detail)
    bins           Lab histogram bins per axis used by the colour proposers
    tol            how many robust sigmas around the background colour count as background
    sigma_frac     smoothing of the background field, as a fraction of image size
    kernel_frac    median method: median kernel size as a fraction of image size
    blur           Gaussian sigma (px) applied to the distance map before thresholding
    min_area_frac  connected blobs smaller than this fraction of the image are removed
    fill_holes     fill enclosed holes (pale cell centres)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BgConfig:
    method: str = "consensus"
    proposers: tuple[str, ...] = ("mode", "bright", "kmeans", "border", "neutral")
    agree_tol: float = 12.0
    min_cluster_frac: float = 0.05
    border_frac: float = 0.03
    scale: float = 0.25
    bins: int = 16
    tol: float = 3.0
    sigma_frac: float = 0.125
    kernel_frac: float = 0.25
    blur: float = 2.0
    min_area_frac: float = 2e-4
    fill_holes: bool = True
