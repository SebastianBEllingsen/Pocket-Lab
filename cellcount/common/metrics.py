"""Mask comparison metrics (pure functions on boolean arrays).

Public API
    iou(pred, ref)     intersection over union, 1.0 when both are empty
    dice(pred, ref)    Dice coefficient, 1.0 when both are empty
"""
from __future__ import annotations

import numpy as np


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = pred.astype(bool), ref.astype(bool)
    union = np.logical_or(pred, ref).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, ref).sum() / union)


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = pred.astype(bool), ref.astype(bool)
    total = pred.sum() + ref.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(pred, ref).sum() / total)
