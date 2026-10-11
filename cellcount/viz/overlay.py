"""Build transparent RGBA layers (outlines, boxes, difference maps) at image size.

Public API
    outline_rgba(mask, color, thickness)           contour lines of a boolean mask
    boxes_rgba(shape, boxes, colors, thickness)    rectangles; ``boxes`` is any objects with
                                                   label/xmin/ymin/xmax/ymax, ``colors`` maps label -> RGB
    diff_rgba(ours, ref, ours_only, ref_only)      tint pixels where two masks disagree
"""
from __future__ import annotations

from typing import Iterable, Mapping

import cv2 as cv
import numpy as np

RGB = tuple[int, int, int]


def outline_rgba(mask: np.ndarray, color: RGB, thickness: int = 2) -> np.ndarray:
    out = np.zeros((*mask.shape, 4), np.uint8)
    contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    layer = np.zeros(mask.shape, np.uint8)
    cv.drawContours(layer, contours, -1, 255, thickness)
    out[layer > 0] = (*color, 255)
    return out


def boxes_rgba(shape: tuple[int, int], boxes: Iterable, colors: Mapping[str, RGB],
               thickness: int = 2) -> np.ndarray:
    out = np.zeros((*shape, 4), np.uint8)
    for b in boxes:
        c = colors.get(b.label, (255, 255, 255))
        cv.rectangle(out, (b.xmin, b.ymin), (b.xmax, b.ymax), (*c, 255), thickness)
    return out


def diff_rgba(ours: np.ndarray, ref: np.ndarray, ours_only: RGB, ref_only: RGB,
              alpha: int = 140) -> np.ndarray:
    ours, ref = ours.astype(bool), ref.astype(bool)
    out = np.zeros((*ours.shape, 4), np.uint8)
    out[ours & ~ref] = (*ours_only, alpha)
    out[ref & ~ours] = (*ref_only, alpha)
    return out
