"""normalise() must bring red cells to radius R_CANON within 2 px."""
import glob
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from count_cells import R_CANON, count_masks, foreground, normalise, round_blob_radii  # noqa: E402


def find_image(env, pattern):
    path = os.environ.get(env) or next(iter(sorted(glob.glob(pattern, recursive=True))), None)
    if not path or not os.path.exists(path):
        pytest.skip(f"no image: set {env} or add files matching {pattern}")
    return path


def radius_after_normalise(nrm):
    field, _ = count_masks(nrm)
    mask = cv2.morphologyEx(foreground(nrm["img"], field), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    return float(np.median(round_blob_radii(mask)))


def test_own_photo_with_fov():
    nrm = normalise(cv2.imread(find_image("OWN_PHOTO", "data/own/*.jpg")))
    assert nrm["has_fov"]
    assert abs(radius_after_normalise(nrm) - R_CANON) <= 2


def test_public_crop_without_fov():
    nrm = normalise(cv2.imread(find_image("PUBLIC_CROP", "data/public/*/raw/**/*.[jp][pn]g")))
    assert not nrm["has_fov"]
    assert abs(radius_after_normalise(nrm) - R_CANON) <= 2
