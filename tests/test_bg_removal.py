"""Tests for the shared metrics and the background-removal pipeline (run: python -m unittest discover -s tests -t .)."""
import unittest

import cv2 as cv
import numpy as np

from cellcount.bg_removal import BgConfig, remove_background
from cellcount.common.metrics import dice, iou


def synthetic(corner_cell: bool = False):
    """Discs on a vignetted background; returns (BGR image, boolean truth mask)."""
    h, w = 240, 320
    yy, xx = np.mgrid[:h, :w]
    vignette = 1.0 - 0.25 * (((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    img = (np.dstack([np.full((h, w), c) for c in (225, 232, 228)]) * vignette[..., None]).astype(np.uint8)
    truth = np.zeros((h, w), np.uint8)
    centres = [(60, 70), (150, 160), (200, 90), (90, 240)] + ([(5, 5)] if corner_cell else [])
    for cy, cx in centres:
        cv.circle(truth, (cx, cy), 25, 255, -1)
    img[truth > 0] = (140, 90, 170)
    cv.circle(img, (160, 150), 8, (225, 232, 228), -1)  # pale hole inside a cell
    return img, truth > 0


def pale_cells_with_dark_nucleus():
    """Pale cells (hard for Otsu) next to one dark blob that pulls Otsu up."""
    img = np.full((240, 320, 3), (225, 232, 228), np.uint8)
    for cy, cx in [(60, 70), (150, 160), (200, 90), (90, 250)]:
        cv.circle(img, (cx, cy), 22, (200, 190, 215), -1)
    cv.circle(img, (160, 60), 14, (90, 30, 90), -1)
    return img


class MetricsTest(unittest.TestCase):
    def test_identical_and_disjoint(self):
        a = np.zeros((10, 10), bool); a[:5] = True
        self.assertEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, ~a), 0.0)
        self.assertEqual(dice(a, a), 1.0)

    def test_both_empty_is_perfect(self):
        e = np.zeros((4, 4), bool)
        self.assertEqual(iou(e, e), 1.0)


class BackgroundRemovalTest(unittest.TestCase):
    def test_methods_find_the_cells(self):
        img, truth = synthetic()
        for method in ("mode", "median"):
            for picker, floor in (("otsu", 0.9), ("lower", 0.8)):   # lower trades some clean-image IoU for pale cells
                r = remove_background(img, BgConfig(method=method, threshold=picker))
                self.assertGreater(iou(r.mask > 0, truth), floor, f"{method}/{picker}")

    def test_cell_touching_corner_does_not_flood_mask(self):
        img, truth = synthetic(corner_cell=True)
        r = remove_background(img)
        self.assertLess(float((r.mask > 0).mean()), 0.6)
        self.assertGreater(iou(r.mask > 0, truth), 0.85)

    def test_trace_records_every_step(self):
        img, _ = synthetic()
        names = [s.name for s in remove_background(img).trace]
        self.assertEqual(names[0], "Input")
        self.assertEqual(names[-1], "Specks removed")
        self.assertIn("Background colour votes", names)          # consensus is the default
        self.assertEqual(len(names), 9)
        self.assertEqual(len(remove_background(img, BgConfig(method="mode")).trace), 9)
        self.assertEqual(len(remove_background(img, BgConfig(method="median")).trace), 8)

    def test_consensus_beats_a_wrong_proposer(self):
        """Cells cover most of the frame, so the most common colour is a cell colour; the others outvote it."""
        img = np.full((240, 320, 3), (225, 232, 228), np.uint8)
        truth = np.zeros((240, 320), np.uint8)
        for cy in range(30, 240, 55):
            for cx in range(30, 320, 55):
                cv.circle(truth, (cx, cy), 29, 255, -1)  # packed, ~60% coverage
        img[truth > 0] = (150, 150, 150)
        r = remove_background(img)
        self.assertGreater(iou(r.mask > 0, truth > 0), 0.85)
        self.assertLess(r.stats["votes_agree"], r.stats["votes_total"] + 1)


if __name__ == "__main__":
    unittest.main()


class ThresholdAndFovTest(unittest.TestCase):
    def test_lower_pickers_keep_pale_cells(self):
        img = pale_cells_with_dark_nucleus()
        fg = {n: remove_background(img, BgConfig(threshold=n)).mask[150, 160] for n in ("otsu", "lower")}
        self.assertEqual(fg["otsu"], 0)          # pale cell missed by the global Otsu cut
        self.assertEqual(fg["lower"], 255)

    def test_unknown_threshold_is_an_error(self):
        with self.assertRaises(ValueError):
            remove_background(pale_cells_with_dark_nucleus(), BgConfig(threshold="nope"))

    def test_fov_circle_excludes_rim(self):
        img = np.zeros((300, 400, 3), np.uint8)
        cv.circle(img, (200, 150), 120, (200, 220, 230), -1)
        cv.circle(img, (200, 150), 15, (140, 90, 170), -1)
        r = remove_background(img)
        self.assertIsNotNone(r.fov)
        self.assertEqual(r.mask[5, 5], 0)        # rim is never foreground
        self.assertEqual(r.mask[150, 200], 255)  # the cell inside the circle is

    def test_no_fov_on_full_frame_image(self):
        img, _ = synthetic()
        self.assertIsNone(remove_background(img).fov)
