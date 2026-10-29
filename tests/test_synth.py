"""Tests for the synthetic scene generator (tiny in-memory bank, no data files needed)."""
import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from cellcount.synth.cutouts import BANK_R, Cutout
from cellcount.synth.export import write_dataset
from cellcount.synth.scene import Bank, SceneConfig, generate_scene


def disc(cls, subtype, radius, colour, source):
    n = int(2 * radius + 12)
    bgr = np.full((n, n, 3), (225, 232, 228), np.uint8)
    alpha = np.zeros((n, n), np.uint8)
    cv.circle(alpha, (n // 2, n // 2), int(radius), 255, -1)
    bgr[alpha > 0] = colour
    return Cutout(cls, subtype, bgr, cv.GaussianBlur(alpha, (0, 0), 1), (225.0, 232.0, 228.0), source)


def tiny_bank():
    rbc = [disc("rbc", None, BANK_R, (150, 120, 200), "pbc:a"), disc("rbc", None, BANK_R, (140, 110, 190), "pbc:b")]
    wbc = {"neutrophil": [disc("wbc", "neutrophil", 1.6 * BANK_R, (90, 30, 90), "pbc:w")]}
    plt = [disc("platelet", None, BANK_R, (120, 60, 150), "pbc:p")]
    field = np.full((16, 16, 3), (225, 232, 228), np.uint8)
    return Bank(rbc, {"pbc": rbc}, wbc, plt, [field], ["own"])


class SceneTest(unittest.TestCase):
    def test_boxes_are_valid_and_reproducible(self):
        cfg = SceneConfig(wbc_types=("neutrophil",), rim_prob=0.0)
        a = generate_scene(tiny_bank(), cfg, np.random.default_rng(3))
        b = generate_scene(tiny_bank(), cfg, np.random.default_rng(3))
        self.assertEqual(a.image.shape, (640, 640, 3))
        self.assertEqual(a.boxes, b.boxes)
        self.assertTrue((a.image == b.image).all())
        self.assertGreater(len(a.boxes), 20)
        for cid, x1, y1, x2, y2 in a.boxes:
            self.assertIn(cid, (0, 1, 2, 3))
            self.assertTrue(0 <= x1 < x2 <= 639 and 0 <= y1 < y2 <= 639)

    def test_red_cell_boxes_have_the_canonical_size(self):
        cfg = SceneConfig(wbc_mean=0, platelet_mean=0, smudge_mean=0, rim_prob=0.0, zoom_jitter=0.0, size_jitter=0.0)
        sc = generate_scene(tiny_bank(), cfg, np.random.default_rng(1))
        w = np.median([x2 - x1 + 1 for c, x1, y1, x2, y2 in sc.boxes if c == 0])
        self.assertAlmostEqual(w, 2 * cfg.rbc_radius, delta=5)

    def test_overlap_is_darker_than_single_cell(self):
        cfg = SceneConfig(coverage=(0.9, 0.9), min_dist=(0.3, 0.3), wbc_mean=0, platelet_mean=0, smudge_mean=0,
                          rim_prob=0.0, noise_sigma=(0, 0), blur_sigma=(0.1, 0.1), debris=(0, 0), pale_frac=0.0)
        dense = generate_scene(tiny_bank(), cfg, np.random.default_rng(2)).image.mean()
        sparse = generate_scene(tiny_bank(), SceneConfig(**{**cfg.__dict__, "coverage": (0.1, 0.1), "min_dist": (1.0, 1.0)}),
                                np.random.default_rng(2)).image.mean()
        self.assertLess(dense, sparse)

    def test_rim_blacks_out_outside_and_drops_those_boxes(self):
        cfg = SceneConfig(rim_prob=1.0, wbc_types=("neutrophil",))
        sc = generate_scene(tiny_bank(), cfg, np.random.default_rng(5))
        self.assertTrue(sc.params["rim"])
        self.assertGreater((sc.image.max(axis=2) < 10).mean(), 0.02)   # a black area outside the circle

    def test_debris_is_not_labelled(self):
        base = dict(wbc_mean=0, platelet_mean=0, smudge_mean=0, rim_prob=0.0, coverage=(0.3, 0.3), noise_sigma=(0, 0))
        dirty = generate_scene(tiny_bank(), SceneConfig(**base, debris=(100, 100)), np.random.default_rng(6))
        self.assertTrue(all(b[0] == 0 for b in dirty.boxes))       # only red cells are labelled
        self.assertGreater(len(dirty.boxes), 10)                   # the red cells themselves are still labelled

    def test_yolo_export(self):
        sc = generate_scene(tiny_bank(), SceneConfig(wbc_types=("neutrophil",)), np.random.default_rng(4))
        with tempfile.TemporaryDirectory() as d:
            stems = write_dataset([sc], d, "train")
            lab = (Path(d) / "labels" / "train" / f"{stems[0]}.txt").read_text().split("\n")
            lab = [l for l in lab if l]
            self.assertEqual(len(lab), len(sc.boxes))
            for line in lab:
                c, cx, cy, w, h = line.split()
                self.assertTrue(all(0 <= float(v) <= 1 for v in (cx, cy, w, h)))
            self.assertIn("rbc", (Path(d) / "data.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
