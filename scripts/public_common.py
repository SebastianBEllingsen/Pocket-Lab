"""
Shared helpers for converting public datasets into our YOLO format.

Every image is rescaled so a red blood cell has radius R_CANON (16 px), like our
own photos. For labelled datasets the RBC radius comes from the expert RBC boxes,
which works even in images with too few isolated round cells for normalise().
Rescaled images are packed side by side onto 640 x 640 tiles.
"""
import os

import cv2
import numpy as np

from count_cells import R_CANON

CLASSES = ["rbc", "wbc", "platelet", "smudge"]
COLORS = [(0, 200, 0), (255, 60, 0), (0, 140, 255), (200, 0, 200)]
TILE, STRIDE = 640, 512
GAP = 4                  # px of black between packed images
MAX_UPSCALE = 4.0


def rbc_radius_from_boxes(boxes, w, h):
    """Median RBC radius in raw px from (cls, x1, y1, x2, y2) boxes not cut by the image edge."""
    r = [((x2 - x1) + (y2 - y1)) / 4 for c, x1, y1, x2, y2 in boxes
         if c == 0 and x1 > 1 and y1 > 1 and x2 < w - 2 and y2 < h - 2]
    return float(np.median(r)) if r else None


def rescale(img, boxes, rbc_radius):
    """Rescale image and boxes so the RBC radius becomes R_CANON (never upscale more than 4x)."""
    s = min(R_CANON / rbc_radius, MAX_UPSCALE)
    out = cv2.resize(img, None, fx=s, fy=s,
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out, [(c, x1 * s, y1 * s, x2 * s, y2 * s) for c, x1, y1, x2, y2 in boxes], s


def pieces(img, boxes):
    """Cut an image larger than a tile into TILE x TILE pieces (stride 512), keeping
    boxes that are at least 60 % inside a piece. Small images stay whole."""
    h, w = img.shape[:2]
    if h <= TILE and w <= TILE:
        return [(img, boxes)]
    xs = list(range(0, max(w - TILE, 0) + 1, STRIDE)) or [0]
    ys = list(range(0, max(h - TILE, 0) + 1, STRIDE)) or [0]
    if xs[-1] + TILE < w: xs.append(w - TILE)
    if ys[-1] + TILE < h: ys.append(h - TILE)
    out = []
    for ty in ys:
        for tx in xs:
            kept = []
            for c, x1, y1, x2, y2 in boxes:
                cx1, cy1 = max(x1, tx), max(y1, ty)
                cx2, cy2 = min(x2, tx + TILE), min(y2, ty + TILE)
                if cx2 > cx1 and cy2 > cy1 and (cx2 - cx1) * (cy2 - cy1) >= 0.6 * (x2 - x1) * (y2 - y1):
                    kept.append((c, cx1 - tx, cy1 - ty, cx2 - tx, cy2 - ty))
            out.append((img[ty:ty + TILE, tx:tx + TILE], kept))
    return out


class TilePacker:
    """Places images side by side (row by row) on black TILE x TILE tiles and
    writes every full tile with its YOLO labels."""

    def __init__(self, out_dir, split, prefix):
        self.out, self.split, self.prefix, self.n = out_dir, split, prefix, 0
        self._reset()

    def _reset(self):
        self.canvas = np.zeros((TILE, TILE, 3), np.uint8)
        self.labels, self.members = [], []
        self.x = self.y = self.row_h = 0

    def name(self):
        return f"{self.prefix}_{self.split}_{self.n:05d}"

    def add(self, img, boxes, member):
        """Add one image (at most TILE x TILE); returns the name of the tile it lands on."""
        h, w = img.shape[:2]
        if self.x + w > TILE:                               # start a new row
            self.x, self.y, self.row_h = 0, self.y + self.row_h + GAP, 0
        if self.y + h > TILE:                               # tile full
            self.flush()
        self.canvas[self.y:self.y + h, self.x:self.x + w] = img
        for c, x1, y1, x2, y2 in boxes:
            x1, x2 = np.clip([x1, x2], 0, w) + self.x
            y1, y2 = np.clip([y1, y2], 0, h) + self.y
            if x2 - x1 > 1 and y2 - y1 > 1:
                self.labels.append((c, x1, y1, x2, y2))
        self.members.append(member)
        tile = self.name()
        self.x += w + GAP
        self.row_h = max(self.row_h, h)
        return tile

    def flush(self):
        if self.members:
            stem = self.name()
            cv2.imwrite(os.path.join(self.out, "images", self.split, stem + ".jpg"), self.canvas)
            with open(os.path.join(self.out, "labels", self.split, stem + ".txt"), "w") as f:
                f.write("\n".join(
                    f"{c} {(x1 + x2) / 2 / TILE:.6f} {(y1 + y2) / 2 / TILE:.6f} "
                    f"{(x2 - x1) / TILE:.6f} {(y2 - y1) / TILE:.6f}"
                    for c, x1, y1, x2, y2 in self.labels))
            self.n += 1
        self._reset()


def draw_preview(img_path, label_path, out_path):
    """Draw the YOLO labels of one tile, for a visual check."""
    img = cv2.imread(img_path)
    for line in open(label_path).read().splitlines():
        if line.strip():
            c, xc, yc, bw, bh = line.split()
            xc, yc, bw, bh = (float(v) * TILE for v in (xc, yc, bw, bh))
            cv2.rectangle(img, (int(xc - bw / 2), int(yc - bh / 2)),
                          (int(xc + bw / 2), int(yc + bh / 2)), COLORS[int(c)], 1)
    cv2.imwrite(out_path, img)