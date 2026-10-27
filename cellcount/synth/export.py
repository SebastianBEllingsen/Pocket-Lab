"""Write synthetic scenes as a YOLO dataset (the teammate's format) plus sidecar files.

Public API
    write_dataset(scenes, out_dir, split, start=0, names=CLASSES) -> list[str]   image stems written
    box_layers(scene, thickness)    one RGBA box layer per class (viewer / previews)

Layout (same as the detector's `make_yolo_dataset.py`)
    <out>/images/<split>/syn_00000.jpg      image, 640x640 tile at red-cell radius 16 px
    <out>/labels/<split>/syn_00000.txt      one line per cell: class cx cy w h (normalised)
    <out>/data.yaml                         class names in the detector's order (0 rbc, 1 wbc, 2 platelet, 3 smudge)
    <out>/wbc_types.csv                     image, line number, WBC type of each wbc box (sidecar for the type model)
    <out>/manifest.csv                      image, seed-independent scene parameters (coverage, zoom, stain, ...)
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2 as cv
import numpy as np

from ..viz import Layer
from .scene import CLASSES, Scene

COLORS = {"rbc": (0, 200, 0), "wbc": (60, 60, 255), "platelet": (255, 140, 0), "smudge": (200, 0, 200)}   # RGB, as the teammate's viewer


def write_dataset(scenes: list[Scene], out_dir: str | Path, split: str = "train", start: int = 0,
                  names: tuple[str, ...] = CLASSES) -> list[str]:
    out = Path(out_dir)
    for sub in ("images", "labels"):
        (out / sub / split).mkdir(parents=True, exist_ok=True)
    stems = []
    types_path, man_path = out / "wbc_types.csv", out / "manifest.csv"
    new_types, new_man = not types_path.exists(), not man_path.exists()
    with open(types_path, "a", newline="") as ft, open(man_path, "a", newline="") as fm:
        wt, wm = csv.writer(ft), csv.writer(fm)
        if new_types:
            wt.writerow(["image", "line", "wbc_type"])
        if new_man:
            wm.writerow(["image", "split", "n_boxes"] + list(scenes[0].params) if scenes else ["image"])
        for k, sc in enumerate(scenes):
            stem = f"syn_{start + k:05d}"
            h, w = sc.image.shape[:2]
            cv.imwrite(str(out / "images" / split / f"{stem}.jpg"), sc.image, [cv.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            for j, ((cid, x1, y1, x2, y2), sub) in enumerate(zip(sc.boxes, sc.subtypes)):
                lines.append(f"{cid} {(x1 + x2 + 1) / 2 / w:.6f} {(y1 + y2 + 1) / 2 / h:.6f} {(x2 - x1 + 1) / w:.6f} {(y2 - y1 + 1) / h:.6f}")
                if sub:
                    wt.writerow([stem, j, sub])
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            wm.writerow([stem, split, len(sc.boxes)] + list(sc.params.values()))
            stems.append(stem)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    return stems


def box_layers(scene: Scene, thickness: int = 1, visible: bool = True) -> list[Layer]:
    """One layer per class so the viewer can toggle r/w/p/s (rbc, wbc, platelet, smudge)."""
    h, w = scene.image.shape[:2]
    layers = []
    for cid, name in enumerate(CLASSES):
        rgba = np.zeros((h, w, 4), np.uint8)
        n = 0
        for c, x1, y1, x2, y2 in scene.boxes:
            if c == cid:
                cv.rectangle(rgba, (x1, y1), (x2, y2), (*COLORS[name], 255), thickness)
                n += 1
        layers.append(Layer(name, name[0], rgba, COLORS[name], f"{name} boxes ({n})", visible))
    return layers
