"""Build the cut-out bank from the public datasets and show contact sheets to check it by eye.

Run:  python -m cellcount.synth.bank [flags]

Sources
    Data_Set_1 (reference masks)  isolated red cells, scaled by each image's own median cell size
    Data_Set_3 PBC                one WBC (subtype = folder name) or platelet per image, cut out with
                                  our background removal; rejected when merged with neighbours

Backgrounds: the glass colour field that stage 1 estimates (smooth colour plus lighting fall-off, no cells)
is saved as a small image per source photo, so scenes can take their glass look from your own
photos, which have no cell-free areas. Inside an eyepiece circle only the inner square is used.

Flags
    --out DIR            bank folder (default Data_Sets/synth_bank, git-ignored)
    --data-root DIR      folder with Data_Set_1, ... (default $CELLCOUNT_DATA_ROOT or ./Data_Sets)
    --rbc-images N       Data_Set_1 images to harvest red cells from (default 120)
    --rbc-per-image N    most red cells taken from one image (default 25)
    --pbc-per-class N    PBC images tried per class (default 120)
    --backgrounds N      glass fields per public dataset (default 15); own photos are all used (--own-photos)
    --own-photos DIR     folder with the real photos (default <data root>/own_v2/own_v2)
    --seed N             random seed (default 0)
    --no-pbc-rbc         do not take red cells from PBC images (they have a pale centre, unlike Data_Set_1)
    --sheets             also write contact sheets (sheet_<class>.png) into the bank folder
    --show               open the contact sheets in a window as well
    --stats              print how many cut-outs each class and subtype produced
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np

from ..bg_removal import BgConfig, remove_background
from ..common.datasets import get_dataset, resolve_data_root
from dataclasses import replace

from .cutouts import BANK_R, Cutout, extract_central, extract_neighbour_rbcs, extract_rbcs, save_bank

PBC_WBC = ("neutrophil", "eosinophil", "basophil", "lymphocyte", "monocyte", "ig", "erythroblast")


def build_bank(root: Path, rbc_images: int, rbc_per_image: int, pbc_per_class: int, seed: int,
               rbc_from_pbc: bool = True) -> tuple[list[Cutout], Counter]:
    rng = np.random.default_rng(seed)
    cfg = BgConfig()
    cutouts: list[Cutout] = []
    tried: Counter = Counter()

    ds1 = get_dataset("1", root)
    for j in rng.choice(len(ds1), min(rbc_images, len(ds1)), replace=False):
        s = ds1[int(j)]
        bg = remove_background(s.load_image(), cfg).background
        cutouts += extract_rbcs(s, bg, rbc_per_image, rng=rng)

    pbc = get_dataset("3", root)
    for label in PBC_WBC + ("platelet",):
        items = [s for s in pbc if s.label == label]
        for j in rng.choice(len(items), min(pbc_per_class, len(items)), replace=False):
            s = items[int(j)]
            tried[label] += 1
            img = s.load_image()
            res = remove_background(img, cfg)
            # WBC: the stricter 'otsu' cut-off finds the dark core; it is then grown into the cytoplasm inside the wide mask
            wbc_mask = remove_background(img, replace(cfg, threshold="otsu")).mask if label != "platelet" else res.mask
            c = extract_central(s, wbc_mask, res.background, "platelet" if label == "platelet" else "wbc",
                                None if label == "platelet" else label, wide=None if label == "platelet" else res.mask > 0)
            if c is not None:
                cutouts.append(c)
            if rbc_from_pbc and j % 2 == 0:
                cutouts += extract_neighbour_rbcs(s, res.mask, res.background, max_cells=2)
    return cutouts, tried


def build_backgrounds(root: Path, own_dir: Path, per_dataset: int, seed: int, size: int = 64) -> tuple[list[np.ndarray], list[dict]]:
    """Small glass-colour fields (BGR uint8, size x size) and their origin records."""
    rng = np.random.default_rng(seed)
    cfg = BgConfig()
    fields, info = [], []

    def add(field: np.ndarray, source: str, kind: str):
        fields.append(cv.resize(field, (size, size), interpolation=cv.INTER_AREA))
        info.append(dict(source=source, kind=kind))

    for f in sorted(own_dir.glob("unk_*.jpg")) if own_dir.exists() else []:
        res = remove_background(cv.imread(str(f)), cfg)
        if res.fov is None:
            continue
        x, y, w, h = cv.boundingRect(res.fov)
        cx, cy, half = x + w // 2, y + h // 2, int(0.65 * min(w, h) / 2)
        add(res.background[cy - half:cy + half, cx - half:cx + half], f.stem, "own")
    for ds in ("1", "2", "3"):
        samples = get_dataset(ds, root)
        for j in rng.choice(len(samples), min(per_dataset, len(samples)), replace=False):
            s = samples[int(j)]
            add(remove_background(s.load_image(), cfg).background, f"{s.dataset}:{s.name}", "public")
    return fields, info


def contact_sheet(cutouts: list[Cutout], cols: int = 12, cell: int = 96, limit: int = 96) -> np.ndarray:
    """Cut-outs pasted on their own glass colour, each shrunk to fit a square cell."""
    shown = cutouts[:limit]
    rows = max(1, -(-len(shown) // cols))
    sheet = np.full((rows * cell, cols * cell, 3), 128, np.uint8)
    for k, c in enumerate(shown):
        s = min(1.0, (cell - 4) / max(c.bgr.shape[:2]))
        bgr = cv.resize(c.bgr, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
        a = cv.resize(c.alpha, None, fx=s, fy=s, interpolation=cv.INTER_AREA)[..., None] / 255.0
        h, w = bgr.shape[:2]
        tile = np.empty((cell, cell, 3), np.uint8)
        tile[:] = np.array(c.bg_bgr, np.uint8)
        y, x = (cell - h) // 2, (cell - w) // 2
        tile[y:y + h, x:x + w] = (bgr * a + tile[y:y + h, x:x + w] * (1 - a)).astype(np.uint8)
        r, col = divmod(k, cols)
        sheet[r * cell:(r + 1) * cell, col * cell:(col + 1) * cell] = tile
    return sheet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cellcount.synth.bank", description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help="bank folder (default Data_Sets/synth_bank)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--rbc-images", type=int, default=120)
    p.add_argument("--rbc-per-image", type=int, default=25)
    p.add_argument("--pbc-per-class", type=int, default=120)
    p.add_argument("--backgrounds", type=int, default=15)
    p.add_argument("--own-photos", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pbc-rbc", action="store_true", help="skip red cells from PBC images")
    p.add_argument("--sheets", action="store_true", help="write contact sheets into the bank folder")
    p.add_argument("--show", action="store_true", help="open the contact sheets in a window")
    p.add_argument("--stats", action="store_true", help="print counts per class and subtype")
    a = p.parse_args(argv)
    root = resolve_data_root(a.data_root)
    out = Path(a.out) if a.out else root / "synth_bank"
    cutouts, tried = build_bank(root, a.rbc_images, a.rbc_per_image, a.pbc_per_class, a.seed, not a.no_pbc_rbc)
    save_bank(cutouts, out)
    fields, info = build_backgrounds(root, Path(a.own_photos) if a.own_photos else root / "own_v2" / "own_v2", a.backgrounds, a.seed)
    for k, (f, e) in enumerate(zip(fields, info)):
        e["file"] = f"bg_{k:03d}.png"
        cv.imwrite(str(out / e["file"]), f)
    (out / "bg_index.json").write_text(json.dumps(info, indent=1))
    print(f"saved {len(fields)} background fields ({sum(e['kind'] == 'own' for e in info)} from own photos)")
    print(f"saved {len(cutouts)} cut-outs to {out}")
    if a.stats:
        kept = Counter((c.subtype or c.cls) for c in cutouts)
        for k in sorted(kept):
            print(f"  {k:14s} kept {kept[k]:5d}" + (f" of {tried[k]} tried" if k in tried else ""))
    if a.sheets or a.show:
        groups = {"rbc": [c for c in cutouts if c.cls == "rbc"], "platelet": [c for c in cutouts if c.cls == "platelet"]}
        for t in PBC_WBC:
            groups[t] = [c for c in cutouts if c.subtype == t]
        for name, items in groups.items():
            if items:
                cv.imwrite(str(out / f"sheet_{name}.png"), contact_sheet(items))
        print(f"contact sheets in {out}")
        if a.show:
            for name in groups:
                f = out / f"sheet_{name}.png"
                if f.exists():
                    cv.imshow(name, cv.imread(str(f)))
            cv.waitKey(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
