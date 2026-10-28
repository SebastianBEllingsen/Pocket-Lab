"""
Convert public datasets into one YOLO dataset in our class order
(0 rbc, 1 wbc, 2 platelet, 3 smudge).

    python scripts/prepare_public.py                   # every dataset with a reader
    python scripts/prepare_public.py --only bbbc041    # one dataset, then re-merge all

Per dataset: data/public/<name>/yolo/{images,labels}/{train,val,test}/ + manifest.csv
Merged:      data/merged/{images,labels}/..., data.yaml, manifest.csv, previews/
"""
import argparse, csv, glob, hashlib, json, os, random, shutil, sys
from collections import Counter, defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # project root: count_cells.py
sys.path.insert(0, HERE)
from public_common import CLASSES, TilePacker, draw_preview, pieces, rbc_radius_from_boxes, rescale  # noqa: E402
from count_cells import foreground  # noqa: E402
PUBLIC, MERGED = "data/public", "data/merged"
SPLITS = ("train", "val", "test")


def frac(name):
    """Stable number in [0, 1) from a name, for reproducible splits."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 / 1000


# ------------------------------------------------------------------ readers
# Each reader yields (image name, BGR image, split, [(cls, x1, y1, x2, y2) in raw px]).

TXL_TO_OURS = {0: 1, 1: 0, 2: 2}                           # TXL: 0 WBC, 1 RBC, 2 Platelets


def read_txl_pbc(raw):
    for split in SPLITS:
        for path in sorted(glob.glob(f"{raw}/images/{split}/*.png")):
            img = cv2.imread(path)
            h, w = img.shape[:2]
            name = os.path.splitext(os.path.basename(path))[0]
            boxes = []
            for line in open(f"{raw}/labels/{split}/{name}.txt").read().splitlines():
                if line.strip():
                    k, xc, yc, bw, bh = line.split()
                    xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
                    boxes.append((TXL_TO_OURS[int(k)], xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
            yield name, img, split, boxes


BBBC_TO_OURS = {"red blood cell": 0, "ring": 0, "trophozoite": 0, "schizont": 0,
                "gametocyte": 0, "leukocyte": 1}           # "difficult" is painted out


def paint_out(img, mask):
    """Fill the masked pixels in from their surroundings (OpenCV inpainting),
    so hidden cells look like empty glass instead of flat grey rectangles."""
    if mask.any():
        img = cv2.inpaint(img, cv2.dilate(mask, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
    return img


def read_bbbc041(raw):
    root = f"{raw}/malaria"
    for js, official in (("training.json", "train"), ("test.json", "test")):
        for entry in json.load(open(f"{root}/{js}")):
            img = cv2.imread(root + entry["image"]["pathname"])
            h, w = img.shape[:2]
            name = os.path.splitext(os.path.basename(entry["image"]["pathname"]))[0]
            split = official if official == "test" else ("val" if frac(name) < 0.1 else "train")
            boxes, hide = [], np.zeros((h, w), np.uint8)
            for o in entry["objects"]:
                bb = o["bounding_box"]                      # r = row (y), c = column (x)
                x1, y1, x2, y2 = bb["minimum"]["c"], bb["minimum"]["r"], bb["maximum"]["c"], bb["maximum"]["r"]
                if o["category"] == "difficult":
                    cv2.rectangle(hide, (x1, y1), (x2, y2), 255, -1)
                else:
                    boxes.append((BBBC_TO_OURS[o["category"]], x1, y1, x2, y2))

            # Visible cells without a box: hide them too, so YOLO never sees an unlabelled cell
            r = rbc_radius_from_boxes(boxes, w, h)
            if r:
                covered = np.zeros((h, w), np.uint8)
                for c, bx1, by1, bx2, by2 in boxes:
                    cv2.rectangle(covered, (bx1, by1), (bx2, by2), 255, -1)
                fg = foreground(img, np.full((h, w), 255, np.uint8))
                n, lbl, st, _ = cv2.connectedComponentsWithStats(fg)
                for i in range(1, n):
                    x, y, bw, bh, area = st[i]
                    if area < 0.5 * np.pi * r * r:          # smaller than half a red cell
                        continue
                    comp = lbl[y:y + bh, x:x + bw] == i
                    if (covered[y:y + bh, x:x + bw][comp] > 0).mean() <= 0.3:
                        hide[y:y + bh, x:x + bw][comp] = 255
            yield name, paint_out(img, hide), split, boxes



READERS = {"txl_pbc": read_txl_pbc, "bbbc041": read_bbbc041}

# Blob radius / expert-box radius, measured on train+val images per source and image size.
# Converts the box radius into the blob radius that normalise() uses for our own photos.
BOX_TO_BLOB = {
    ("txl_pbc", "256x256"): 0.95, ("txl_pbc", "360x363"): 0.90,
    ("txl_pbc", "366x369"): 0.90,       # only 3 round cells measured; same source as 360x363
    ("txl_pbc", "575x575"): 0.88, ("txl_pbc", "640x480"): 0.91,
    ("bbbc041", "1600x1200"): 0.94,
    ("bbbc041", "1944x1383"): 0.94,     # test-only size: training factor, never measured on test
}


# ------------------------------------------------------------------ conversion
def convert(source, reader):
    out = f"{PUBLIC}/{source}/yolo"
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("images", "labels"):
        for sp in SPLITS:
            os.makedirs(f"{out}/{sub}/{sp}")
    packers = {sp: TilePacker(out, sp, source) for sp in SPLITS}
    rows, radii, skipped, missing = [], defaultdict(list), 0, set()
    for name, img, split, boxes in reader(f"{PUBLIC}/{source}/raw"):
        h, w = img.shape[:2]
        r = rbc_radius_from_boxes(boxes, w, h)
        if r is None:                                       # no full RBC box to measure size
            skipped += 1
            continue
        size = f"{w}x{h}"
        if (source, size) not in BOX_TO_BLOB:
            missing.add(size)
        r *= BOX_TO_BLOB.get((source, size), 1.0)           # box radius -> blob radius
        radii[size].append(r)
        small, sboxes, s = rescale(img, boxes, r)
        tiles = [packers[split].add(p, b, name) for p, b in pieces(small, sboxes)]
        n = Counter(c for c, *_ in boxes)
        rows.append(dict(image=name, source=source, split=split, pseudo_rbc=False,
                         rbc_radius_px=round(r, 1), scale=round(s, 3),
                         **{f"n_{c}": n[i] for i, c in enumerate(CLASSES)},
                         tiles=" ".join(sorted(set(tiles)))))
    for p in packers.values():
        p.flush()
    with open(f"{out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"{source}: {len(rows)} images converted, {skipped} skipped (no full RBC box)")
    if missing:
        print(f"  WARNING: no box->blob factor for sizes {sorted(missing)}; used 1.0")
    for size, rs in sorted(radii.items()):
        print(f"  {size:10} {len(rs):5} images  RBC radius median {np.median(rs):5.1f} px")
    for sp in SPLITS:
        print(f"  {sp:5}: {packers[sp].n} tiles")




def merge(repeat):
    """Rebuild data/merged from every converted dataset in data/public/*/yolo.
    repeat = {source: n}: copy that source's TRAIN tiles n times in total (never val/test)."""
    shutil.rmtree(MERGED, ignore_errors=True)
    for sub in ("images", "labels"):
        for sp in SPLITS:
            os.makedirs(f"{MERGED}/{sub}/{sp}")
    os.makedirs(f"{MERGED}/previews")
    rows = []
    for out in sorted(glob.glob(f"{PUBLIC}/*/yolo")):
        source = out.split("/")[-2]
        rows += list(csv.DictReader(open(f"{out}/manifest.csv")))
        stems = []
        for sp in SPLITS:
            for img in glob.glob(f"{out}/images/{sp}/*.jpg"):
                stem = os.path.splitext(os.path.basename(img))[0]
                shutil.copy2(img, f"{MERGED}/images/{sp}/")
                shutil.copy2(f"{out}/labels/{sp}/{stem}.txt", f"{MERGED}/labels/{sp}/")
                stems.append((sp, stem))
                for j in range(1, repeat.get(source, 1) if sp == "train" else 1):
                    shutil.copy2(img, f"{MERGED}/images/train/{stem}_r{j}.jpg")
                    shutil.copy2(f"{out}/labels/train/{stem}.txt", f"{MERGED}/labels/train/{stem}_r{j}.txt")
        for sp, stem in random.Random(0).sample(sorted(stems), min(10, len(stems))):
            draw_preview(f"{MERGED}/images/{sp}/{stem}.jpg", f"{MERGED}/labels/{sp}/{stem}.txt",
                         f"{MERGED}/previews/{stem}.jpg")
    with open(f"{MERGED}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(f"{MERGED}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(MERGED)}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print(f"merged: {len(rows)} source images ->",
          ", ".join(f"{sp} {len(os.listdir(f'{MERGED}/images/{sp}'))} tiles" for sp in SPLITS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(READERS), help="convert one dataset, then re-merge all")
    ap.add_argument("--repeat", nargs="*", default=["txl_pbc=4"],
                    help="copy a source's TRAIN tiles N times, e.g. txl_pbc=4")
    a = ap.parse_args()
    for source in ([a.only] if a.only else sorted(READERS)):
        convert(source, READERS[source])
    merge({k: int(v) for k, v in (r.split("=") for r in a.repeat)})
