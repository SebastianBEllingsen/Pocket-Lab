"""
Pick random squares in the test photos for hand-counting red cells.

    python scripts/make_count_crops.py data/own_v2/unk_40x_08.jpg ... --out data/own_gt

For every photo, N squares (default 4) are drawn at random (fixed seed) inside the
counted area. The square size is fixed in normalised px (RBC radius 16), so a square
holds about the same number of cells at 40x and 100x (~30-40). Writes
  <out>/crops/<photo>_c<k>.jpg   the square cut from the raw photo, enlarged, with context
  <out>/crops.csv                square positions + an empty rbc_hand column to fill in
Counting rule: count a red cell when its CENTRE lies inside the black square.
"""
import argparse, csv, os, sys, zlib
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from count_cells import normalise, count_masks

SIZE = 240            # square side in normalised px (about 6 RBC diameters)
MARGIN = 48         # context shown around the square, normalised px
SHOW = 900          # side of the saved image in px


def pick_squares(inner, n, rng, tries=20000):
    """n non-overlapping squares, all completely inside the counted area. Whole sets are
    drawn at random until one fits, so no position is favoured by the drawing order."""
    ii = cv2.integral((inner > 0).astype(np.uint8))     # sum of any rectangle in O(1)
    full = (ii[SIZE:, SIZE:] - ii[:-SIZE, SIZE:] - ii[SIZE:, :-SIZE]
            + ii[:-SIZE, :-SIZE]) == SIZE * SIZE
    ys, xs = np.nonzero(full)                           # every top-left corner that fits
    if len(xs) < n:
        return []
    for _ in range(tries):
        k = rng.choice(len(xs), n, replace=False)
        sq = list(zip(xs[k].tolist(), ys[k].tolist()))
        if all(abs(a - c) >= SIZE or abs(b - d) >= SIZE
               for i, (a, b) in enumerate(sq) for c, d in sq[i + 1:]):
            return sq
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="data/own_gt")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    csv_path = os.path.join(a.out, "crops.csv")
    if os.path.exists(csv_path):
        sys.exit(f"{csv_path} already exists and may hold your counts - move it away first")
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)

    m = SHOW * MARGIN // (SIZE + 2 * MARGIN)            # square edge in the saved image
    rows = []
    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        raw = cv2.imread(path)
        nrm = normalise(raw)
        _, inner = count_masks(nrm)
        rng = np.random.default_rng([a.seed, zlib.crc32(name.encode())])
        squares = pick_squares(inner, a.n, rng)
        if len(squares) < a.n:
            print(f"{name}: only {len(squares)} squares fit")
        s = nrm["scale"]
        side = int(round((SIZE + 2 * MARGIN) / s))         # crop side in raw px
        big = cv2.copyMakeBorder(raw, side, side, side, side, cv2.BORDER_CONSTANT)
        for k, (x, y) in enumerate(squares, 1):
            # cut from the raw photo (more detail than the normalised one)
            rx, ry = int(round((x - MARGIN) / s)) + side, int(round((y - MARGIN) / s)) + side
            crop = cv2.resize(big[ry:ry + side, rx:rx + side], (SHOW, SHOW),
                              interpolation=cv2.INTER_CUBIC)
            for t in range(1, 4):                           # 4x4 helper grid
                g = m + (SHOW - 2 * m) * t // 4
                cv2.line(crop, (g, m), (g, SHOW - m), (190, 190, 190), 1)
                cv2.line(crop, (m, g), (SHOW - m, g), (190, 190, 190), 1)
            cv2.rectangle(crop, (m, m), (SHOW - m, SHOW - m), (0, 0, 0), 2)
            cv2.imwrite(os.path.join(a.out, "crops", f"{name}_c{k}.jpg"), crop)
            rows.append(dict(image=name, crop=k, x0=x, y0=y, x1=x + SIZE, y1=y + SIZE,
                             raw_x0=round(x / s), raw_y0=round(y / s),
                             raw_x1=round((x + SIZE) / s), raw_y1=round((y + SIZE) / s),
                             rbc_hand=""))
        print(f"{name}: {len(squares)} squares")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} crops to {a.out}/crops and {csv_path}")


if __name__ == "__main__":
    main()