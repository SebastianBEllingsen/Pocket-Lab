import csv, glob, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from make_count_crops import SIZE, MARGIN
from ultralytics import YOLO
from count_cells import process
from count_yolo26 import detect

GT = "ground_truth"
TEST_DIR = "data/own_v2"
CLASSES = ["rbc", "wbc", "platelet", "smudge"]           # class ids of the classical counter
WEIGHTS = {"stageA": "runs/detect/stageA_public/weights/best.pt",   # public data only
           "stageB": "runs/detect/stageB_own/weights/best.pt"}      # + our photos (main model)


def crop_name(path):
    """'19b73252-unk_40x_09_c2.txt' -> ('unk_40x_09', 2); Label Studio adds an id prefix."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.split("-", 1)[1] if "-" in stem else stem
    image, k = stem.rsplit("_c", 1)
    return image, int(k)


def inside(box, sq):
    """The counting rule: a box counts for a square if its centre lies inside it."""
    cx, cy = (box[1] + box[3]) / 2, (box[2] + box[4]) / 2
    return sq["x0"] <= cx < sq["x1"] and sq["y0"] <= cy < sq["y1"]


def load_ground_truth(gt=GT):
    """{image: [square, ...]}; each square has its position, the reference RBC count
    (rbc_ref) and, if it was boxed, the hand-drawn boxes whose centre is inside it."""
    names = [l.strip() for l in open(os.path.join(gt, "classes.txt")) if l.strip()]
    labels = {}
    for f in glob.glob(os.path.join(gt, "labels", "*.txt")):
        key = crop_name(f)
        if key in labels:
            sys.exit(f"two label files for {key}: {labels[key]} and {f}")
        labels[key] = f
    side = SIZE + 2 * MARGIN                             # crop image side in normalised px
    out = {}
    for r in csv.DictReader(open(os.path.join(gt, "crops.csv"))):
        image, k = r["image"], int(r["crop"])
        sq = dict(crop=k, x0=float(r["x0"]), y0=float(r["y0"]), x1=float(r["x1"]),
                  y1=float(r["y1"]), rbc_hand=float(r["rbc_hand"]),
                  boxed=(image, k) in labels, boxes=[])
        if sq["boxed"]:
            for line in open(labels[(image, k)]):
                if not line.strip():
                    continue
                c, cx, cy, w, h = line.split()
                cx = sq["x0"] - MARGIN + float(cx) * side       # crop -> normalised image px
                cy = sq["y0"] - MARGIN + float(cy) * side
                w, h = float(w) * side, float(h) * side
                b = (names[int(c)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if inside(b, sq):                       # margin cells were not all boxed
                    sq["boxes"].append(b)
        n_box = sum(1 for b in sq["boxes"] if b[0] == "rbc")
        sq["rbc_ref"] = n_box if sq["boxed"] else sq["rbc_hand"]
        out.setdefault(image, []).append(sq)
    unused = set(labels) - {(i, s["crop"]) for i, sqs in out.items() for s in sqs}
    if unused:
        print("label files without a row in crops.csv:", sorted(unused))
    return out

def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def match(pred, ref, rule):
    """Number of one-to-one matches between predicted and reference boxes (x0, y0, x1, y1).
    'iou': IoU >= 0.5. 'centre': the predicted centre lies inside the reference box (fair
    when the box size convention differs). Greedy, best pairs first."""
    pairs = []
    for i, p in enumerate(pred):
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        for j, r in enumerate(ref):
            if rule == "iou":
                s = iou(p, r)
            elif r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
                s = 1 / (1 + (cx - (r[0] + r[2]) / 2) ** 2 + (cy - (r[1] + r[3]) / 2) ** 2)
            else:
                s = 0
            if s >= (0.5 if rule == "iou" else 1e-12):
                pairs.append((s, i, j))
    used_p, used_r = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
    return len(used_p)


def rbc_squares(gt, dets, extra):
    """One row per (method, square): reference vs predicted RBC count; for hand-boxed
    squares also the box matches under both rules."""
    rows = []
    for m, per in dets.items():
        for image, sqs in gt.items():
            for sq in sqs:
                pred = [b[1:] for b in per[image] if b[0] == "rbc" and inside(b, sq)]
                n = len(pred)
                if m == "classical":                    # + its area-based clump correction
                    n += sum(k for b, k in extra[image] if inside(b, sq))
                row = dict(method=m, image=image, crop=sq["crop"], mag=image.split("_")[1],
                           ref=sq["rbc_ref"], pred=n, err=n - sq["rbc_ref"], boxed=sq["boxed"])
                if sq["boxed"]:
                    ref = [b[1:] for b in sq["boxes"] if b[0] == "rbc"]
                    row.update(n_pred_boxes=len(pred), n_ref_boxes=len(ref),
                               tp_iou=match(pred, ref, "iou"),
                               tp_centre=match(pred, ref, "centre"))
                rows.append(row)
    return rows


def summarise_counts(rows):
    """Count error per method and magnification. sqrt_ref is the Poisson noise of the
    reference counts themselves (mean sqrt(N) per square): an MAE below it is within
    the noise of counting cells in a small square."""
    out = []
    for m in dict.fromkeys(r["method"] for r in rows):
        for mag in ("40x", "100x", "all"):
            rs = [r for r in rows if r["method"] == m and mag in ("all", r["mag"])]
            ref = np.array([r["ref"] for r in rs], float)
            pred = np.array([r["pred"] for r in rs], float)
            out.append(dict(method=m, mag=mag, squares=len(rs),
                            mae=np.abs(pred - ref).mean(), bias=(pred - ref).mean(),
                            mape=100 * (np.abs(pred - ref) / ref).mean(),
                            sqrt_ref=np.sqrt(ref).mean(),
                            total_ref=ref.sum(), total_pred=pred.sum()))
    return out


def summarise_boxes(rows):
    """RBC box precision / recall in the hand-boxed squares, for both matching rules."""
    out = []
    for m in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == m and r["boxed"]]
        n_pred, n_ref = sum(r["n_pred_boxes"] for r in rs), sum(r["n_ref_boxes"] for r in rs)
        for rule in ("iou", "centre"):
            tp = sum(r[f"tp_{rule}"] for r in rs)
            out.append(dict(method=m, rule=rule, squares=len(rs), ref_boxes=n_ref,
                            precision=tp / max(n_pred, 1), recall=tp / max(n_ref, 1)))
    return out

def run_methods(images, weights=WEIGHTS, conf=0.25, tmp="data/eval_tmp"):
    """Boxes of every method per test photo, in normalised px:
    {method: {image: [(class_name, x0, y0, x1, y1), ...]}}. Also returns the classical
    counter's clumps that hold more cells than boxes: {image: [(box, n_missing), ...]},
    so its square counts include the area-based clump correction, as its totals do."""
    os.makedirs(tmp, exist_ok=True)
    models = {m: YOLO(w) for m, w in weights.items()}
    dets = {m: {} for m in ["classical", *models]}
    extra = {}
    for image in images:
        # one normalise() per photo, shared by all methods -> identical coordinates
        _, nrm, cdets, review = process(os.path.join(TEST_DIR, image + ".jpg"), tmp,
                                        return_dets=True)
        dets["classical"][image] = [(CLASSES[c], x0, y0, x1, y1) for c, x0, y0, x1, y1 in cdets]
        extra[image] = [(("rbc", x0, y0, x1, y1), n) for x0, y0, x1, y1, n in review]
        for m, model in models.items():
            b, c = detect(model, nrm, conf)
            dets[m][image] = [(model.names[int(k)], *map(float, bb)) for bb, k in zip(b, c)]
    return dets, extra


if __name__ == "__main__":
    gt = load_ground_truth()
    # agreed rule: the 100x CSV counts are accepted if they differ from the boxes by
    # at most 1.5 cells on average in the boxed 100x squares
    d = [abs(s["rbc_hand"] - s["rbc_ref"]) for i, sqs in gt.items() if "_100x_" in i
         for s in sqs if s["boxed"]]
    print(f"100x CSV vs boxes: {len(d)} squares, mean |diff| {np.mean(d):.2f} "
          f"-> {'accepted' if d and np.mean(d) <= 1.5 else 'NOT accepted'}")
    dets, extra = run_methods(sorted(gt))
    rows = rbc_squares(gt, dets, extra)
    print("\nRBC count per square (sqrt_ref = Poisson noise of the reference)")
    for s in summarise_counts(rows):
        print(f"{s['method']:10s} {s['mag']:5s} n={s['squares']:2d}  MAE {s['mae']:5.2f}  "
              f"bias {s['bias']:+6.2f}  MAPE {s['mape']:5.1f}%  sqrt_ref {s['sqrt_ref']:4.2f}  "
              f"total {s['total_pred']:.0f} vs {s['total_ref']:.0f} +/- {np.sqrt(s['total_ref']):.0f}")
    print("\nRBC boxes in hand-boxed squares")
    for s in summarise_boxes(rows):
        print(f"{s['method']:10s} {s['rule']:6s} squares={s['squares']:2d} "
              f"ref_boxes={s['ref_boxes']}  P {s['precision']:.3f}  R {s['recall']:.3f}")