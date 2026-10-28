"""
Compare the classical counter and YOLO26 (Stage A, Stage B) on our own test photos.

    python evaluate.py

Ground truth (ground_truth/, made by hand, never used for training)
  crops.csv      random squares per test photo (scripts/make_count_crops.py) with
                 rbc_hand = red cells counted by eye (centre inside the square)
  labels/*.txt   Label Studio YOLO export of boxes drawn on the crop images. Where a
                 crop has boxes, the box count is the reference instead of rbc_hand
                 (counting by eye missed ~17 % of the small cells at 40x)
  classes.txt    class names of that export (Label Studio sorts them alphabetically)
  wbc_types.csv  hand label of every white / smudge cell crop (scripts/make_wbc_crops.py)
All boxes are converted to normalised image px (RBC radius 16), the space every
method works in. Writes reports/eval.md, reports/eval.csv, reports/eval_wbc_types.csv.
"""
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
TYPES = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte",
         "neutrophil"]
PRED_TYPES = "data/wbc_types/predictions.csv"    # written by scripts/make_wbc_crops.py


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
    """Count error per method and magnification. sqrt_ref is the Poisson uncertainty of
    the reference counts (mean sqrt(N) per square): how precisely a square's count
    estimates the cell density of the whole slide, not the error between methods."""
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


def wbc_type_rows(gt=GT, pred=PRED_TYPES):
    """One row per white/smudge cell crop: split, what the classical counter called it,
    the type model's answer and the hand label (ground_truth/wbc_types.csv)."""
    hand = {r["crop"]: r for r in csv.DictReader(open(os.path.join(gt, "wbc_types.csv")))}
    return [dict(crop=p["crop"], split=hand[p["crop"]]["split"], found_as=p["found_as"],
                 predicted=p["predicted"], hand=hand[p["crop"]]["type"])
            for p in csv.DictReader(open(pred))]


def summarise_types(rows):
    """(1) what the hand labels say about the cells the counter called wbc / smudge,
    (2) type accuracy on cells that are a white-cell type by the hand label,
    (3) confusion counts (hand label, model),
    (4) a calibration learned on the train + val photos only: each model answer is
        replaced by the hand label it most often meant there. It is scored on the test
        photos, next to always answering the most common train + val type."""
    found = {f: Counter(r["hand"] for r in rows if r["found_as"] == f) for f in ("wbc", "smudge")}
    typed = [r for r in rows if r["found_as"] == "wbc" and r["hand"] in TYPES]
    acc = {}
    for name, splits in (("train + val photos", ("train", "val")), ("test photos", ("test",)),
                         ("all photos", ("train", "val", "test"))):
        rs = [r for r in typed if r["split"] in splits]
        acc[name] = (sum(r["predicted"] == r["hand"] for r in rs), len(rs))
    calib = [r for r in typed if r["split"] != "test"]
    test = [r for r in typed if r["split"] == "test"]
    remap = {p: Counter(r["hand"] for r in calib if r["predicted"] == p).most_common(1)[0][0]
             for p in {r["predicted"] for r in calib}}
    common = Counter(r["hand"] for r in calib).most_common(1)[0][0]
    cal = dict(remap=remap, common=common, n=len(test),
               calibrated=sum(remap.get(r["predicted"], r["predicted"]) == r["hand"] for r in test),
               baseline=sum(r["hand"] == common for r in test))
    return found, acc, Counter((r["hand"], r["predicted"]) for r in typed), cal


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_report(check, count_sum, box_sum, type_rows, out="reports"):
    """reports/eval.md: all tables of the evaluation, ready to read or copy."""
    found, acc, confusion, cal = summarise_types(type_rows)
    L = ["# Evaluation on our own photos (preliminary)", "",
         "Made by `evaluate.py`. Ground truth: `ground_truth/` (made by hand, never used "
         "for training).", "", f"Check of the 100x hand counts: {check}", "",
         "## Red blood cells per square (12 test photos, 48 squares)", "",
         "| Method | Magnification | Squares | Mean abs. error | Bias | Mean % error | "
         "Total (manual ± √N) |", "|---|---|---|---|---|---|---|"]
    for s in count_sum:
        L.append(f"| {s['method']} | {s['mag']} | {s['squares']} | {s['mae']:.2f} | "
                 f"{s['bias']:+.2f} | {s['mape']:.1f}% | {s['total_pred']:.0f} "
                 f"({s['total_ref']:.0f} ± {np.sqrt(s['total_ref']):.0f}) |")
    L += ["", "## Red blood cell boxes in the hand-boxed squares", "",
          "| Method | Matching | Squares | Manual boxes | Precision | Recall |",
          "|---|---|---|---|---|---|"]
    for s in box_sum:
        L.append(f"| {s['method']} | {s['rule']} | {s['squares']} | {s['ref_boxes']} | "
                 f"{s['precision']:.3f} | {s['recall']:.3f} |")
    L += ["", "## Cells the classical counter called white / smudge, by hand label "
          "(all 100x photos)", "",
          "| Counter said | White cell | Smudge | Not a white cell | Unsure |",
          "|---|---|---|---|---|"]
    for f, c in found.items():
        L.append(f"| {f} | {sum(c[t] for t in TYPES)} | {c['smudge']} | {c['not_wbc']} | "
                 f"{c['unsure']} |")
    L += ["", "## White cell types (first-release YOLO11 on PBC-like crops)", "",
          "Cells the counter called white and the hand label gives a white-cell type. "
          "The type model never saw our photos.", "",
          "| Photos | Correct | Accuracy |", "|---|---|---|"]
    for k, (c, n) in acc.items():
        L.append(f"| {k} | {c} / {n} | {100 * c / max(n, 1):.0f}% |")
    preds = sorted({p for _, p in confusion})
    L += ["", "Rows: hand label, columns: model.", "",
          "| hand \\ model | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
    for h in [t for t in TYPES if any(h == t for h, _ in confusion)]:
        L.append(f"| {h} | " + " | ".join(str(confusion[(h, p)]) for p in preds) + " |")
    n = max(cal["n"], 1)
    L += ["", "Calibration learned on the train + val photos only:", "",
          "| Model answer | Most often meant (train + val) |", "|---|---|"]
    L += [f"| {p} | {h} |" for p, h in sorted(cal["remap"].items())]
    L += ["", "Scored on the test photos:", "", "| Method | Correct | Accuracy |", "|---|---|---|",
          f"| model, raw | {acc['test photos'][0]} / {cal['n']} | "
          f"{100 * acc['test photos'][0] / n:.0f}% |",
          f"| model + calibration | {cal['calibrated']} / {cal['n']} | "
          f"{100 * cal['calibrated'] / n:.0f}% |",
          f"| always '{cal['common']}' | {cal['baseline']} / {cal['n']} | "
          f"{100 * cal['baseline'] / n:.0f}% |"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval.md"), "w") as f:
        f.write("\n".join(L) + "\n")


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
    check = (f"{len(d)} boxed squares, mean |hand - boxes| {np.mean(d):.2f} -> "
             f"{'accepted' if d and np.mean(d) <= 1.5 else 'NOT accepted'}")
    dets, extra = run_methods(sorted(gt))
    rows = rbc_squares(gt, dets, extra)
    types = wbc_type_rows()
    write_report(check, summarise_counts(rows), summarise_boxes(rows), types)
    write_csv("reports/eval.csv", rows)                 # one row per (method, square)
    write_csv("reports/eval_wbc_types.csv", types)      # one row per white/smudge crop
    print(open("reports/eval.md").read())