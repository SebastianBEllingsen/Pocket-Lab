"""Compare the classical counter, YOLO26 and the first-release pipeline on our own test photos."""
import csv, glob, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from make_count_crops import SIZE, MARGIN
from ultralytics import YOLO
from count_cells import process
from count_yolo26 import detect
import cv2
import torch
from torchvision.ops import nms
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_release"))
import yolo_detection as first_release
first_release.MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "models", "blood_cells.pt")

GT = "ground_truth"
TEST_DIR = "data/own_v2"
CLASSES = ["rbc", "wbc", "platelet", "smudge"]
WEIGHTS = {"stageA": "runs/detect/stageA_public/weights/best.pt",
           "stageB": "runs/detect/stageB_own/weights/best.pt"}
TYPES = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte",
         "neutrophil"]
PRED_TYPES = "data/wbc_types/predictions.csv"
PBC_TILE, PBC_STRIDE, PBC_RBC_RADIUS = 360, 180, 35.4


def crop_name(path):
    # label studio prefixes an id
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.split("-", 1)[1] if "-" in stem else stem
    image, k = stem.rsplit("_c", 1)
    return image, int(k)


def inside(box, sq):
    cx, cy = (box[1] + box[3]) / 2, (box[2] + box[4]) / 2
    return sq["x0"] <= cx < sq["x1"] and sq["y0"] <= cy < sq["y1"]


def load_ground_truth(gt=GT):
    names = [l.strip() for l in open(os.path.join(gt, "classes.txt")) if l.strip()]
    labels = {}
    for f in glob.glob(os.path.join(gt, "labels", "*.txt")):
        key = crop_name(f)
        if key in labels:
            sys.exit(f"two label files for {key}: {labels[key]} and {f}")
        labels[key] = f
    side = SIZE + 2 * MARGIN
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
                cx = sq["x0"] - MARGIN + float(cx) * side
                cy = sq["y0"] - MARGIN + float(cy) * side
                w, h = float(w) * side, float(h) * side
                b = (names[int(c)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if inside(b, sq):  # margin cells weren't all boxed
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
    """Greedy one-to-one matches. 'iou': IoU >= 0.5. 'centre': pred centre inside ref box."""
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
    rows = []
    for m, per in dets.items():
        for image, sqs in gt.items():
            for sq in sqs:
                pred = [b[1:] for b in per[image] if b[0] == "rbc" and inside(b, sq)]
                n = len(pred)
                if m == "classical":  # add the area-based clump correction
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
    hand = {r["crop"]: r for r in csv.DictReader(open(os.path.join(gt, "wbc_types.csv")))}
    return [dict(crop=p["crop"], split=hand[p["crop"]]["split"], found_as=p["found_as"],
                 predicted=p["predicted"], hand=hand[p["crop"]]["type"])
            for p in csv.DictReader(open(pred))]


def summarise_types(rows):
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


def write_report(check, count_sum, box_sum, type_rows, boxed_sum, out="reports"):
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
    L += ["", "## Red blood cells in the hand-boxed squares only (reference: boxes)", "",
          "Main result: the 100x check rule failed, so only boxed squares have a reliable "
          "reference.", "",
          "| Method | Magnification | Squares | Mean abs. error | Bias | Mean % error | "
          "Total (manual ± √N) |", "|---|---|---|---|---|---|---|"]
    for s in boxed_sum:
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


def first_release_boxes(raw, nrm, squares):
    """First-release YOLO11 pipeline, as-is and on PBC-scaled tiles."""
    s = nrm["scale"]
    kind = lambda t: t if t in ("rbc", "platelet") else "wbc"
    asis = [(kind(c["type"]), *(v * s for v in c["box"]))
            for c in first_release.detect_cells(raw)]

    f = PBC_RBC_RADIUS / nrm["rbc_radius_px"]
    pbc = cv2.resize(raw, None, fx=f, fy=f,
                     interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    pad = PBC_TILE
    pbc = cv2.copyMakeBorder(pbc, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    boxes, scores, types = [], [], []
    for sq in squares:
        x0, y0 = (int(sq[k] / s * f) + pad - PBC_STRIDE for k in ("x0", "y0"))
        x1, y1 = (int(sq[k] / s * f) + pad + PBC_STRIDE for k in ("x1", "y1"))
        for ty in range(y0, y1 - PBC_TILE + PBC_STRIDE, PBC_STRIDE):
            for tx in range(x0, x1 - PBC_TILE + PBC_STRIDE, PBC_STRIDE):
                for c in first_release.detect_cells(pbc[ty:ty + PBC_TILE, tx:tx + PBC_TILE]):
                    bx0, by0, bx1, by1 = c["box"]
                    if min(bx0, by0) <= 2 or max(bx1, by1) >= PBC_TILE - 2:
                        continue  # cut by the tile edge
                    boxes.append([bx0 + tx, by0 + ty, bx1 + tx, by1 + ty])
                    scores.append(c["confidence"] + 1e-3)
                    types.append(kind(c["type"]))
    if not boxes:
        return asis, []
    keep = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores), 0.5).tolist()
    back = s / f
    return asis, [(types[i], *((v - pad) * back for v in boxes[i])) for i in keep]


def run_methods(gt, weights=WEIGHTS, conf=0.25, tmp="data/eval_tmp"):
    os.makedirs(tmp, exist_ok=True)
    models = {m: YOLO(w) for m, w in weights.items()}
    dets = {m: {} for m in ["classical", *models, "first_release_asis", "first_release_pbc"]}
    extra = {}
    for image in sorted(gt):
        path = os.path.join(TEST_DIR, image + ".jpg")
        # one normalise() per photo, shared across methods
        _, nrm, cdets, review = process(path, tmp, return_dets=True)
        dets["classical"][image] = [(CLASSES[c], x0, y0, x1, y1) for c, x0, y0, x1, y1 in cdets]
        extra[image] = [(("rbc", x0, y0, x1, y1), n) for x0, y0, x1, y1, n in review]
        for m, model in models.items():
            b, c = detect(model, nrm, conf)
            dets[m][image] = [(model.names[int(k)], *map(float, bb)) for bb, k in zip(b, c)]
        dets["first_release_asis"][image], dets["first_release_pbc"][image] = \
            first_release_boxes(cv2.imread(path), nrm, gt[image])
    return dets, extra


if __name__ == "__main__":
    gt = load_ground_truth()
    # agreed rule: accept the 100x hand counts if they differ from the boxes by
    # at most 1.5 cells on average in the boxed 100x squares
    d = [abs(s["rbc_hand"] - s["rbc_ref"]) for i, sqs in gt.items() if "_100x_" in i
         for s in sqs if s["boxed"]]
    check = (f"{len(d)} boxed squares, mean |hand - boxes| {np.mean(d):.2f} -> "
             f"{'accepted' if d and np.mean(d) <= 1.5 else 'NOT accepted'}")
    dets, extra = run_methods(gt)
    rows = rbc_squares(gt, dets, extra)
    types = wbc_type_rows()
    write_report(check, summarise_counts(rows), summarise_boxes(rows), types,
                 summarise_counts([r for r in rows if r["boxed"]]))
    write_csv("reports/eval.csv", rows)
    write_csv("reports/eval_wbc_types.csv", types)
    print(open("reports/eval.md").read())
