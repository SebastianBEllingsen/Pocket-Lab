"""
Cut out every white cell and smudge cell found by the classical counter in our 100x
photos, exactly as the type model sees it, so the type can be labelled by hand.

    python scripts/make_wbc_crops.py data/own_v2/unk_100x_*.jpg --exclude unk_100x_23 unk_100x_27

The type model never saw our photos, so every photo can be used to measure it; the
split column still allows reporting train/val and test photos separately. Writes
  <out>/crops/<photo>_w<k>.jpg   one crop per cell, the cell in the centre
  <out>/labels.csv               crop, photo, split, type (empty: fill in by hand)
  <out>/predictions.csv          what the classical counter and the model said:
                                 do NOT show this file to the person labelling
Types to write: basophil, eosinophil, erythroblast, ig, lymphocyte, monocyte,
neutrophil, smudge, not_wbc, unsure
"""
import argparse, csv, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell_analysis import analyse, pbc_crop

VAL = {"unk_100x_01", "unk_100x_02", "unk_100x_03"}      # split from make_yolo_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="data/wbc_types")
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "labels.csv")):
        sys.exit(f"{a.out}/labels.csv already exists and may hold labels - move it away first")
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    test = {r["image"] for r in csv.DictReader(open("ground_truth/crops.csv"))}
    labels, preds = [], []
    for path in sorted(a.images):
        photo = os.path.splitext(os.path.basename(path))[0]
        if photo in a.exclude:
            continue
        image = cv2.imread(path)
        r = analyse(image)
        split = "test" if photo in test else "val" if photo in VAL else "train"
        cells = ([(b, "wbc", t) for b, t in zip(r["boxes"]["wbc"], r["wbc_types"])] +
                 [(b, "smudge", "-") for b in r["boxes"]["smudge"]])
        for k, (box, found_as, predicted) in enumerate(cells, 1):
            name = f"{photo}_w{k}"
            cv2.imwrite(os.path.join(a.out, "crops", name + ".jpg"),
                        pbc_crop(image, box, r["rbc_radius"]))
            labels.append(dict(crop=name, photo=photo, split=split, type=""))
            preds.append(dict(crop=name, found_as=found_as, predicted=predicted))
        print(f"{photo} ({split}): {len(cells)} cells")
    for name, rows in (("labels.csv", labels), ("predictions.csv", preds)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(labels)} crops to {a.out}/crops")


if __name__ == "__main__":
    main()