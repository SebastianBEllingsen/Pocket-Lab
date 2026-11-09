"""Cut every white / smudge cell the counter found in our 100x photos, for hand labelling."""
import argparse, csv, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell_analysis import analyse, pbc_crop

VAL = {"unk_100x_01", "unk_100x_02", "unk_100x_03"}


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
