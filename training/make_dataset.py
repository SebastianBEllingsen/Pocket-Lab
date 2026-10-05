"""
Build a YOLO dataset from the PBC dataset folders using auto_label.py.

Usage:
    python training/make_dataset.py /path/to/bloodcells_dataset --per-folder 500

Output (inside the project, ignored by git):
    datasets/blood_cells/images/{train,val,test}/*.jpg
    datasets/blood_cells/labels/{train,val,test}/*.txt
    datasets/blood_cells/data.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2

from auto_label import CLASSES, SIZE, label_image

OUTPUT = Path(__file__).resolve().parent.parent / "datasets" / "blood_cells"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def to_yolo_line(class_id, x1, y1, x2, y2):
    """YOLO format: class x_centre y_centre width height (all between 0 and 1)."""
    xc = (x1 + x2) / 2 / SIZE
    yc = (y1 + y2) / 2 / SIZE
    w = (x2 - x1) / SIZE
    h = (y2 - y1) / SIZE
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="folder with basophil/, eosinophil/, ...")
    parser.add_argument("--per-folder", type=int, default=500, help="images to use per cell type")
    args = parser.parse_args()

    random.seed(42)  # same split every time
    shutil.rmtree(OUTPUT, ignore_errors=True)
    for split in SPLITS:
        (OUTPUT / "images" / split).mkdir(parents=True)
        (OUTPUT / "labels" / split).mkdir(parents=True)

    for folder in sorted(p for p in args.source.iterdir() if p.is_dir()):
        if folder.name not in CLASSES:
            print("Skipping unknown folder:", folder.name)
            continue

        files = sorted(folder.glob("*.jpg"))
        random.shuffle(files)

        labelled = []
        for path in files:
            if len(labelled) == args.per_folder:
                break
            img = cv2.imread(str(path))
            if img is None:
                continue
            img, boxes = label_image(img, folder.name)
            if boxes is None:  # too unclear to label, skip it
                continue
            labelled.append((path.stem, img, boxes))

        n_train = int(len(labelled) * SPLITS["train"])
        n_val = int(len(labelled) * SPLITS["val"])
        for i, (name, img, boxes) in enumerate(labelled):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            cv2.imwrite(str(OUTPUT / "images" / split / f"{name}.jpg"), img)
            lines = [to_yolo_line(*box) for box in boxes]
            (OUTPUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")

        print(f"{folder.name:13} {len(labelled)} images")

    yaml = [f"path: {OUTPUT}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {name}" for i, name in enumerate(CLASSES)]
    (OUTPUT / "data.yaml").write_text("\n".join(yaml) + "\n")
    print("Wrote", OUTPUT / "data.yaml")


if __name__ == "__main__":
    main()