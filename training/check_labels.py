"""
Draw the auto-generated boxes so you can check them by eye.

Usage:
    python training/check_labels.py test

Saves images to datasets/check/<split>/. Look through them; if a label is
wrong, delete that image from datasets/blood_cells/images/<split>/ and its
.txt from datasets/blood_cells/labels/<split>/.
"""
import sys
from pathlib import Path

import cv2

from auto_label import CLASSES

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "blood_cells"
COLOURS = {"rbc": (0, 0, 255), "platelet": (255, 0, 255)}  # WBC types: green


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    out_dir = DATASET.parent / "check" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted((DATASET / "images" / split).glob("*.jpg")):
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        label_path = DATASET / "labels" / split / f"{image_path.stem}.txt"

        for line in label_path.read_text().split("\n"):
            if not line.strip():
                continue
            k, xc, yc, bw, bh = line.split()
            name = CLASSES[int(k)]
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            p1 = (int(xc - bw / 2), int(yc - bh / 2))
            p2 = (int(xc + bw / 2), int(yc + bh / 2))
            colour = COLOURS.get(name, (0, 170, 0))
            cv2.rectangle(img, p1, p2, colour, 1)
            if name not in COLOURS:
                cv2.putText(img, name, (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        cv2.imwrite(str(out_dir / image_path.name), img)

    print("Saved to", out_dir)


if __name__ == "__main__":
    main()