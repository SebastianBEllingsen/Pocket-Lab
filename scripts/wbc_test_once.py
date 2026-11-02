"""
Score the fixed nucleus rule once on the test cells. Run this one time only: the
threshold was set on train and validation photos and must not change after this.

    python scripts/wbc_test_once.py

Prints only, writes nothing.
"""
import csv, os
import cv2
from wbc_features import CROPS, LABELS, features, nucleus_mask

THRESHOLD = 0.94            # neutrophil below this nucleus thickness


def main():
    cells = [r for r in csv.DictReader(open(LABELS))
             if r["split"] == "test" and r["type"] in ("lymphocyte", "neutrophil")]
    correct = 0
    for r in cells:
        crop = cv2.imread(os.path.join(CROPS, r["crop"] + ".jpg"))
        f = features(nucleus_mask(crop))
        answer = "unknown" if f is None else (
            "neutrophil" if f["thickness"] < THRESHOLD else "lymphocyte")
        correct += answer == r["type"]
        print(f"{r['crop']:<20} hand={r['type']:<11} rule={answer:<11} "
              f"thickness={f['thickness'] if f else float('nan'):.2f}")
    print(f"\nNucleus rule: {correct} / {len(cells)} correct")
    print("Current model + calibration: 11 / 14. Always lymphocyte: 7 / 14.")


if __name__ == "__main__":
    main()