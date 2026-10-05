"""
Check the trained model on the test images it has never seen.

Usage:
    python training/evaluate.py

Prints:
  1. YOLO's own scores (mAP50 = how well boxes match, per class)
  2. Counting accuracy for several confidence thresholds, so you can pick
     the best CONFIDENCE value for yolo_detection.py
"""
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "blood_cells" / "data.yaml"
TEST = ROOT / "datasets" / "blood_cells" / "images" / "test"
MODEL = ROOT / "models" / "blood_cells.pt"
CONFIDENCES = (0.15, 0.25, 0.35, 0.5)


def count_accuracy(model, conf):
    """Compare predicted counts with the label counts on every test image."""
    names = model.names
    wbc_types = [n for n in names.values() if n not in ("rbc", "platelet")]
    rbc_errors = []
    wbc_right = 0

    for result in model.predict(sorted(TEST.glob("*.jpg")), imgsz=352, conf=conf, verbose=False, stream=True):
        predicted = Counter(names[int(k)] for k in result.boxes.cls.tolist())

        label_file = Path(result.path.replace("/images/", "/labels/")).with_suffix(".txt")
        true = Counter(names[int(line.split()[0])] for line in label_file.read_text().split("\n") if line.strip())

        rbc_errors.append(abs(predicted["rbc"] - true["rbc"]))
        if all(predicted[t] == true[t] for t in wbc_types):
            wbc_right += 1

    total = len(rbc_errors)
    print(f"conf={conf:.2f}  RBC off by {sum(rbc_errors) / total:.1f} cells  |  "
          f"WBC count + type right: {wbc_right / total:.0%}  ({total} images)")


def main():
    model = YOLO(str(MODEL))

    print("=== 1. Detection scores on the test set ===")
    model.val(data=str(DATA), split="test", imgsz=352, plots=False)

    print("\n=== 2. Counting accuracy on the test set ===")
    for conf in (0.15, 0.25, 0.35, 0.5):
        count_accuracy(model, conf)


if __name__ == "__main__":
    main()