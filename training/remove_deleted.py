"""
After checking labels by eye: remove every image you deleted from
datasets/check/<split>/ from the real dataset too (image + label).

Usage:
    python training/remove_deleted.py test
"""
import sys
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "blood_cells"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    kept = {p.stem for p in (DATASET.parent / "check" / split).glob("*.jpg")}

    removed = 0
    for image in (DATASET / "images" / split).glob("*.jpg"):
        if image.stem not in kept:
            image.unlink()
            (DATASET / "labels" / split / f"{image.stem}.txt").unlink(missing_ok=True)
            removed += 1

    # YOLO caches the file list; delete it so the change is picked up
    (DATASET / "labels" / f"{split}.cache").unlink(missing_ok=True)

    print(f"Removed {removed} images, {len(kept)} left in {split}")


if __name__ == "__main__":
    main()