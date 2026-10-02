"""
Train YOLO on the dataset made by make_dataset.py.

Usage:
    python training/train.py --epochs 60

The best model is copied to models/blood_cells.pt, which the app uses.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "blood_cells" / "data.yaml"
MODEL_OUT = ROOT / "models" / "blood_cells.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--model", default="yolo11n.pt", help="starting weights (downloaded automatically)")
    parser.add_argument("--device", default="mps", help="mps (Apple GPU), cpu, or 0 for an NVIDIA GPU")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=352,          # images are 360x360; YOLO needs a multiple of 32
        batch=16,
        device=args.device,
        project=str(ROOT / "runs"),
        name="blood_cells",
        exist_ok=True,
        patience=15,        # stop early if it stops improving
        flipud=0.5,         # cells look the same upside down
    )

    MODEL_OUT.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, MODEL_OUT)
    print("Saved model to", MODEL_OUT)


if __name__ == "__main__":
    main()