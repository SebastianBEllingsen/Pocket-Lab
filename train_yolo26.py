"""
Train YOLO26 on the blood-cell dataset made by make_yolo_dataset.py.

    pip install ultralytics
    python train_yolo26.py --data dataset/data.yaml --model yolo26s.pt --epochs 150

Notes
- Images are already scale-normalised (RBC radius = 16 px), so scale
  augmentation is kept small; rotation and vertical flips are free for cells.
- Colour augmentation is kept strong so the model tolerates different stains
  and white balance.
- Start with yolo26n/s on a laptop GPU or Colab; move to yolo26m when the
  dataset is larger than a few hundred tiles.
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset/data.yaml")
ap.add_argument("--model", default="yolo26s.pt")     # n / s / m / l / x
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)     # -1 = auto for GPU memory
ap.add_argument("--device", default=None)            # 0, "cpu", "mps"
ap.add_argument("--name", default="bloodcells")
ap.add_argument("--eval-split", default="test",
                help="split for the final score; use val while test labels are drafts")
ap.add_argument("--optimizer", default="auto")       # "auto" IGNORES lr0 -> use SGD with --lr0
ap.add_argument("--lr0", type=float, default=None)   # None = Ultralytics default (0.01 for SGD)
ap.add_argument("--patience", type=int, default=40)
a = ap.parse_args()
if a.lr0 is not None and a.optimizer == "auto":
    raise SystemExit("--lr0 has no effect with --optimizer auto; add --optimizer SGD")

model = YOLO(a.model)                                # pretrained COCO weights
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    name=a.name, patience=a.patience, seed=0, exist_ok=True, optimizer=a.optimizer,
    **({} if a.lr0 is None else {"lr0": a.lr0}),
    # geometry: cells have no "up", tiles are already at a fixed scale
    degrees=90, flipud=0.5, fliplr=0.5, scale=0.15, translate=0.1,
    mosaic=1.0, close_mosaic=15,
    # colour: different stains, lamps and phone white balance
    hsv_h=0.03, hsv_s=0.6, hsv_v=0.4,
    max_det=1000,                                    # a tile can hold ~400 RBCs
)
metrics = model.val(data=a.data, split=a.eval_split, max_det=1000)
print(f"{a.eval_split} mAP50-95:", metrics.box.map, " mAP50:", metrics.box.map50)
present = metrics.box.ap_class_index            # classes that actually occur in the test split
print("per-class mAP50-95:", {model.names[int(i)]: round(float(metrics.box.maps[i]), 3) for i in present})