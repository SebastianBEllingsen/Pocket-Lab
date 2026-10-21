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
a = ap.parse_args()

model = YOLO(a.model)                                # pretrained COCO weights
model.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    name=a.name, patience=40, seed=0,
    # geometry: cells have no "up", tiles are already at a fixed scale
    degrees=90, flipud=0.5, fliplr=0.5, scale=0.15, translate=0.1,
    mosaic=1.0, close_mosaic=15,
    # colour: different stains, lamps and phone white balance
    hsv_h=0.03, hsv_s=0.6, hsv_v=0.4,
    max_det=1000,                                    # a tile can hold ~400 RBCs
)
metrics = model.val(data=a.data, split="test", max_det=1000)
print("test mAP50-95:", metrics.box.map, " mAP50:", metrics.box.map50)
print("per-class mAP50-95:", dict(zip(model.names.values(), metrics.box.maps)))