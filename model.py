from ultralytics import YOLO

# 4-class blood cell classification (dataset2-master), already split by class
# into TRAIN/TEST/TEST_SIMPLE. train/val/test symlinks (lowercase, pointing
# at TRAIN/TEST/TEST_SIMPLE) live next to the class folders since ultralytics
# expects those exact names.
DATA_DIR = "Data_Sets/Data_Set_2/dataset2-master/dataset2-master/images"

model = YOLO("yolo11n-cls.pt")

model.train(
    data=DATA_DIR,
    epochs=20,
    imgsz=224,
    batch=64,
    device=0,
)

metrics = model.val(split="test")
print(metrics.top1, metrics.top5)
