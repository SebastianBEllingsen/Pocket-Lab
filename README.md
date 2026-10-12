# Pocket-Lab

Machine Vision course project (IKT213): a web app for editing microscope images and analysing blood cells.

The app finds and counts **red blood cells**, **platelets** and **7 kinds of white blood cells** (basophil, eosinophil, erythroblast, IG, lymphocyte, monocyte, neutrophil) with a YOLO model, and corrects YOLO's red cell count with an OpenCV double-check.

## Run the app

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5001

The trained model is included (`models/blood_cells.pt`), so no training is needed to run the app.

## How it works

```
Picture ──> YOLO finds the cells ──> OpenCV double-checks the red cells ──> counts + boxes
            (yolo_detection.py)      (postprocessing.py)
```

1. **YOLO** (`yolo_detection.py`) finds every cell in one pass and names its type. It runs once when a picture is opened; the Count and Show buttons reuse the result.
2. **OpenCV double-check** (`postprocessing.py`). YOLO sometimes draws one box around two overlapping red cells. OpenCV separates the cells from the background, splits the cell mask into single cells, and replaces a YOLO box that holds 2+ separate cells with one box per cell. It also adds red cells YOLO missed. Switch it off with `REFINE = False` in `yolo_detection.py`.

The double-check runs *after* YOLO because YOLO was trained on unprocessed pictures.

## How the model was made

The PBC blood cell dataset has about 17,000 pictures in 8 folders, but no boxes around the cells, which YOLO needs to learn. The scripts in `training/` solve that and train the model:

| Step | Script | What it does |
|---|---|---|
| 1 | `auto_label.py` | Draws the boxes with OpenCV. Each picture has one white cell in the centre and its folder names the type; purple shapes are nuclei or platelets; everything else that differs from the background is a red cell, split into single cells with a watershed. |
| 2 | `make_dataset.py` | Labels 500 pictures per folder and splits them 80/10/10 into train, validation and test, in YOLO format. |
| 3 | `check_labels.py`, `remove_deleted.py` | Draws the boxes for checking by eye, then removes the pictures you rejected. |
| 4 | `train.py` | Fine-tunes YOLO11n (pre-trained on everyday photos) and saves `models/blood_cells.pt`. |
| 5 | `evaluate.py` | Scores the model on the test pictures it never saw. |

To retrain (about one hour on an Apple M4):

```bash
python training/make_dataset.py /path/to/bloodcells_dataset --per-folder 500
python training/train.py --epochs 60
python training/evaluate.py
```

Training is deterministic: the same data and settings give the same model.

## Results

Measured on test pictures the model never saw.

| | Result |
|---|---|
| White cells: number and type exactly right | 96% of 400 pictures |
| Red cells counted, 20 pictures counted by hand (344 cells) | 85% with the double-check, 80% with YOLO alone |
| Red cell count error per picture | 3.2 cells with the double-check, 3.8 with YOLO alone |
| Detection score (mAP50, all 9 classes) | 0.958 |
| Time per picture | about 20 ms |

Red cells were checked against hand counts because the automatic labels are least reliable where cells overlap.

**Still hard:** overlapping red cells (about 15% still missed) and platelets, which are very small (mAP50 0.74).

## Project structure

```
Pocket-Lab/
├── app.py              # Flask app: open image, edit, count and show cells
├── yolo_detection.py   # Finds cells with the trained YOLO model
├── postprocessing.py   # OpenCV double-check of YOLO's red cells
├── cell_detection.py   # Older OpenCV-only detection (no longer used by the app)
├── requirements.txt
├── models/
│   ├── blood_cells.pt     # Current model, used by the app
│   └── blood_cells_v1.pt  # First model, kept for comparison
├── training/           # Labelling, dataset, training and evaluation scripts
├── templates/
│   └── index.html
└── static/
    ├── app.js
    ├── style.css
    └── uploads/        # created at runtime, not tracked by git
```

`datasets/` and `runs/` are created by the training scripts and are not tracked by git.

## Next steps

1. **Process the pictures before YOLO**: enhance them or remove the background, retrain YOLO on the processed pictures, and compare the red cell count with the current double-check on the same 20 hand-counted pictures.
2. **Explore pre-trained models** made for cell detection, and compare them with our own YOLO model.
3. **Try a bigger model** (YOLO11s) to separate overlapping red cells better.
4. **Improve the labels for overlapping red cells**, the main source of the remaining errors.
