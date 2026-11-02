# Pocket Lab

Machine Vision course project (IKT213, UiA). A web photo editor that also counts blood cells in phone photos taken through a microscope eyepiece.

> Educational tool, not a diagnostic device. This is a first release: the test set is small, so read all results as preliminary.

## What it does

Upload a photo at 40× or 100× and the app can:

- count red cells, white cells, platelets and smudge cells
- draw a box around each counted cell
- name the type of each white cell (100× photos only)
- edit the image: crop, zoom, text, shapes, undo, redo, reset, save

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5001

## How the analysis works

1. **Normalise:** find the eyepiece circle and rescale the photo so a red cell has a 16 px radius.
2. **Count:** the classical counter finds cells with Lab colour, hysteresis thresholding, morphology and the distance transform. An area rule corrects the count for clumps. A cell counts if its centre is inside the counting circle.
3. **Name white cells:** the first-release YOLO11 model looks at a crop rescaled to look like its PBC training images. A calibration then turns its "erythroblast" answer into "lymphocyte".

The app therefore uses classical machine vision for counting and a deep-learning model for white-cell types.

## Methods we compared

| Method | Summary |
|---|---|
| Classical counter | Colour and size rules, no training (`count_cells.py`) |
| YOLO26 Stage A | Trained on the public TXL-PBC and BBBC041 datasets |
| YOLO26 Stage B | Stage A fine-tuned on our train photos, with labels drafted by the classical counter |
| First-release YOLO11 | Trained on the PBC dataset; tested as-is and on PBC-scale tiles |

## Preliminary results

Red cells in 16 hand-boxed test squares (12 at 40×, 4 at 100×):

| Method | 40× (true 140) | 100× (true 164) |
|---|---|---|
| Classical counter | 142 | 146 |
| YOLO26 Stage A | 111 | 142 |
| YOLO26 Stage B | 113 | 139 |
| First release, PBC-scale tiles | 91 | 132 |
| First release, as-is | 0 | 0 |

White-cell types on 14 test cells at 100×:

| Method | Correct |
|---|---|
| YOLO11 + calibration | 11 / 14 |
| Always "lymphocyte" | 7 / 14 |
| YOLO11, raw | 4 / 14 |

Full tables: [`reports/eval.md`](reports/eval.md).

## Known limits

- Only 16 squares and 14 white cells were tested, and only 4 of the squares are at 100×.
- The type model only answers lymphocyte or neutrophil, so other types cannot be named.
- White-cell detection was checked only on cells the counter found, so missed cells are not measured.
- The app keeps the current image in one global variable, so it supports one user at a time.

## Project layout

```
app.py                Flask app: editor and analysis routes
cell_analysis.py      Analysis used by the website
count_cells.py        Classical counter
count_yolo26.py       YOLO26 detection
evaluate.py           Compares all methods, writes reports/
make_yolo_dataset.py  Builds the YOLO dataset from our photos
train_yolo26.py       Trains YOLO26
first_release/        First-release pipeline, for comparison
ground_truth/         Hand-made test labels (never used for training)
models/               Trained weights
reports/              Evaluation results
scripts/              Crops for labelling, public dataset conversion
processing/           Image editing functions
static/, templates/   Website frontend
tests/                Tests
```

## Data

Our photos and the public datasets are not in this repository (`data/` is git-ignored). The website works without them, but `evaluate.py` and training need them. `evaluate.py` expects our photos in `data/own_v2/`.

Dataset licences: TXL-PBC (MIT), BBBC041 (CC BY-NC-SA, non-commercial).