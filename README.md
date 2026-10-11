# Blood Cell Counting and Classification

Machine Vision course project. Goal: from a microscope image of a blood smear, count the cells, tell red cells (RBC) from white cells (WBC), and classify the WBC subtype (neutrophil, eosinophil, lymphocyte, monocyte, basophil). It should work at any microscope zoom and across different stains and background colors.

## Approach (short)
1. Pre-process: flatten illumination, remove the background, normalize color.
2. Segment cells.
3. Per image, measure typical RBC size (zoom-invariant reference).
4. Flag cells that are much bigger than the reference as overlapping clumps and estimate how many cells they contain.
5. Classify RBC vs. WBC and the WBC subtype (relative size is a feature).
6. Small-data strategy: augmentation plus a synthetic generator that pastes cut-out cells onto backgrounds with exact ground truth.
7. Compare several methods (classical segmentation + classifier vs. detector) against a hand-checked gold standard.

Full detail is in `docs/`. Start with `docs/PLAN.md`.

## Layout
```
README.md           this file
docs/               plan, architecture, ideas, datasets, decisions, conventions
cellcount/          the package: common/ (datasets, selection, metrics), viz/, bg_removal/
tests/              unit tests
main.py             early watershed experiment (hardcoded image path)
model.py            YOLO11n-cls training on Data_Set_2 (experiment)
Data_Sets/          datasets (not in version control; see docs/DATASETS.md)
runs/               Ultralytics training output
```

## Reading guide (to get familiar with the code)
1. `docs/PLAN.md` (what and why), then `docs/CONVENTIONS.md` (how code is organised).
2. `cellcount/bg_removal/pipeline.py`: the whole stage-1 algorithm in one readable file (steps are recorded in a trace).
3. `cellcount/bg_removal/proposers.py` and `methods.py`: how the background colour is found and how the background field is built.
4. `cellcount/bg_removal/config.py`: every tunable parameter and its default (each has a CLI flag).
5. `cellcount/bg_removal/cli.py` and `panels.py`: the command line and what each view shows.
6. `cellcount/common/` and `cellcount/viz/`: dataset loading, image selection, metrics and figure rendering (shared by all later stages).
Run `python -m cellcount.bg_removal --explain --random 1` while reading: it shows each pipeline step with the values used.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Datasets are not included (3 so far: Data_Set_1 RBC masks, Data_Set_2 BCCD, Data_Set_3 PBC). Layout and sources: `docs/DATASETS.md`. Point to them with `--data-root DIR` or `export CELLCOUNT_DATA_ROOT=DIR` (default: `./Data_Sets`).

## Running stage 1: background removal
```
python -m cellcount.bg_removal --help
python -m cellcount.bg_removal --random 4                      # original vs. result, interactive window
python -m cellcount.bg_removal --full --worst 6 --stats        # the 6 lowest-IoU images, full comparison
python -m cellcount.bg_removal --explain --image <name>        # every step with its values
python -m cellcount.bg_removal --dataset 2 --boxes --index 7   # BCCD with annotated boxes
python -m cellcount.bg_removal --random 8 --save out/          # write PNGs instead of a window
python -m cellcount.bg_removal --list                          # image indices and names
python -m cellcount.bg_removal --method mode --worst 6 --stats # compare against the older single-rule estimator
python -m cellcount.bg_removal --proposers mode,border --full  # choose which proposers vote
```
In the window: left/right arrows change image, keys `o`/`g`/`d`/`b` (or the check boxes) toggle our outline, the dataset's reference mask, the difference map and annotated boxes; `q` closes.

Tests: `python -m unittest discover -s tests -t .`

## Status
Early stage. Stage 1 is built and awaiting review (mean IoU about 0.89 on Data_Set_1). See "Where we left off" in `docs/PLAN.md`. For each component we check maintained open-source frameworks first and weigh adopting them against building our own (`docs/IDEAS.md` section 8).

## Version control
Local git, committed regularly. We do not push automatically. `Data_Sets/`, `runs/` and model weights (`*.pt`) are git-ignored.
