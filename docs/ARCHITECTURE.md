# Architecture (draft)

Stage 1 (background removal) and the shared packages exist. Later stages are the intended structure and will change.

## Implemented (2026-10-11)
```
cellcount/
  common/datasets.py    Sample/Box, dataset registry (rbc-masks "1", bccd "2"), --data-root / $CELLCOUNT_DATA_ROOT
  common/selection.py   shared flags: --image --index --random --worst --pool --seed --list
  common/metrics.py     iou, dice
  viz/colors.py         contrast-aware outline colours, colour parsing
  viz/overlay.py        RGBA layers: outlines, boxes, difference
  viz/figure.py         Page/Panel/Layer; interactive viewer (toggles, navigation) and PNG saving
  bg_removal/config.py  BgConfig dataclass (all defaults)
  bg_removal/methods.py estimator registry: "mode" (default), "median"
  bg_removal/pipeline.py remove_background() -> BgResult with a full step trace
  bg_removal/panels.py  trace -> viewer pages (--minimal/--full/--explain)
  bg_removal/cli.py     python -m cellcount.bg_removal
tests/test_bg_removal.py  unittest (python -m unittest discover -s tests -t .)
```
Coupling rules: `viz` and `common` never import a stage; stage `pipeline`/`methods` never import `viz`; only `panels.py` and `cli.py` know both sides. Algorithms return plain dataclasses, and the viewer renders from them.

Background removal in short: the `mode` estimator finds the dominant colour (background), smooths a background field from matching pixels only (so vignetting is followed and cells ignored), then the mask is the Lab distance to that field, Otsu-thresholded, holes filled, specks removed. Measured: mean IoU about 0.90 on random Data_Set_1 images; BCCD judged by eye.

The old `src/preprocess.py` prototype was removed (superseded).

## Proposed layout (later stages, not built)
```
cellcount/
  bg_removal/     # DONE (see above); stain/colour normalization may join it
  segment.py      # cell masks and boxes (watershed or learned)
  scale.py        # per-image RBC reference size
  overlap.py      # clump detection, cell-count estimate, splitting
  classify.py     # RBC/WBC and WBC subtype
  synth/          # cut-out extraction and synthetic image generator
  pipeline.py     # end-to-end: image -> cell records + counts
```
Existing scripts `main.py` and `model.py` stay as experiments until superseded.

## Cell record (one entry per detected cell, single list)
| field | meaning |
|---|---|
| cell_type | RBC, or WBC with subtype (eosinophil, lymphocyte, monocyte, neutrophil, ...) |
| bbox / extreme points | location in image pixels |
| norm_size | size divided by the image's RBC reference (zoom-invariant) |
| area, circularity | shape descriptors |
| clump_count | estimated number of cells (1 = single, >1 = overlap) |
| confidence | classifier confidence |

Red and white views are filters over this one list, not separate stores.

## Scale reference
Computed from RBCs only (the most numerous cells). Use a robust statistic (median, not mean) so clumps and WBCs don't skew it. Caveat: RBC size varies in some conditions (e.g. anemia). Note it in the report.

## Data flow
image -> preprocess -> segment -> scale reference -> overlap handling -> classify -> cell records -> counts
