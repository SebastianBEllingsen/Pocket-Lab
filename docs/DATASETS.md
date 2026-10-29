# Datasets

The datasets are not final. Replace or add as needed. Root: `Data_Sets/` (git-ignored, about 5 GB). Inspected on 2026-10-11. Loader names for the CLIs: `rbc-masks`/`1`, `bccd`/`2`, `pbc`/`3`.

## Data_Set_1: RBC segmentation masks
- Path: `Data_Sets/Data_Set_1/Final data/Final data/New folder/{Original,Mask}`
- 1328 pairs, same filename in both folders. Images are 1600x1200 PNG (a few originals are .jpg).
- Mask is binary (0/255), 3-channel. It is a semantic mask (cell vs. background), not an instance mask. Touching cells are separated only by a thin dark line in some places and merged in others.
- Mostly purple/blue on a near-white background at high zoom (about 60 cells per image), but some images are pale grey and low contrast with a vignette. Image sizes vary (1600x1200, 1944x1383, ...).
- The mask includes the WBC blobs too. There is no RBC/WBC distinction and no subtype labels.
- Use: training and evaluating segmentation and background removal. Source of RBC cut-outs for the synthetic generator. Source of the "normal size" statistics.

## Data_Set_2/dataset-master: BCCD (full fields with boxes)
- Path: `Data_Sets/Data_Set_2/dataset-master/dataset-master/{JPEGImages,Annotations,labels.csv}`
- 366 images, 640x480. 343 Pascal VOC XML files with boxes: 3854 RBC boxes counted by grepping the annotation files, plus WBC and platelet boxes.
- `labels.csv` gives the WBC subtype per image (about 207 neutrophil, 87 eosinophil, 33 lymphocyte, 21 monocyte, 3 basophil, 44 blank).
- Pink/orange RBCs on a grey-green background, different color and zoom than Data_Set_1. Has heavily overlapping RBC clumps.
- Use: the only source with full fields, boxes, RBC/WBC distinction and subtype labels. This is the realistic test set for counting and for the overlap work.

## Data_Set_2/dataset2-master: classification set (derived from BCCD)
- Path: `Data_Sets/Data_Set_2/dataset2-master/dataset2-master/images`
- Classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL. 240x320 images (cropped/augmented from BCCD, not single-cell crops).
- TRAIN: about 2.5k per class. TEST: about 620 per class. TEST_SIMPLE: 13/6/4/48 images.
- Lowercase symlinks `train`->TRAIN, `val`->TEST, `test`->TEST_SIMPLE (Ultralytics naming), so `model.val(split="test")` uses only the 71 skewed TEST_SIMPLE images.
- Leakage check (2026-10-11), INCONCLUSIVE: about 12k images derive from about 360 originals. A flip-invariant thumbnail comparison showed test images are no closer to train than train images are to each other, and ORB matching against 1500 train images found no exact duplicates (best good-match counts 4-26 of 500 features, a few higher). Rotated/shifted augmentations would defeat both tests. From general knowledge, this classification set was built by augmenting first and splitting after, which would leak (unverified). Safe approach: build our own splits by original BCCD image.

## Data_Set_3: PBC (single-cell images by class)
- Path: `Data_Sets/Data_Set_3/extracted/PBC_dataset_normal_DIB/<class>/*.jpg` (zip kept next to it as `PBC_dataset_normal_DIB.zip`). Licence CC BY 4.0. Source: Acevedo et al. 2020, Mendeley Data (see IDEAS.md section 7).
- 17,093 JPG files (360x363, one cell in the centre), 8 classes: neutrophil 3329, eosinophil 3117, ig (immature granulocyte) 2895, platelet 2348, erythroblast 1551, monocyte 1420, basophil 1218, lymphocyte 1214.
- Loader: dataset name `pbc` or `3`; `Sample.label` holds the class folder name. No masks or boxes.
- Use: WBC subtype classifier (all five subtypes, with far more basophils, lymphocytes and monocytes than BCCD), plus single-cell cut-outs for the synthetic generator. Not for counting or zoom invariance (one fixed zoom, one cell per image, CellaVision stain).
- Caveat: different scanner, stain and background than BCCD, so a classifier trained on PBC alone may not transfer to BCCD-style images; test cross-dataset.

## Own phone/eyepiece photos (the target look)
- Path: `Data_Sets/own_v2/own_v2/` (unzipped from `own_v2.zip`, git-ignored). 63 photos `unk_40x_*.jpg` and `unk_100x_*.jpg` plus a few `blood_cell*.jpg`, 2880x2160, a bright circle (eyepiece view) inside a dark rim, yellow/orange/pink glass, faint low-contrast red cells, strong vignette, sometimes a dark shadow across the circle.
- Used for: glass-colour fields of the synthetic scenes (stage 1 estimates them), visual comparison, and the teammate's evaluation (his 48 hand-counted squares live in his branch under `ground_truth/`). Not used for cutting cells out (cells too faint for stage 1).
- The circle finder (`--fov auto`) finds the circle in 62 of the 63 photos; `unk_100x_10` fails (large shadow).

## Generated data (git-ignored, under `Data_Sets/`)
- `synth_bank/`: cut-out bank (`rbc_*.png`, `wbc_*.png`, `platelet_*.png` as BGRA, `index.json`), glass fields (`bg_*.png`, `bg_index.json`) and contact sheets (`sheet_*.png`). Build: `python -m cellcount.synth.bank --sheets --stats`.
- `synth_handover.zip` (401 MB): the clean `synth_out` (5,000 train + 300 val scenes, glass fields only from photos outside the teammate's validation/test lists) plus `README_dataset.txt` (source: `docs/handover/README_dataset.txt`), made for the teammate on 2026-10-29. `synth_out_old_bgs/` is the earlier set whose glass fields used held-out photos: do not share.
- `synth_out/`: synthetic YOLO dataset in the teammate's format (classes 0 rbc, 1 wbc, 2 platelet, 3 smudge), with `wbc_types.csv` and `manifest.csv`. Build: `python -m cellcount.synth --n 300 --seed 1 --out Data_Sets/synth_out --split train`.

## Teammate's working data (`Data_Sets/colleague_data/data/`, from his `data.zip`, 2026-10-29; git-ignored)
- `merged/`: his public "Stage A" training set, already normalised (red-cell radius 16 px, 640 px tiles): train 1,440 / val 144 / test 133 tiles from TXL-PBC (MIT; 881 train images) and BBBC041 (CC BY-NC-SA; 1,089 train images). Boxes train: 116,252 rbc, 3,720 wbc, 1,528 platelet, **0 smudge** (the real data has no smudge class; only synthetic data provides it). `manifest.csv` gives source, split and scale per image.
- `own_v2_yolo/`: tiles of his own phone photos with classical-counter pseudo-labels (train 202 tiles from 19 photos, val 84 from the 6 validation photos, test 100 from the 12 test photos). Pseudo-labels leave clumped cells unlabelled, so they are weak ground truth.
- `own_gt/`, `own_gt_size192/`: his hand-counted test squares and crops. Never used for training or tuning. `wbc_types/`, `own_v2_yolo_v1`, `own_v2_yolo_crashed`: earlier artefacts.

## Notes
- Stain, background and zoom differ strongly between Data_Set_1 and BCCD (see IDEAS.md section 4).
- No dataset has instance masks separating overlapping cells. The overlap method has to supply that.
- Still missing: a real set with WBC subtypes at the higher zoom of Data_Set_1, and a second source of subtype labels with more lymphocytes and monocytes. Basophil is nearly absent (3 images).
