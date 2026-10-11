# Datasets

The datasets are not final. Replace or add as needed. Root: `Data_Sets/`. Inspected on 2026-10-11.

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

## Notes
- Stain, background and zoom differ strongly between Data_Set_1 and BCCD (see IDEAS.md section 4).
- No dataset has instance masks separating overlapping cells. The overlap method has to supply that.
- Still missing: a real set with WBC subtypes at the higher zoom of Data_Set_1, and a second source of subtype labels with more lymphocytes and monocytes. Basophil is nearly absent (3 images).
