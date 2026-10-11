# Datasets

The datasets are not final. Replace or add as needed. Root: `Data_Sets/` (git-ignored, about 5 GB). Inspected on 2026-10-11. Loader names for the CLIs: `rbc-masks`/`1`, `bccd`/`2`, `pbc`/`3`.

## Data_Set_1: RBC segmentation masks
- Path: `Data_Sets/Data_Set_1/Final data/Final data/New folder/{Original,Mask}`
- Source: https://github.com/Deponker/Blood-cell-segmentation-dataset (BBBC041Seg, Depto et al. 2021; images hosted on Google Drive; repo is MIT, dataset licence not stated). Images derive from BBBC041v1 (Ljosa et al. 2012). How the masks were made is not documented.
- 1328 pairs, same filename in both folders. Images are 1600x1200 PNG (a few originals are .jpg).
- Mask is binary (0/255), 3-channel. It is a semantic mask (cell vs. background), not an instance mask. Touching cells are separated only by a thin dark line in some places and merged in others.
- Mostly purple/blue on a near-white background at high zoom (about 60 cells per image), but some images are pale grey and low contrast with a vignette. Image sizes vary (1600x1200, 1944x1383, ...).
- The mask includes the WBC blobs too. There is no RBC/WBC distinction and no subtype labels.
- Use: training and evaluating segmentation and background removal. Source of RBC cut-outs for the synthetic generator. Source of the "normal size" statistics.

## Data_Set_2/bccd-shenggan: BCCD (full fields with boxes)
- Path: `Data_Sets/Data_Set_2/bccd-shenggan/BCCD/{JPEGImages,Annotations,ImageSets}` (a git clone of https://github.com/Shenggan/BCCD_Dataset, MIT licence). This is what the `bccd`/`2` loader reads.
- 364 images (640x480), 364 Pascal VOC XML files with 4888 boxes: 4155 RBC, 372 WBC, 361 Platelets. The repo README says the XML is "automatically generate[d] by the label tools"; that most likely means the labelling tool wrote the XML, not that the boxes were found by an algorithm. How the boxes were drawn is not documented. WBC boxes carry no subtype.
- Origin (repo README): re-organised into VOC format from https://github.com/cosmicad/dataset and https://github.com/akshaylamba/all_CELL_data.
- Pink/orange RBCs on a grey-green background, different color and zoom than Data_Set_1. Has heavily overlapping RBC clumps.
- Use: full fields, RBC/WBC/platelet boxes. The realistic test set for counting and for the overlap work.

## Data_Set_2/bccd-kaggle: BCCD subtype labels
- Path: `Data_Sets/Data_Set_2/bccd-kaggle/dataset-master/{JPEGImages,Annotations,labels.csv}` (zip kept next to it). This is the `dataset-master` folder of the Kaggle "Blood Cell Images" dataset (https://www.kaggle.com/datasets/paultimothymooney/blood-cells, Paul Mooney, MIT-style, acknowledges Shenggan/BCCD_Dataset). Confirmed 2026-10-11 from the Kaggle description, which lists `dataset-master` (original images with subtype labels and boxes) and `dataset2-master` (augmented, 4 classes), and quotes 88/33/21/207 images per class, matching our `labels.csv`. Kaggle says 410 images; our copy has 366 JPEGs and 343 XMLs.
- Compared against bccd-shenggan (2026-10-11): 366 vs 364 JPEGs. All 364 shared JPEGs are byte-identical; the extra two are BloodImage_00096 and BloodImage_00329. Only 343 XML files, and they contain **RBC boxes only** (3854); Shenggan's version has WBC and platelet boxes and 4155 RBC boxes, so the two annotation sets differ and Shenggan's is the superset. An earlier version of this file wrongly described the 343 XMLs as having WBC and platelet boxes.
- `labels.csv` (411 rows, `Image` index 0-410 matching `BloodImage_NNNNN`; 45 of those images are not in either folder) gives the WBC subtype per image: 207 neutrophil, 88 eosinophil, 33 lymphocyte, 21 monocyte, 3 basophil, 44 blank, and 15 rows with two labels (e.g. "NEUTROPHIL, EOSINOPHIL"). It covers all 364 Shenggan images; 43 of the 44 blank rows have no image.
- Use: the only source of subtype labels on full fields. Join to bccd-shenggan by image number. Caveat: 6 labelled images have no WBC box in Shenggan's annotations.

## Data_Set_2/dataset2-master: classification set (derived from BCCD)
- Path: `Data_Sets/Data_Set_2/dataset2-master/dataset2-master/images`. **Currently not on disk** (it was removed or never re-downloaded when Data_Set_2 was reorganised on 2026-10-11); `model.py` and `main.py` still point at it. It is the `dataset2-master` folder of the same Kaggle dataset as bccd-kaggle (12,500 augmented images, 4 classes). The lowercase `train`/`val`/`test` symlinks used by `model.py` were made locally and must be recreated after re-downloading.
- Classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL. 240x320 images (cropped/augmented from BCCD, not single-cell crops).
- TRAIN: about 2.5k per class. TEST: about 620 per class. TEST_SIMPLE: 13/6/4/48 images.
- Lowercase symlinks `train`->TRAIN, `val`->TEST, `test`->TEST_SIMPLE (Ultralytics naming), so `model.val(split="test")` uses only the 71 skewed TEST_SIMPLE images.
- Leakage check (2026-10-11), INCONCLUSIVE: about 12k images derive from about 360 originals. A flip-invariant thumbnail comparison showed test images are no closer to train than train images are to each other, and ORB matching against 1500 train images found no exact duplicates (best good-match counts 4-26 of 500 features, a few higher). Rotated/shifted augmentations would defeat both tests. From general knowledge, this classification set was built by augmenting first and splitting after, which would leak (unverified). Safe approach: build our own splits by original BCCD image.

## Data_Set_3: PBC (single-cell images by class)
- Path: `Data_Sets/Data_Set_3/extracted/PBC_dataset_normal_DIB/<class>/*.jpg` (zip kept next to it as `PBC_dataset_normal_DIB.zip`). Licence CC BY 4.0. Source: Acevedo et al. 2020, https://data.mendeley.com/datasets/snkd93bnjr (DOI 10.17632/snkd93bnjr.1; see IDEAS.md section 7).
- 17,093 JPG files (360x363, one cell in the centre), 8 classes: neutrophil 3329, eosinophil 3117, ig (immature granulocyte) 2895, platelet 2348, erythroblast 1551, monocyte 1420, basophil 1218, lymphocyte 1214.
- Loader: dataset name `pbc` or `3`; `Sample.label` holds the class folder name. No masks or boxes.
- Use: WBC subtype classifier (all five subtypes, with far more basophils, lymphocytes and monocytes than BCCD), plus single-cell cut-outs for the synthetic generator. Not for counting or zoom invariance (one fixed zoom, one cell per image, CellaVision stain).
- Caveat: different scanner, stain and background than BCCD, so a classifier trained on PBC alone may not transfer to BCCD-style images; test cross-dataset.

## Notes
- Stain, background and zoom differ strongly between Data_Set_1 and BCCD (see IDEAS.md section 4).
- No dataset has instance masks separating overlapping cells. The overlap method has to supply that.
- Still missing: a real set with WBC subtypes at the higher zoom of Data_Set_1, and a second source of subtype labels with more lymphocytes and monocytes. Basophil is nearly absent (3 images).
