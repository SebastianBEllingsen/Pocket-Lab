# Plan

## Goal
From a blood smear image, count cells, classify red vs. white, then classify white cell subtypes. It must work regardless of microscope zoom and across stain and background color differences. Any technique is allowed (classical, deep learning, or a mix).

## Pipeline
1. **Pre-processing:** background removal, color and stain normalization, blur/noise handling.
2. **Segmentation:** one mask or box per cell (watershed in `main.py` is the starting point).
3. **Scale reference:** per image, take the extreme points of each RBC and normalize them. Store the typical RBC size as the zoom-invariant reference.
4. **Overlap handling:** flag cells whose size deviates from the reference as likely clumps. Estimate the number of cells in the clump (area ratio, and a fit using the reference size), then split or redraw them.
5. **Classification:** RBC vs. WBC, then the WBC subtype. Relative size (WBC size divided by the RBC reference) is a feature.
6. **Output:** per-image counts and a per-cell record list (see ARCHITECTURE.md).

## Small-data strategy
- Classic augmentation: rotation, flips, blur, mild skew, and especially color/stain jitter.
- Synthetic compositing: paste cut-out single cells onto backgrounds at random positions, rotations and zooms. Ground truth is exact, including overlap counts. See IDEAS.md.
- Always keep some real full-field images as the final honest test.

## Roadmap
- [x] 0. Inspect datasets (done 2026-10-11, see DATASETS.md). Still to do: look for more data and check the Data_Set_2 split for leakage.
- [~] 1. Pre-processing: background removal built (awaiting Sebastian's verdict); stain normalization not started
- [ ] 2. Segmentation of cells
- [ ] 3. Per-image scale reference from RBCs
- [ ] 4. Overlap detection and clump counting
- [ ] 5. Cell cut-out extraction (masks) for the synthetic generator
- [ ] 6. Synthetic data generator
- [ ] 7. RBC/WBC and WBC-subtype classifier
- [ ] 8. End-to-end evaluation on real images

## Current status (2026-10-11)
- Step 0 done (datasets inspected; candidate extra datasets in IDEAS.md section 7, frameworks in section 8).
- Step 1 (background removal) BUILT, AWAITING SEBASTIAN'S VISUAL VERDICT: `python -m cellcount.bg_removal`. Mean IoU about 0.89 against Data_Set_1 masks with the default `consensus` estimator (30 random images, seed 5: mean 0.890, min 0.753; the old `mode` rule had min 0.058). BCCD has no masks; the output looked plausible on the 3 images checked, with touching RBCs merged into clumps (expected; splitting is step 2/4).
- One YOLO11n-cls run exists in `runs/classify/train/` (top-1 about 0.90 after epoch 1; CSV incomplete). Do not trust it until the split leakage question is settled.

## Where we left off (2026-10-11)
- Built the shared infrastructure and the bg_removal CLI, with tests (5 passing). Rendered figures were only checked as saved PNGs; the interactive window (toggle keys, check boxes, arrow navigation) has NOT been run by Sebastian yet.
- Waiting on Sebastian: run e.g. `python -m cellcount.bg_removal --full --worst 6 --stats`, `--explain --dataset 2 --boxes --index 7`, and give a verdict (good enough / what looks wrong), per the confer-at-every-step rule.
- FIXED (2026-10-11): image 604478c7 (pale grey, cells packed edge to edge) scored IoU 0.37 because the `mode` estimator took the cell colour as background. New default `consensus` estimator (Sebastian's idea): five proposers vote on the background colour, the majority wins; IoU now 0.90. See DECISIONS.md for the evidence. The viewer shows the votes as colour swatches in `--full` and `--explain`.
- Still open: worst cases left are mostly about what the reference mask counts as a cell (dark stain blob, pale ghost cells, dust, dark corner). Need Sebastian's view on what should count as foreground.
- Viewer fix (2026-10-11): legend and check boxes moved out of the images into a reserved footer strip (check boxes coloured like their outlines when interactive; legend below the image in saved PNGs). Sebastian to confirm.
- Known weak spots to look at: pale/low-contrast images in Data_Set_1 (worst IoU cases), pale WBC cytoplasm and ghost RBCs in BCCD, outline of merged clumps.
- Pending decisions: whether the course requires implementing methods ourselves vs. using frameworks (Cellpose, StarDist, SAM, ...); which extra datasets to download (PBC first, with Sebastian's okay); build our own splits by BCCD original image.
- Next after sign-off: step 2 (segmentation and splitting touching cells), starting with a framework baseline (Cellpose/StarDist) next to a classical watershed, compared with the same viewer; then a rating tool for the golden set.
- Server: Sebastian is requesting access to a machine with (probably) an RTX 4070. Until then, only the local Quadro T2000 (4 GB).
