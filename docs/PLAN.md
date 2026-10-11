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
- [x] 0. Inspect datasets (done 2026-10-11, see DATASETS.md). PBC added as Data_Set_3. Still to do: check/replace the Data_Set_2 classification split (leakage question), look for more full-field data.
- [~] 1. Pre-processing: background removal BUILT, awaiting Sebastian's visual verdict (he is reading the code first); stain normalization not started
- [ ] 2. Segmentation of cells
- [ ] 3. Per-image scale reference from RBCs
- [ ] 3b. Debris/artifact filter (own step with its own CLI and parameters): drops blobs that are not cells (stain smudges, dust, dark corners) using size and shape relative to the RBC reference. Needs step 3, so it comes after it. Stage 1 stays pure background removal (Sebastian's decision 2026-10-11: split into separate steps to reduce coupling and allow tuning each step on its own).
- [ ] 4. Overlap detection and clump counting
- [ ] 5. Cell cut-out extraction (masks) for the synthetic generator
- [ ] 6. Synthetic data generator
- [ ] 7. RBC/WBC and WBC-subtype classifier
- [ ] 8. End-to-end evaluation on real images

## Current status (end of session 1, 2026-10-11)
- Done: datasets inspected (3 now); shared package `cellcount/` (datasets, selection, metrics, viz); stage 1 `bg_removal` with the `consensus` background estimator, a step-trace and a viewer CLI; 6 unit tests passing; docs and conventions written; local git history (7 commits, never pushed).
- Stage 1 numbers: mean IoU about 0.89 on 30 random Data_Set_1 images (seed 5, min 0.75) against the dataset's reference masks. Worst cases left are about what counts as a cell, not about colour (see DECISIONS.md). BCCD and PBC have no masks: judged by eye only, and looked plausible on the few images checked.
- Not yet signed off: Sebastian has not yet looked at the viewer output (he wants to read the code and methods first). Per the confer-at-every-step rule, stage 1 is not "done" until he gives a verdict.
- Old experiments (`main.py` watershed, `model.py` YOLO11n-cls with one incomplete run in `runs/`) are untouched; do not trust the 0.90 top-1 from that run until the Data_Set_2 split leakage question is settled.

## Where we left off (2026-10-11)
1. Waiting for Sebastian's verdict on stage 1. He should run:
   - `python -m cellcount.bg_removal --full --worst 6 --stats`
   - `python -m cellcount.bg_removal --explain --dataset 2 --boxes --index 7`
   - `python -m cellcount.bg_removal --dataset 3 --random 4 --full`
   and say good enough / what looks wrong, including whether the window layout (check-box strip at the bottom, arrow keys, o/g/d/b toggles) works on his screen. Only saved PNGs and the toggle logic were checked headlessly; the real window has not been seen yet.
2. Decisions made this session (details in DECISIONS.md): `consensus` background colour estimator is the default; stage 1 is pure background removal and a separate step 3b filters debris; ambiguous images are TAGGED (not removed) so they stay available and are scored separately, starting at image level with region-level "ignore" possible later; any approach is allowed if explained in the report, so log reasoning for each choice; check open source first and weigh build vs. adopt; commit locally, never push.
3. Next steps, in order:
   a. Get the stage-1 verdict and fix whatever Sebastian finds.
   b. Build the rating tool (CLI where Sebastian rates good/bad/ambiguous per image, saved to disk) to start the golden set.
   c. Step 2 (segmentation and splitting touching cells): check Cellpose, StarDist (and SAM) licences, install size (4 GB GPU) and fit; compare against a classical distance-transform watershed with the same viewer; present pros/cons to Sebastian before choosing.
   d. Build our own train/test splits by BCCD original image (settles the leakage question).
   e. Steps 3, 3b, 4 (RBC scale reference, debris filter, clump counting), then synthetic generator and classifiers.
4. Open questions for Sebastian: see DECISIONS.md "Open questions" (server access details, further datasets, gold-standard design for the method comparison).
5. Server: Sebastian is requesting access to a machine with (probably) an RTX 4070. Until then, only the local Quadro T2000 (4 GB).
6. Known weak spots to revisit: pale ghost RBCs and pale WBC cytoplasm in BCCD, dust, dark corners/vignette, touching cells merged into one blob (expected until step 2/4), pale low-contrast Data_Set_1 images.
