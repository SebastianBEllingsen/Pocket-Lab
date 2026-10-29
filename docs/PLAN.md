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

## Revised scope and roadmap (2026-10-27)
Scope: this part produces synthetic training data only; a teammate trains and evaluates the detector (his code: branch `origin/cell-detection-v2`, YOLO26, real test crops with hand labels). Roadmap below this section is the original plan; items on classification, method comparison and end-to-end evaluation are out of scope here.

Output contract (to confirm against the teammate's final class list):
- YOLO txt labels, 640x640 tiles, `data.yaml`, `images|labels/{train,val,test}`; synthetic data goes into `train` only.
- Scale: RBC radius 16 px (the teammate normalises all real images to this before detection).
- Detector classes (teammate, confirmed 2026-10-27), in this order: 0 rbc, 1 wbc, 2 platelet, 3 smudge. WBC subtypes are NOT detector classes: a separate type model classifies each wbc crop into basophil, eosinophil, erythroblast, ig, lymphocyte, monocyte, neutrophil (or unknown). The generator therefore writes the 4-class YOLO labels plus a sidecar with the WBC subtype per box, and can export PBC-style single-cell crops with type labels for the type model.
- Real photos: `Data_Sets/own_v2/own_v2/` (unk_40x_*, unk_100x_*, 2880x2160 with a dark eyepiece rim around a bright circle). Stage 1 currently takes the rim as background and the whole circle as foreground, so it needs a field-of-view step for these photos.
- Target look: own eyepiece/phone photos at 40x and 100x (yellow, pink or white backgrounds, white-balance spread, optional eyepiece rim). Real photos not on this machine yet; Sebastian can get them.

Revised steps:
- [x] A. Stage 1 background removal: `lower` threshold and circle finder chosen by Sebastian (2026-10-27). Not usable for cutting cells out of the own photos (faint cells); used for the public datasets and for glass fields.
- [~] B. (built, contact sheets shown, verdict pending) Cut-out bank: RBC (Data_Set_1 isolated cells, BCCD), WBC and platelet (PBC, class kept), smudge (synthesised pale or broken blobs), cell-free backgrounds. Each cut-out stored with mask and source class; rescaled to RBC radius 16 px.
- [~] C. (built, first scenes compared with real photos, verdict pending) Generator v1: random placement, rotation, optical-density blending for overlaps, soft alpha edges, border cells, class ratios, exact boxes. CLI with seed.
- [~] D. (first pass inside the generator: blur, noise, JPEG, white balance, vignette, rim) Realism: white-balance and stain re-tint, blur, noise, JPEG artefacts, vignette, optional eyepiece rim.
- [~] E. (writer built and tested; teammate's exact conventions still to confirm) Export in the teammate's format, previews for Sebastian to check, hand-over notes.
- [ ] F. Check with the teammate whether the synthetic data helps on his real test crops; tune generator from that.

Where we left off (2026-10-27): bank (9,595 cut-outs: 6,318 red cells, 420-446 per WBC type, 297 platelets; 101 glass fields, 56 from own photos), sample dataset of 300 scenes in `Data_Sets/synth_out` (59k rbc, 421 wbc, 1.1k platelet, 293 smudge boxes; Sebastian judged the scenes 'very good', somewhat too clustered, which led to the layout mix), and generator v1 exist. Scenes were compared with real 100x/40x crops: the middle ground looks right (size, density, soft pink cells, WBC, platelets). Next: Sebastian's verdict on scenes (`python -m cellcount.synth --n 6 --show`), confirm box convention and rim handling with the teammate, then the realism pass (hexagonal packing at 100x, rouleaux at 40x, platelet clumps, WBC cytoplasm halo), more smudge variety, bigger bank, and a first hand-over dataset.

## Roadmap (original, 2026-10-11)
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

## Current status (2026-10-29)
- Scope: synthetic training data for the teammate's YOLO detector (see "Revised scope" above). Stage 1 background removal is signed off (`lower` threshold, circle shrink 0.95); cut-out bank, generator and YOLO export are built (`cellcount/synth/`), 15 tests pass, about 20 local commits, nothing pushed.
- Bank: 9,595 cut-outs (6,318 red cells, 420-446 per WBC type, 297 platelets) plus 101 glass fields (56 from the own photos). Smudges are generated.
- Generator: optical-density blending (overlaps darken), per-scene stain, three layouts (random, packed, clumps), teammate's counting-circle rim rule, blur/noise/JPEG. Sebastian's verdict on the scenes: "very good, I would have thought it was real", some scenes too clustered (addressed by the layout mix and lower density).
- Sample dataset: `Data_Sets/synth_out` (600 train + 60 val scenes, git-ignored).
- Experiment (server, see docs/SERVER.md): YOLO26n trained on 600 synthetic scenes only, scored on the teammate's 48 hand-counted real squares: count MAE 4.16 vs 3.64 for his classical counter, box precision/recall at IoU 0.5 0.78/0.75 vs 0.74/0.62 (details in DECISIONS.md). Undercounts about 10% at 100x; predicted red-cell boxes are 37 px against 32 px hand boxes.

## Where we left off (2026-10-29)
0. Latest: clean dataset (5,000 train + 300 val, 1.06 M rbc / 7.7k wbc / 20k platelet / 5.0k smudge boxes) zipped for the teammate (`Data_Sets/synth_handover.zip`, 401 MB). Next: repeat the server experiment on the clean data, scored on his 12 validation squares (not the test squares); ask Sebastian before starting.
1. Waiting for Sebastian to choose the next experiment (he asked to be consulted before runs):
   - option 1: 5,000 scenes + YOLO26s, about 100 epochs (cheap, shows whether data volume closes the gap); recommended first;
   - option 2: realism gap (softer edges and sizes; target the 37 vs 32 px boxes and the 100x undercount);
   - option 3: make smudge and platelet learnable (more and better smudges) and extend the evaluation to wbc/platelet/smudge with his hand boxes;
   - option 4: mix in the teammate's real labelled data (not in the repo; needs his data).
2. Open questions: the exact WBC share wanted in the handover set (default 1.5 WBC per tile; `--wbc-mean` raises it); whether to hand over now or after more tuning.
3. Known weak spots: a few WBC cut-outs carry scraps of neighbouring red cells; smudges are not learned after 40 epochs; own photos have faint cells so stage 1 cannot cut cells out of them; the teammate's own detector weights are not available for comparison.
4. Not yet tried: Poisson blending, SAM-based cut-outs, generative models, per-class scoring on real photos.

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
