# Decisions and Open Questions

## Decisions
- 2026-10-11: One cell list with type labels, not separate red/white lists. Red and white views are filters.
- 2026-10-11: Scale reference is computed from RBCs only, using the median.
- 2026-10-11: Relative cell size is used as a classification feature.
- 2026-10-11: Overlap handling estimates the number of cells in a clump (counting is the goal).
- 2026-10-11: No technique constraints from the course (classical, deep learning or mixed all allowed).
- 2026-10-11: Both classic augmentation and synthetic compositing will be pursued. Real images are kept for the final test.
- 2026-10-11: Experiment with both approaches (detector and segmentation + classifier) and compare them. Possibly add a gold-standard reference and cluster or compare the methods' outputs. Not yet designed.
- 2026-10-11: Target is all WBC subtypes the data allows (best case: all five), plus platelets if available.
- 2026-10-11: Hardware: local Quadro T2000 (4 GB) and a training server (details not yet known). Deadline is not a concern for now; focus on the project.
- 2026-10-11: Datasets inspected. Data_Set_1 is RBC masks, BCCD has boxes and subtype labels (see DATASETS.md).

- 2026-10-11: Foreground mask uses per-image background estimation plus Otsu (no fixed colour thresholds). Evaluated against Data_Set_1 reference masks (see the decision record below for numbers).
- 2026-10-11: Check open source first, then weigh pros and cons of adopting vs. building our own; both are acceptable and the better one wins (Sebastian). Not a rule against custom tools. Candidates in IDEAS.md section 8, unverified. Wrap behind our own interfaces.
- 2026-10-11: Code conventions agreed (CONVENTIONS.md): package `cellcount/`, flags for every option, file header docstrings, config dataclasses, swappable methods.
- 2026-10-11: Viewer design agreed: contrasting auto outline colours, legend naming each line, layer toggles (o/g/d/b + check boxes), `--minimal/--full/--explain`, `--stats` optional, every parameter a flag.
- 2026-10-11: Sebastian's visual verdict is the sign-off for each step; his ratings will later feed a golden set via a rating tool (not built yet).
- 2026-10-11: Course rules allow any approach (frameworks or our own) as long as the choices are explained in the report. So every significant choice needs its reasoning recorded here (options considered, pros and cons, why chosen, evidence such as scores) to feed the report.

### Decision record: how to pick the background colour (2026-10-11)
- Problem: the single "most common colour" rule fails when cells cover most of the frame (pale grey Data_Set_1 image 604478c7: IoU 0.37, 96% marked foreground).
- Options considered: (a) median filter of the image (`median`); (b) most common colour (`mode`); (c) let several independent proposers vote and take the colour most of them agree on (`consensus`, Sebastian's idea). Proposers: mode, bright (lightest large cluster), kmeans (lightest of 3 clusters), border (frame colour), neutral (least colourful large cluster).
- Pros/cons: median is simple but wrong with dense cells. Mode is simple and usually right, but has a clear failure case. Consensus is robust to any single proposer failing, and the votes are visible in the viewer; it costs about 5x the colour-finding time (small, done at 0.25 scale) and needs the proposers to fail in different ways. Ties or no agreement fall back to the first proposer and are flagged.
- Chosen: `consensus` as default. 
- Evidence (30 random Data_Set_1 images, `--random 30 --seed 5`): mean IoU median 0.789 (min 0.090), mode 0.864 (min 0.058), consensus 0.890 (min 0.753). Image 604478c7: 0.368 -> 0.904 (mode was outvoted 4 to 1). Reproduce: `python -m cellcount.bg_removal --method {median,mode,consensus} --random 30 --seed 5 --stats`.
- Remaining low scores (consensus, `--worst 5 --pool 150 --seed 1`: 0.46, 0.59, 0.71, 0.74, 0.74) are mostly label-policy questions rather than colour errors: a huge dark stain blob the reference mask leaves out (c6ab535c), and sparse pale/ghost cells, dust and a dark corner that we outline but the reference does not (ebb53337). To discuss with Sebastian.

- 2026-10-11: Ambiguous images (where the reference mask and the image disagree about what counts as a cell) are TAGGED, not removed, so they stay available for training and other stages and are scored separately (Sebastian agreed). Start at image level; region-level "ignore" masks can come later if needed.
- 2026-10-11: PBC (Data_Set_3) added with a `pbc`/`3` loader; image content and counts in DATASETS.md.
- 2026-10-11: Git: local commits only, never push; data, runs and weights are ignored.

## Open questions
- Overlap idea as training-data preprocessing vs. an inference-time step (discussed; leaning: build and validate the splitter first, then decide; BCCD boxes and synthetic data give the validation).
- Data_Set_2 classification split leakage: check was inconclusive, see DATASETS.md. Decide whether to build our own split by BCCD original.
- PBC downloaded by Sebastian as Data_Set_3 and extracted (2026-10-11). Other candidate datasets still need his okay.
- DECIDED 2026-10-11 (Sebastian): stage 1 stays pure background removal (foreground = everything unlike the background). "Is it a cell?" is a separate later step (3b, debris filter) with its own CLI and parameters, so each step can be tuned independently. Reason: less coupling, and removing something early cannot be undone. Consequence: scores against the Data_Set_1 reference mask understate stage-1 quality on images with debris; images with a policy dispute can be marked ambiguous in the golden set. Still to confirm: whether to mark them ambiguous.
- Server: probably an RTX 4070; access requested, details pending.
- Where to find more data: full fields with subtype labels, more lymphocytes/monocytes/basophils, higher zoom WBC images.
- What would the "gold standard" for the method comparison be?
