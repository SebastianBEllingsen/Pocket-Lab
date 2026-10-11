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

- 2026-10-11: Foreground mask uses per-image background estimation plus Otsu (no fixed colour thresholds). Evaluated on Data_Set_1 masks (mean IoU 0.865).
- 2026-10-11: Check open source first, then weigh pros and cons of adopting vs. building our own; both are acceptable and the better one wins (Sebastian). Not a rule against custom tools. Candidates in IDEAS.md section 8, unverified. Wrap behind our own interfaces.
- 2026-10-11: Code conventions agreed (CONVENTIONS.md): package `cellcount/`, flags for every option, file header docstrings, config dataclasses, swappable methods.
- 2026-10-11: Viewer design agreed: contrasting auto outline colours, legend naming each line, layer toggles (o/g/d/b + check boxes), `--minimal/--full/--explain`, `--stats` optional, every parameter a flag.
- 2026-10-11: Sebastian's visual verdict is the sign-off for each step; his ratings will later feed a golden set via a rating tool (not built yet).
- 2026-10-11: Course rules allow any approach (frameworks or our own) as long as the choices are explained in the report. So every significant choice needs its reasoning recorded here (options considered, pros and cons, why chosen, evidence such as scores) to feed the report.

## Open questions
- Overlap idea as training-data preprocessing vs. an inference-time step (being discussed).
- Data_Set_2 classification split leakage: check was inconclusive, see DATASETS.md. Decide whether to build our own split by BCCD original.
- Which of the candidate datasets (IDEAS.md section 7) should be downloaded? Needs Sebastian's okay.
- Server: probably an RTX 4070; access requested, details pending.
- Where to find more data: full fields with subtype labels, more lymphocytes/monocytes/basophils, higher zoom WBC images.
- What would the "gold standard" for the method comparison be?
