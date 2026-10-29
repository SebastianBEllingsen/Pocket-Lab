SYNTHETIC BLOOD-SMEAR DETECTION DATASET
=======================================
Made by Sebastian for the cell-counting project. Synthetic training images with exact labels for
every cell, including cells inside clumps. Use it for TRAINING only.

CONTENTS
  images/train, images/val    640x640 JPEG tiles (train 5000, val 300)
  labels/train, labels/val    YOLO .txt, one line per cell: class cx cy w h (normalised 0-1)
  data.yaml                   class names; no absolute path (Ultralytics uses the folder it sits in)
  wbc_types.csv               image, label line number, WBC type of each wbc box
                              (basophil, eosinophil, erythroblast, ig, lymphocyte, monocyte, neutrophil)
  manifest.csv                per-scene settings: zoom, layout (random/packed/clumps), coverage, stain, rim, ...

CLASSES (detector order)   0 rbc   1 wbc   2 platelet   3 smudge

SCALE AND BOXES
  Every scene is drawn at red-cell radius 16 px (median red-cell box 32 px), the same normalisation as the
  detector's own pipeline. Boxes are tight around the whole cell outline (red cell with pallor, white cell with
  cytoplasm, platelet = the dot, smudge = the whole pale patch), one box per cell also in clumps.
  Some scenes contain an eyepiece edge: everything outside the counting circle (93% of the eyepiece radius
  minus one red-cell radius) is black and unlabelled; cells whose box centre is inside it are labelled.

HOW IT WAS MADE
  Real single-cell cut-outs (soft alpha edge) are pasted onto glass-colour fields and blended as optical
  density, so overlapping cells darken like in a real smear. Layouts: random, tightly packed (100x look),
  small clumps (40x look). Then stain jitter, blur, noise, white balance, vignette, JPEG compression.
  Red cells: Data_Set_1 (RBC masks) and PBC neighbour cells. White cells and platelets: PBC. Smudges are
  generated (pale ragged purple blobs). Glass colours: estimated from the 37 own phone photos that are NOT in
  the validation or test list (no photo of unk_40x_01-03, 40x_08-10, unk_100x_01-03, 100x_17-26 was used).
  No test photo or hand-labelled square was used for anything in this dataset.

CAVEATS (honest list)
  - Synthetic cells are cleaner than real ones; boxes predicted on real photos came out about 15% larger
    than hand boxes in a first test (37 px vs 32 px), so a real/synthetic gap exists.
  - A few white-cell cut-outs carry a scrap of a neighbouring red cell.
  - Smudges and platelets are the least realistic classes (a first 40-epoch nano run did not learn smudges).
  - WBC share is the natural one (about 1.4 per scene); ask for a WBC-heavy set if you want more.

SUGGESTED USE
  Run A: your real data only.   Run B: real + this dataset (keep your own val/test real).
  Compare A and B on the validation squares; report the chosen setup once on the test squares.

SOURCES AND LICENCES (attribution)
  PBC dataset: Acevedo et al. 2020, Mendeley Data, DOI 10.17632/snkd93bnjr.1, CC BY 4.0.
  Data_Set_1 (RBC segmentation masks): licence not yet verified; check before publishing the images.
  Reproduce: the generator is in Sebastian's repository (python -m cellcount.synth, see its README).
