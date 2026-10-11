# Ideas, Problems and Solutions

## 1. Overlap counting via size statistics (user's idea)
Detectors tend to draw one box around two overlapping cells. Instead: measure each cell's normalized extreme points, build the per-image distribution, flag outliers as clumps, estimate the count, and redraw using the image's reference size.
- Use the median of RBC sizes as the reference (robust to the clumps themselves).
- Count estimate: clump area divided by single-cell reference area, rounded, with a correction because overlapping cells share area. Refine by fitting N reference-sized circles/ellipses to the clump (e.g. distance-transform peaks, or optimizing circle positions).
- Splitting alternatives: watershed on the distance transform, concave-point splitting of the contour.

## 2. Augmentation
Rotation, flips, blur, noise, mild skew, brightness and contrast. Color/stain jitter (hue, saturation, per-channel shifts) is the most useful for the cross-dataset color problem. Avoid heavy shape distortion; it destroys the morphology that separates WBC subtypes. Augmentation does not add new information.

## 3. Synthetic compositing (user's idea)
Paste cut-out single cells onto backgrounds at random positions, rotations and zoom levels, then apply the augmentations on the composite. Gives exact boxes, counts and overlap labels for free.

Known problems and the planned solution for each:
| problem | solution |
|---|---|
| Domain gap: model learns pasting artifacts, not cells | Soft alpha edges; add noise, blur and JPEG-like artifacts after compositing; mix in real images; validate on real images only |
| Overlap realism: real overlapping RBCs get darker where they overlap | Multiply-style (optical density) blending, not plain occlusion |
| Need clean single-cell cut-outs with masks | Use our own segmentation to extract them; manually check a sample |
| Backgrounds | Real cell-free background patches, inpaint cells out of real smears, or synthesize from estimated background color/texture. Randomize across stains. |
| Cutouts carry their source stain color | Normalize cut-outs, then re-tint randomly at composition time |
| Class balance and realism (RBC far outnumber WBC, about 1000:1 in healthy smears) | Control ratios in the generator; oversample WBCs for training but test on realistic ratios |
| Cells touching the image border, partial cells | Let the generator place cells partially outside the frame |

## 4. Stain and color normalization
Stain and background color differ between datasets. Options: per-image white-balance / background normalization, histogram matching to a reference, Reinhard (LAB statistics) normalization, Macenko stain normalization, or grayscale/optical-density input. Start simple (background normalization + LAB statistics) and compare.

## 5. Size as a classification feature
Relative size (WBC size / RBC reference) helps separate subtypes (e.g. lymphocytes are smaller than monocytes). It is zoom-invariant by construction.

## 7. Candidate extra datasets (found by web search 2026-10-11; nothing downloaded, licences unchecked)
- PBC dataset (Acevedo et al. 2020, Mendeley Data DOI 10.17632/snkd93bnjr.1, page: https://data.mendeley.com/datasets/snkd93bnjr): 17,092 single-cell images, 360x363 JPG, CellaVision DM96 at Hospital Clinic Barcelona, 8 classes (neutrophil, eosinophil, basophil, lymphocyte, monocyte, immature granulocyte, erythroblast, platelet). Licence CC BY 4.0 (read on the Mendeley page, 2026-10-11). Sebastian approved the download 2026-10-11; the page is a web app with no scriptable direct link, so it must be downloaded by hand into `Data_Sets/PBC/`. Also see: WBCAtt adds morphological attributes (arXiv 2306.13531). Single-cell images, good for subtype classification, with many more basophils/lymphocytes/monocytes than BCCD.
- WBC detection dataset, Liangzun Fu, IEEE DataPort (DOI 10.21227/qxk8-s148): five subtypes with boxes. May need an IEEE DataPort subscription.
- Scientific Data 2023 dataset (s41597-023-02378-7): 16,027 annotated WBCs, 9 classes, varied lighting and artifacts.
- DATA-CELLSEG (Figshare 23804523): instance segmentation + WBC classification (PBC and RV-PBS). Closest hit for instance masks; need to check what it labels.
- Overlap papers: arXiv 2012.01321 (ellipse-based RBC overlap separation); SciTePress 2022 (synthetic overlaps from segmented free-standing RBCs, same idea as ours).
- Warning from a 2024 arXiv audit: BCCD has missing labels and heavy overlap, so do not treat its boxes as perfect ground truth.
- Caveat: PBC is single-cell crops at one fixed zoom, so it helps classification but not counting or zoom invariance.

## 8. Open-source frameworks to evaluate instead of building from scratch
Requested by Sebastian 2026-10-11: check existing open source first and modify it where it fits. Building our own is equally valid when it is the better option; weigh pros and cons for each stage. The list below is from general knowledge and has NOT been verified this session (versions, licences, maintenance, install size on a 4 GB GPU). Check each before adopting. Each would sit behind one of our stage interfaces (adapter), so it can be compared with our own methods.

| need | candidates | notes |
|---|---|---|
| Cell instance segmentation (splitting touching cells) | Cellpose, StarDist | Purpose-built for separating touching cells; pretrained models exist. Strong candidate for steps 2 and 4 and for auto-labelling overlaps |
| Promptable / generic segmentation | Segment Anything (SAM / SAM 2) | Useful for cut-outs for the synthetic generator and for label bootstrapping |
| Classical image ops, watershed, morphology, regionprops, Hough | scikit-image, SciPy, OpenCV (already used) | Cheap, no GPU; good for the classical-pipeline baseline |
| Detection and classification | Ultralytics YOLO (already installed); torchvision detection; MMDetection / Detectron2 | Detector baseline for the method comparison |
| Augmentation | Albumentations, Kornia (GPU) | Rotation, blur, colour/stain jitter, elastic; replaces hand-written augmentation |
| Stain / colour normalization | torchstain, staintools (Macenko, Reinhard, Vahadane) | Directly addresses the cross-dataset colour problem |
| Synthetic compositing | Albumentations + custom paste logic; possibly existing copy-paste augmentation implementations | The paste logic with optical-density blending is likely custom |
| Experiment tracking / comparison | MLflow, Weights and Biases (free tier), or plain CSV | For the method-comparison idea |
| Annotation and rating tools | Label Studio, CVAT, or our own small CLI | For the golden set; our own rating CLI is planned |

Decision rule: for each stage, try the framework as a baseline first, and only write custom code where it falls short or where the course asks us to implement the method ourselves (to confirm with Sebastian per stage, since this is a course project).

## 6. Open idea space
- Learned segmentation (e.g. Cellpose, StarDist, SAM) as an alternative or complement to classical watershed. Not yet evaluated.
- Detector (YOLO-style) vs. classical segmentation + classifier. Undecided.
