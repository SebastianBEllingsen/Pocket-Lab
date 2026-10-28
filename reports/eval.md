# Evaluation on our own photos (preliminary)

Made by `evaluate.py`. Ground truth: `ground_truth/` (made by hand, never used for training).

Check of the 100x hand counts: 4 boxed squares, mean |hand - boxes| 2.00 -> NOT accepted

## Red blood cells per square (12 test photos, 48 squares)

| Method | Magnification | Squares | Mean abs. error | Bias | Mean % error | Total (manual ± √N) |
|---|---|---|---|---|---|---|
| classical | 40x | 12 | 1.83 | +0.17 | 16.3% | 142 (140 ± 12) |
| classical | 100x | 36 | 4.26 | -3.15 | 9.8% | 1420 (1534 ± 39) |
| classical | all | 48 | 3.66 | -2.32 | 11.4% | 1562 (1674 ± 41) |
| stageA | 40x | 12 | 2.42 | -2.42 | 19.5% | 111 (140 ± 12) |
| stageA | 100x | 36 | 3.82 | -3.65 | 9.0% | 1402 (1534 ± 39) |
| stageA | all | 48 | 3.47 | -3.34 | 11.6% | 1513 (1674 ± 41) |
| stageB | 40x | 12 | 3.08 | -2.25 | 25.5% | 113 (140 ± 12) |
| stageB | 100x | 36 | 6.15 | -6.15 | 13.8% | 1312 (1534 ± 39) |
| stageB | all | 48 | 5.39 | -5.18 | 16.7% | 1425 (1674 ± 41) |
| first_release_asis | 40x | 12 | 11.67 | -11.67 | 100.0% | 0 (140 ± 12) |
| first_release_asis | 100x | 36 | 42.60 | -42.60 | 100.0% | 0 (1534 ± 39) |
| first_release_asis | all | 48 | 34.86 | -34.86 | 100.0% | 0 (1674 ± 41) |
| first_release_pbc | 40x | 12 | 4.25 | -4.08 | 35.7% | 91 (140 ± 12) |
| first_release_pbc | 100x | 36 | 8.74 | -8.74 | 19.7% | 1219 (1534 ± 39) |
| first_release_pbc | all | 48 | 7.61 | -7.57 | 23.7% | 1310 (1674 ± 41) |

## Red blood cells in the hand-boxed squares only (reference: boxes)

Main result: the 100x check rule failed, so only boxed squares have a reliable reference.

| Method | Magnification | Squares | Mean abs. error | Bias | Mean % error | Total (manual ± √N) |
|---|---|---|---|---|---|---|
| classical | 40x | 12 | 1.83 | +0.17 | 16.3% | 142 (140 ± 12) |
| classical | 100x | 4 | 4.50 | -4.50 | 11.0% | 146 (164 ± 13) |
| classical | all | 16 | 2.50 | -1.00 | 15.0% | 288 (304 ± 17) |
| stageA | 40x | 12 | 2.42 | -2.42 | 19.5% | 111 (140 ± 12) |
| stageA | 100x | 4 | 5.50 | -5.50 | 13.6% | 142 (164 ± 13) |
| stageA | all | 16 | 3.19 | -3.19 | 18.0% | 253 (304 ± 17) |
| stageB | 40x | 12 | 3.08 | -2.25 | 25.5% | 113 (140 ± 12) |
| stageB | 100x | 4 | 6.25 | -6.25 | 15.3% | 139 (164 ± 13) |
| stageB | all | 16 | 3.88 | -3.25 | 22.9% | 252 (304 ± 17) |
| first_release_asis | 40x | 12 | 11.67 | -11.67 | 100.0% | 0 (140 ± 12) |
| first_release_asis | 100x | 4 | 41.00 | -41.00 | 100.0% | 0 (164 ± 13) |
| first_release_asis | all | 16 | 19.00 | -19.00 | 100.0% | 0 (304 ± 17) |
| first_release_pbc | 40x | 12 | 4.25 | -4.08 | 35.7% | 91 (140 ± 12) |
| first_release_pbc | 100x | 4 | 8.00 | -8.00 | 19.2% | 132 (164 ± 13) |
| first_release_pbc | all | 16 | 5.19 | -5.06 | 31.6% | 223 (304 ± 17) |

## Red blood cell boxes in the hand-boxed squares

| Method | Matching | Squares | Manual boxes | Precision | Recall |
|---|---|---|---|---|---|
| classical | iou | 16 | 304 | 0.744 | 0.622 |
| classical | centre | 16 | 304 | 0.949 | 0.793 |
| stageA | iou | 16 | 304 | 0.261 | 0.217 |
| stageA | centre | 16 | 304 | 0.992 | 0.826 |
| stageB | iou | 16 | 304 | 0.516 | 0.428 |
| stageB | centre | 16 | 304 | 0.960 | 0.796 |
| first_release_asis | iou | 16 | 304 | 0.000 | 0.000 |
| first_release_asis | centre | 16 | 304 | 0.000 | 0.000 |
| first_release_pbc | iou | 16 | 304 | 0.789 | 0.579 |
| first_release_pbc | centre | 16 | 304 | 0.915 | 0.671 |

## Cells the classical counter called white / smudge, by hand label (all 100x photos)

| Counter said | White cell | Smudge | Not a white cell | Unsure |
|---|---|---|---|---|
| wbc | 57 | 0 | 0 | 5 |
| smudge | 5 | 0 | 7 | 1 |

## White cell types (first-release YOLO11 on PBC-like crops)

Cells the counter called white and the hand label gives a white-cell type. The type model never saw our photos.

| Photos | Correct | Accuracy |
|---|---|---|
| train + val photos | 2 / 43 | 5% |
| test photos | 4 / 14 | 29% |
| all photos | 6 / 57 | 11% |

Rows: hand label, columns: model.

| hand \ model | erythroblast | neutrophil |
|---|---|---|
| eosinophil | 1 | 0 |
| lymphocyte | 38 | 0 |
| monocyte | 4 | 0 |
| neutrophil | 8 | 6 |

Calibration learned on the train + val photos only:

| Model answer | Most often meant (train + val) |
|---|---|
| erythroblast | lymphocyte |
| neutrophil | neutrophil |

Scored on the test photos:

| Method | Correct | Accuracy |
|---|---|---|
| model, raw | 4 / 14 | 29% |
| model + calibration | 11 / 14 | 79% |
| always 'lymphocyte' | 7 / 14 | 50% |
