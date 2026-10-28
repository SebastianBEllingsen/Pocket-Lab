# Evaluation on our own photos (preliminary)

Made by `evaluate.py`. Ground truth: `ground_truth/` (made by hand, never used for training).

Check of the 100x hand counts: 1 boxed squares, mean |hand - boxes| 1.00 -> accepted

## Red blood cells per square (12 test photos, 48 squares)

| Method | Magnification | Squares | Mean abs. error | Bias | Mean % error | Total (manual ± √N) |
|---|---|---|---|---|---|---|
| classical | 40x | 12 | 1.83 | +0.17 | 16.3% | 142 (140 ± 12) |
| classical | 100x | 36 | 4.24 | -2.96 | 9.7% | 1420 (1526 ± 39) |
| classical | all | 48 | 3.64 | -2.18 | 11.4% | 1562 (1666 ± 41) |
| stageA | 40x | 12 | 2.42 | -2.42 | 19.5% | 111 (140 ± 12) |
| stageA | 100x | 36 | 3.62 | -3.46 | 8.5% | 1402 (1526 ± 39) |
| stageA | all | 48 | 3.32 | -3.20 | 11.3% | 1513 (1666 ± 41) |
| stageB | 40x | 12 | 3.08 | -2.25 | 25.5% | 113 (140 ± 12) |
| stageB | 100x | 36 | 5.96 | -5.96 | 13.4% | 1312 (1526 ± 39) |
| stageB | all | 48 | 5.24 | -5.03 | 16.4% | 1425 (1666 ± 41) |

## Red blood cell boxes in the hand-boxed squares

| Method | Matching | Squares | Manual boxes | Precision | Recall |
|---|---|---|---|---|---|
| classical | iou | 12 | 165 | 0.736 | 0.624 |
| classical | centre | 12 | 165 | 0.943 | 0.800 |
| stageA | iou | 12 | 165 | 0.394 | 0.315 |
| stageA | centre | 12 | 165 | 1.000 | 0.800 |
| stageB | iou | 12 | 165 | 0.436 | 0.352 |
| stageB | centre | 12 | 165 | 0.947 | 0.764 |

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
