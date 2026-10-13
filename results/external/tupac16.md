# External validation on TUPAC16 (73 breast cancer cases, never used for training or tuning)

External validation on TUPAC16 (auxiliary mitosis set, 73 breast cancer cases; never seen in training).

Step 1 (GPU, --predict): run YOLO11, MitoGuard v1 and MitoGuard v2 (all trained on the official MIDOG++
  training split only; thresholds fixed on MIDOG++ validation, no TUPAC16 tuning) on every TUPAC16 image
  and cache the outputs.
Step 2 (CPU, --score): match against three label sets with the official MIDOG criterion (7.5 um radius,
  30 px at 0.25 um/px, grand-challenge-metrics score_detection), pooled F1, case-level bootstrap 95% CIs:
    original     original TUPAC16 ground truth (1,552 mitoses)
    AL-manual    Bertram et al. 2020 alternative labels, manual consensus (1,898)
    AL-augmented Bertram et al. 2020 alternative labels incl. model-assisted, expert-confirmed (1,999)

Images: 656; cases: 73.

| labels       |   label count | system   |     F1 | 95% CI         |   precision |   recall |   TP |   FP |   FN |
|:-------------|--------------:|:---------|-------:|:---------------|------------:|---------:|-----:|-----:|-----:|
| original     |          1552 | yolo     | 0.6653 | [0.629, 0.699] |      0.6209 |   0.7165 | 1112 |  679 |  440 |
| original     |          1552 | v1       | 0.6752 | [0.630, 0.724] |      0.5871 |   0.7945 | 1233 |  867 |  319 |
| original     |          1552 | v2       | 0.6573 | [0.601, 0.715] |      0.54   |   0.8396 | 1303 | 1110 |  249 |
| AL-augmented |          1999 | yolo     | 0.733  | [0.700, 0.759] |      0.7755 |   0.6948 | 1389 |  402 |  610 |
| AL-augmented |          1999 | v1       | 0.7538 | [0.726, 0.778] |      0.7357 |   0.7729 | 1545 |  555 |  454 |
| AL-augmented |          1999 | v2       | 0.7412 | [0.710, 0.772] |      0.6776 |   0.8179 | 1635 |  778 |  364 |
| AL-manual    |          1898 | yolo     | 0.7335 | [0.700, 0.759] |      0.7554 |   0.7129 | 1353 |  438 |  545 |
| AL-manual    |          1898 | v1       | 0.7509 | [0.722, 0.777] |      0.7148 |   0.7908 | 1501 |  599 |  397 |
| AL-manual    |          1898 | v2       | 0.7344 | [0.701, 0.769] |      0.656  |   0.834  | 1583 |  830 |  315 |

## Diagnostic: oracle-threshold F1 (threshold tuned on TUPAC16 itself; NOT a valid performance claim)

Separates ranking quality from threshold transfer. All reported results above use thresholds fixed on MIDOG++ validation.

| labels | YOLO11 | MitoGuard v1 | MitoGuard v2 |
|---|---|---|---|
| AL-augmented | 0.745 (t=0.40) | 0.754 (t=0.40) | 0.750 (t=0.50) |
| AL-manual | 0.742 (t=0.40) | 0.751 (t=0.43) | 0.748 (t=0.53) |

## Comparison with published results on the same external set

"Beyond Classification: Pathology Foundation Models as Detection Encoders for Mitotic Figures" (arXiv 2607.28007,
2026) trains on MIDOG++ and tests on the relabeled TUPAC16 training set (71 cases; label variant not specified):
best F1 0.7349 (H-optimus-0 + RetinaNet), fine-tuned ResNet-50 + RetinaNet 0.7193, best UNI2-h 0.7188.
MitoGuard (73 cases, MIDOG++-validation thresholds): v1 0.754 / 0.751, v2 0.741 / 0.734 (augmented / manual labels).
