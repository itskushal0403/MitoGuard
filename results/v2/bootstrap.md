# Paired bootstrap 95% confidence intervals (held-out regions)

Paired bootstrap confidence intervals over held-out regions (pooled TP/FP/FN F1, as in MIDOG).

Regions are resampled with replacement (2,000 draws) within each protocol; every system is evaluated on
the same resampled regions, so differences are paired. Reports F1 [95% CI] and the F1 gain over the
YOLO11 baseline [95% CI] with a one-sided bootstrap p-value P(gain <= 0).

| protocol   | system                             |   regions |     F1 | 95% CI         |   gain vs YOLO11 | gain 95% CI     |   p (gain <= 0) |
|:-----------|:-----------------------------------|----------:|-------:|:---------------|-----------------:|:----------------|----------------:|
| loto       | yolo                               |       503 | 0.7601 | [0.740, 0.778] |         nan      | nan             |        nan      |
| loto       | MitoGuard-v2                       |       503 | 0.7779 | [0.758, 0.795] |           0.0178 | [0.013, 0.023]  |          0      |
| loto       | MitoGuard-v2-SA                    |       503 | 0.7724 | [0.753, 0.789] |           0.0123 | [0.007, 0.018]  |          0      |
| loto       | MitoGuard-v2-SApa                  |       503 | 0.783  | [0.764, 0.800] |           0.0229 | [0.018, 0.028]  |          0      |
| loto       | yolo+phikon2-z112-ctr@m05i10+stack |       503 | 0.7652 | [0.742, 0.785] |           0.0051 | [-0.001, 0.010] |          0.0485 |
| loso       | yolo                               |       503 | 0.5096 | [0.467, 0.552] |         nan      | nan             |        nan      |
| loso       | MitoGuard-v2                       |       503 | 0.7143 | [0.686, 0.740] |           0.2047 | [0.172, 0.242]  |          0      |
| loso       | MitoGuard-v2-SA                    |       503 | 0.623  | [0.591, 0.652] |           0.1134 | [0.081, 0.146]  |          0      |
| loso       | MitoGuard-v2-SApa                  |       503 | 0.7624 | [0.744, 0.779] |           0.2528 | [0.214, 0.292]  |          0      |
| loso       | yolo+phikon2-z112-ctr@m05i10+stack |       503 | 0.5369 | [0.495, 0.579] |           0.0273 | [0.014, 0.039]  |          0      |