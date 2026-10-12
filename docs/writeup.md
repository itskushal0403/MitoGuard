# MitoGuard: imposter-aware, scanner-robust mitotic figure detection

**Two-page summary.** Data: MIDOG++ (503 regions, 7 tumour types, 4 scanner models, 11,937 mitoses,
14,349 imposters).

## 1. Problem

The mitotic count drives tumour grading, but detectors trained in one lab miscount in another. Most of
the errors are "imposters": dying cells and dark nuclei that look like mitoses. Given an H&E region from a
scanner and tumour type unseen in training, the system must detect mitoses, reject imposters, report the
count, and flag regions whose count should not be trusted.

**Research questions.** (i) Does a second-stage imposter filter on a *frozen* pathology foundation model
improve held-out accuracy over a single-stage detector? (ii) Can a simple uncertainty score identify
regions with large count errors?

## 2. Evaluation design

Tabulating cases first showed that scanner and tumour type are strongly confounded (Cramér's V 0.77;
six of seven tumour types were scanned on a single device; the CSV's five scanner names are four devices).
As the brief prescribes, **leave-one-tumour-out (7 folds) is the primary protocol** and
leave-one-scanner-out (4 folds) is secondary. An in-domain model on the official split quantifies the
domain gap. In every fold, 20% of the training-domain cases form a validation set. That set is used for
*all* model selection, feature design, calibration, stacking and thresholds; the held-out domain is
touched once. F1 uses the official MIDOG matching (7.5 µm, `score_detection` from grand-challenge-metrics),
with TP/FP/FN pooled over cases.

## 3. Method

1. **Stage 1.** YOLO11s on 512 px tiles with two classes (mitosis, imposter) and H&E optical-density
   stain jitter. Sliding-window inference keeps *both* class scores per box. The candidate pool takes
   p_mitosis ≥ 0.05 or p_imposter ≥ 0.10: under shift YOLO files many true mitoses under "imposter", and
   including them lifts the pool's recall ceiling from 0.87 to 0.96.
2. **Stage 2.** Each candidate is embedded by a frozen pathology foundation model (Phikon-v2 ViT-L;
   Phikon ViT-B and ImageNet ResNet-50 for comparison). The key finding is the **read-out**. A mitosis
   covers ~50 of 224 px, so the CLS token mostly encodes tissue context. Zooming to the central 112 px
   and concatenating the mean of the central 4×4 patch tokens raises the filter's validation AP from 0.77
   to 0.90. A 5-MLP ensemble is trained on labelled mitoses vs labelled imposters + Stage 1 hard negatives.
3. **Fusion.** A 3-feature logistic stacker on [logit p_filter, logit p_mitosis, logit p_imposter],
   fitted on *validation* candidates only (YOLO is over-confident on its own training cases).
4. **Count and flag.** count = #{p ≥ t}; u = Σ p(1−p) / count; flag if u ≥ τ (τ from validation).

## 4. Results (held-out domains)

| | YOLO11 only | + ImageNet filter | + Phikon | **+ Phikon-v2** |
|---|---|---|---|---|
| Unseen tumour, mean F1 (7 folds) | 0.747 | 0.744 | 0.763 | **0.758** |
| Folds beating YOLO | – | 2/7 | 6/7 | **6/7** |
| Unseen scanner, mean F1 (4 folds) | 0.560 | 0.560 | 0.619 | **0.598** |
| Imposter FP rate (unseen tumour) | 6.4% | 6.0% | 5.6% | **5.2%** |
| Flag AUROC (count error > 20%) | 0.679 | 0.680 | 0.700 | **0.731** |

* **Answer to (i):** yes, but only with the right read-out. The brief's design (MLP on the CLS token)
  *lowers* F1 to 0.668. The cell-centred read-out plus stacking raises it above the baseline. Identical
  stacking with ImageNet features gives no gain, so the improvement is specific to the pathology model.
  Gains are largest where the shift is largest: +3.1 F1 on unseen neuroendocrine tumour, +11–15 F1 on the
  unseen Aperio scanner, and +2.6–3.7 F1 on the S360 fold, the one pure scanner shift (breast cancer only).
* **Answer to (ii):** yes. u reaches AUROC 0.73 (target ≥ 0.7), up from 0.68 with YOLO's probabilities.
  The demo flags an undercounted lymphoma region and leaves a correct breast region unflagged.
* **Why it generalises (detector-free probe):** a linear probe on frozen features, trained on six tumours
  and tested on the seventh, separates mitoses from imposters with AUROC 0.88 for cell-centred Phikon vs
  0.59 for ImageNet ResNet-50, better on all 7 tumours. Yet both feature spaces identify the scanner
  ~100% of the time. The foundation model is *not* scanner-invariant; its mitosis-vs-imposter direction
  is simply more consistent across domains.
* **Context:** on the MIDOG++ paper's own test regions, our leave-one-tumour-out F1 is 0.746–0.756 vs
  0.666 for the published RetinaNet baseline. Our folds had ~25% more training data; the in-domain
  comparison, with equal data, is 0.806 vs 0.719. No state-of-the-art claim: challenge leaderboards use
  hidden test sets.

## 5. Negative results and limitations

* Stain augmentation showed no consistent benefit. In one scanner fold the augmented detector's recall
  collapsed (single seed).
* Per-region feature centring (a label-free test-time adaptation) lowered probe AUROC (0.754 → 0.730).
* The bottleneck under shift is recall (precision ~0.87, recall ~0.68). Thresholds tuned in-domain are
  too strict on unseen domains; label-free threshold adaptation is future work.
* One run per configuration (noise ≈ ±0.005 F1 per fold). Regions of interest only, no whole-slide
  hotspot search. Not a medical device.

**Compute:** 24 YOLO11s runs (30 epochs each, ~35–75 min on A30/A6000), ~16 GPU-h in total, run in parallel
through a resumable SLURM queue. Embedding and Stage 2 took < 2 GPU-h.
