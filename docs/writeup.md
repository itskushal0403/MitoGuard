# MitoGuard: imposter-aware, scanner-robust mitotic figure counting with calibrated uncertainty

**Two-page summary.** Data: MIDOG++ (503 regions, 7 tumour types, human + canine, 4 scanner models,
11,937 mitoses, 14,349 imposters).

## 1. Problem

The mitotic count drives tumour grading, but AI counters trained in one lab miscount in another. Most of
the errors come from "imposters": dying cells and dark nuclei that look like mitoses. Given a region from
an unseen scanner or tumour type, the system must detect mitoses, reject imposters, report the count, and
say when the count should not be trusted.

## 2. Evaluation design

Scanner and tumour type are confounded in MIDOG++ (Cramér's V 0.77). The primary protocol is therefore
leave-one-tumour-out (7 folds), with leave-one-scanner-out (4 folds) as secondary. Every threshold,
calibration and stacker is fitted on validation cases from the training domains. Configurations are
chosen by **nested leave-one-domain-out selection**: a fold's own held-out domain never influences its
configuration. In-domain validation is not enough here, because it favours designs that lean on
in-domain cues. Metrics: official MIDOG matching (7.5 µm), pooled F1, paired bootstrap 95% CIs over
regions, count agreement (CCC, Bland–Altman), quadratic κ on count categories, conformal coverage, ECE.

## 3. Method

1. **Candidates.** Two YOLO11s detectors (classes: mitosis, imposter; with and without H&E stain
   augmentation) propose candidates together. Boxes labelled *imposter* are kept, because under domain
   shift true mitoses often land there.
2. **Foundation-model classifier (main novelty: cell-centred read-out).** At 0.25 µm/px a mitosis covers
   ~50 of 224 px, so a ViT's CLS token mostly encodes tissue context. Zooming to the central 112 px and
   concatenating the mean of the central patch tokens raises frozen-feature validation AP from 0.77 to
   0.90. On top of this read-out, Phikon-v2, Phikon and Lunit DINO are LoRA-fine-tuned with GPU domain
   randomisation (H&E optical-density jitter, photometric, blur, dihedral, translation/scale jitter), with
   8-view TTA, and averaged in logit space.
3. **Detectors propose, the foundation model decides.** The deployed configuration does not fuse in
   YOLO's confidence, which is what degrades under scanner shift.
4. **Calibrated uncertainty.** The count's Poisson-binomial spread is calibrated by normalised
   split-conformal prediction into a 90% interval. A **grade-ambiguity flag** fires when the interval
   straddles a mitotic-count category cut-off (≤7 / 8–14 / ≥15 per ~2 mm²).
5. **Site adaptation (deployment option).** Class-balanced self-training on the new site's unlabelled
   candidates, with a prevalence-aware pseudo-label budget.

## 4. Results (held-out domains)

| | Unseen tumour type | Unseen scanner |
|---|---|---|
| YOLO11 only, F1 | 0.760 | 0.510 |
| v1 (frozen Phikon-v2 filter), F1 | 0.765 | 0.537 |
| **MitoGuard v2**, F1 [95% CI] | **0.778 [0.758, 0.795]** | **0.714 [0.686, 0.740]** |
| Gain over YOLO11 [95% CI] | +1.8 [1.3, 2.3] | +20.5 [17.2, 24.2] |
| Count concordance CCC (YOLO11 → v2) | 0.913 → 0.921 | 0.452 → 0.849 |
| Category κ (YOLO11 → v2) | 0.861 → 0.858 | 0.552 → 0.800 |
| Category accuracy when not flagged | 99.6% | 94.2% |
| Conformal 90% coverage (naive ±1.645σ) | 0.80 (0.57) | 0.76 (0.55) |
| v2 + prevalence-aware site adaptation (post-hoc), F1 | 0.783 | 0.762 |

* The largest gains are where the shift is largest. Aperio CS2 goes from 0.33 to 0.81, 3D Histech
  (unseen scanner + 3 unseen tumours) from 0.45 to 0.62, and the pure scanner shift (Hamamatsu S360,
  breast only) from 0.72 to 0.77.
* Candidate probabilities are well calibrated on unseen tumours: ECE 0.017, vs 0.060 for YOLO's scores.
* On the MIDOG++ paper's own test regions: 0.754 vs 0.666 (published RetinaNet) on unseen tumour types, and
  0.792 vs 0.719 in-domain. A 2025 MIDOG entry reports 0.736 under leave-one-domain-out with selection on
  the held-out domain; MitoGuard v2 reaches 0.771 under nested selection. No leaderboard claim is made:
  challenge test sets are hidden.
* **Why the foundation model generalises.** A detector-free linear probe, trained on six tumours and
  tested on the seventh, separates mitoses from imposters with AUROC 0.88 for the cell-centred Phikon read-out vs 0.59 for ImageNet
  ResNet-50, and wins on all 7 tumours. Yet both feature spaces predict the scanner ~100% of the time. The
  foundation model is not scanner-invariant; its mitosis-vs-imposter direction is more consistent across
  domains.

## 5. Negative results and limitations

* The brief's original Stage 2 (MLP on the CLS token) was *worse* than YOLO alone (0.668 vs 0.747).
* Label-shift EM (Saerens et al. 2002) lowered F1: the shift is a covariate shift, not a label shift.
* Per-region feature centring lowered probe AUROC. Stain augmentation of the detector showed no
  consistent benefit.
* Self-training with a pre-specified fixed budget failed on a low-prevalence site (0.58 vs 0.78). The
  prevalence-aware fix was designed afterwards and needs confirmation on a fresh site.
* Under shift, recall is the bottleneck (0.62–0.70; precision 0.85–0.87), and interval coverage is below
  nominal. These are research results on regions of interest, not a clinical validation.

**Compute.** 24 YOLO11s runs, plus 48 LoRA fine-tuning runs (4 backbones x 12 folds) and 66 self-training runs (A30/A6000 GPUs,
SLURM), about 50 GPU-hours in total.
