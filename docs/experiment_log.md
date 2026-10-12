# Experiment log

Chronological notes on decisions and negative results (things tried and not adopted).

- **Protocol.** Scanner and tumour type are strongly linked (Cramer's V 0.77, see `data_audit.md`), so LOTO is
  primary and LOSO secondary, as the brief prescribes.
- **Data loading.** The first YOLO run trained at 36 tiles/s: numpy BLAS threads inside 12 dataloader workers
  oversubscribed the CPU. Rewriting the HED jitter as a single fused 3x3 optical-density transform without
  BLAS and pinning `OMP_NUM_THREADS=1` gave ~365 tiles/s (10x).
- **Per-region feature centring (rejected).** Phikon features cluster by domain (see UMAP), so we tried a
  label-free test-time adaptation: subtract the mean embedding of all crops in the region. In the
  leave-one-tumour-out linear probe it *lowered* mitosis-vs-imposter AUROC (Phikon 0.754 -> 0.730,
  ResNet-50 0.591 -> 0.561), probably because a region's mean embedding also encodes how mitotically active
  it is. Not used.
- **Interim result, 4 LOTO folds (HBC, CCMCT, CLC, HNET).** The brief's Stage 2 (MLP on Phikon's CLS token of
  a 224 px crop) was *worse* than YOLO alone (mean held-out F1 0.650 vs 0.722; val F1 0.716 vs 0.822). The
  YOLO baseline is strong because its explicit imposter class already keeps imposter FP at ~5%.
  Diagnosis: at 0.25 um/px the candidate cell covers ~50 of 224 px; the CLS token summarises the
  surrounding tissue more than the cell.
- **Cell-centred read-out of the frozen FM (adopted).** Selected on *validation* AP of the Stage 2 MLP
  (training-domain cases only; mean of 4 folds):

  | feature | val AP |
  |---|---|
  | Phikon CLS @224 (brief) | 0.765 |
  | Phikon CLS, zoom 112 | 0.783 |
  | Phikon CLS + centre 4x4 tokens | 0.882 |
  | Phikon CLS + centre tokens, zoom 112 | 0.898 |
  | Phikon CLS + centre tokens, zoom 80 / 144 | 0.890 / 0.901 |
  | Phikon + dihedral TTA, zoom 112, centre tokens | ~0.904 |
  | Phikon-v2 CLS + centre tokens, zoom 112 | 0.900 |
  | ImageNet ResNet-50 / zoom 112 | 0.626 / 0.679 |

- **Fusion with YOLO (adopted: learned stacker).** A 3-feature logistic regression on
  [logit p_filter, logit p_yolo_mitosis, logit p_yolo_imposter], fitted on validation candidates only
  (YOLO's scores on its own training cases are over-confident and would leak). The fixed sqrt(p_yolo*p_filter)
  fusion is reported for comparison.
- **Wider candidate pool (adopted).** Under domain shift YOLO files many true mitoses under its *imposter*
  class: adding boxes with p_imposter >= 0.10 raises the pool's recall ceiling (val 0.95 -> 0.98).
- **Final configuration, fixed before running the remaining folds.** Top validation F1 (0.830):
  YOLO11s + Phikon-v2 (zoom 112, CLS + centre tokens) + wide pool + learned stacker. The top dozen
  configurations differ by <= 0.003 val F1, so the choice also leans on the validation-AP and pool-recall
  evidence above. Phikon (ViT-B, the brief's model) is reported alongside.
- **Observation.** On held-out tumours precision stays high (~0.84) while recall drops (~0.66): the domain
  shift mainly makes true mitoses look *less* mitotic to every model, so a threshold tuned in-domain is
  too strict. Label-free threshold adaptation is future work.
- **Final pass (all 24 detectors).** The crop bank was rebuilt with the candidates of all 24 Stage 1 models
  and every Stage 2 filter retrained. Small shifts from the 17-run interim pass (e.g. Phikon-v2 LOTO mean
  F1 0.764 -> 0.758; it now beats the baseline on 6/7 folds instead of 7/7) give a sense of run-to-run
  noise: about +-0.005 F1 per fold. Phikon (ViT-B) and Phikon-v2 (ViT-L) are within that noise of each
  other. The headline system stays the one fixed in advance (Phikon-v2); Phikon is reported alongside.
- **Stain augmentation.** No consistent effect on LOTO (YOLO -0.008 +- 0.012). On the Aperio CS2 LOSO
  fold the augmented detector's candidate-pool recall collapsed (0.49 vs 0.77 without augmentation;
  precision ~0.98 in both), on both breast cancer and mast cell tumour. Single-seed result: worth
  re-running with several seeds before drawing conclusions.
- **Comparison with literature.** MIDOG++ paper (RetinaNet, Table 6) leave-one-out F1 on the official test
  regions: mean 0.666. Ours on the same regions: YOLO11 0.738, + Phikon 0.756, + Phikon-v2 0.746. See README
  for the caveats (more training data in our LOTO folds; 5-model CV averages in the paper).

## v2 (extension, 2026-10-13)

- **Bottleneck analysis.** Under domain shift the v1 system is precise (~0.87) but misses mitoses
  (recall ~0.68) while the candidate pool still contains ~96% of them: the classifier is the bottleneck.
- **Detector union (adopted).** Each fold already had two Stage 1 detectors (with / without stain
  augmentation). Merging their candidates (8 px, max of class scores) and widening the pool to any box with
  max(p_mitosis, p_imposter) >= 0.05 raises the held-out recall ceiling: HNET 0.922 -> 0.980,
  CS2 0.892 -> 0.966, 3DH 0.781 -> 0.889, CLC 0.967 -> 0.994.
- **Label-shift EM (Saerens et al. 2002), rejected.** Re-estimating the mitosis prior per region or per
  site lowered F1 on HNET (stack 0.733 -> 0.694 per region, 0.654 per site; precision up to 0.97, recall
  down to 0.49). The shift is a *covariate* shift: true mitoses look less mitotic in the new domain, so
  their scores fall and EM reads that as fewer mitoses, pushing posteriors further down. Kept as a
  documented negative result. Motivates adapting the classifier itself (self-training) instead.
- **Clinical metrics on v1 (LOTO, 503 regions).** Count CCC 0.92, Spearman 0.92, quadratic kappa on
  <=7 / 8-14 / >=15 categories 0.86. Normalised split-conformal 90% intervals reach 0.80 coverage on unseen
  tumours (naive Poisson-binomial +-1.645 sigma: 0.54). Regions whose interval does not straddle a category
  cut-off have 99% category accuracy; the flag catches 97-99% of category errors but flags ~56% of regions.
- **Backbone throughput (LoRA r=16, bf16, A30, batch 64).** Phikon ViT-B 510 img/s, Lunit ViT-S/8 305 img/s,
  Phikon-v2 ViT-L 174 img/s; Kaiko Midnight ViT-g did not finish the benchmark in 14 min (too heavy for the
  24 h budget across 12 folds).
