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
