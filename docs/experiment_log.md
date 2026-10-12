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
