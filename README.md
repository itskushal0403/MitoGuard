# MitoGuard

**Imposter-aware, scanner-robust mitotic figure detection with a count-reliability flag.**

MitoGuard counts mitotic figures in H&E tumour regions from scanners and tumour types it has never
seen, rejects the look-alike "imposter" cells that cause most false alarms, and flags regions whose
count should not be trusted.

<!-- RESULTS_HEADLINE -->

## Why this matters

The mitotic count is a core input to tumour grading in human and veterinary pathology. Models trained in
one lab miscount in another, and they mostly fail by **mistaking imposters for mitoses**: dying cells,
dark nuclei and other look-alikes. In MIDOG++, imposters outnumber true mitoses (14,349 vs 11,937). A lab
cannot install a model trained elsewhere and trust its numbers. It needs a detector that holds up on its
scanner, and a warning when it does not.

## Method

```
H&E region ──► 512 px tiles ──► Stage 1: YOLO11s (mitosis + imposter classes, H&E stain augmentation)
                                     │  low threshold (p_mitosis ≥ 0.05) → high-recall candidate pool
                                     ▼
                     224 px crop per candidate ──► frozen Phikon ViT-B (pathology foundation model)
                                     │
                                     ▼
                Stage 2: 5-MLP ensemble, Platt-calibrated → p(mitosis) per candidate
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
   count = #{p ≥ t}                       reliability  u = Σ p(1−p) / count  →  flag if u ≥ τ
```

* **Stage 1 (candidates).** YOLO11s fine-tuned on 512×512 tiles with two classes, mitotic figure and
  imposter. Training the imposter class explicitly makes the detector attend to the look-alikes. H&E
  stain augmentation jitters each pixel in haematoxylin–eosin optical-density space. Inference uses
  overlapping tiles and keeps **both** class scores per box, so a mitosis that YOLO labels "imposter" can
  still be rescued by Stage 2.
* **Stage 2 (imposter filter).** Each candidate is embedded once by a frozen pathology foundation model.
  An MLP ensemble is trained on labelled mitoses (positives), labelled imposters, and Stage 1's own false
  positives (hard negatives).
* **Count and reliability flag.** The count is the number of kept candidates. Treating each candidate
  as Bernoulli(p), the count variance is Σ p(1−p). Normalised by the count, it is high when many
  borderline candidates support a small count.
* **No peeking.** Within every fold, 20% of the *training-domain* cases form a validation set. That set
  is used for all model selection, calibration and thresholds. The held-out domain is touched only once,
  for the final numbers.

## Evaluation protocol

The project brief asked us to check scanner/tumour confounding before training (`docs/data_audit.md`).
In MIDOG++ the two are strongly linked (Cramér's V = 0.77). Every tumour type except human breast cancer
was scanned on exactly one device, and the CSV's "5 scanners" are really 4 (one is a spelling variant).
So:

| Protocol | Folds | What it measures |
|---|---|---|
| **Leave-one-tumour-out (primary)** | 7 | new tumour type (often with a new lab/scanner) |
| Leave-one-scanner-out (secondary) | 4 | new scanner; the S360 fold (breast only) is a pure scanner shift |
| In-domain reference | 1 | official train/test split; quantifies the domain gap |

Detection F1 uses the official MIDOG matching (7.5 µm radius, converted per image from the TIFF
resolution) through `score_detection` from `grand-challenge-metrics` (the renamed `evalutils` package
that the MIDOG evaluation calls). TP/FP/FN are pooled over the cases of a set, as in the challenge.

<!-- RESULTS -->

## Repository layout

```
mitoguard/        library: data & folds, tiling, stain augmentation, detector, features, Stage 2,
                  reliability scores, metrics, end-to-end pipeline
scripts/          numbered workflow: download → tiles → train → detect → embed → stage 2 → evaluate → report
app/app.py        Gradio demo (+ example regions from the MIDOG++ test split, CC0)
slurm/            SLURM job scripts used on the cluster
results/          tables (summary.md, folds.csv, cases.csv) and figures
docs/             data audit, experiment log, 2-page write-up
tests/            unit tests for the dataset-independent pieces
```

## Reproduce

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
export MIDOG_ROOT=/data/midogpp MITOGUARD_WORK=/data/mitoguard_work

python scripts/download_midogpp.py --out $MIDOG_ROOT          # ~70 GB, CC0
python scripts/make_tiles.py                                   # 88.5k tiles + pixel spacing
python scripts/data_audit.py                                   # scanner x tumour table, folds
python scripts/train_yolo.py --fold loto_HBC                   # one run per fold (+ --no-stain-aug)
python scripts/detect_all.py                                   # candidates for every finished run
python scripts/embed_crops.py --backbones phikon phikon2 resnet50 phikon_tta
python scripts/run_stage2.py --runs $(ls $MITOGUARD_WORK/cands | sed 's/.pkl//')
python scripts/evaluate.py && python scripts/make_report.py
python scripts/feature_analysis.py --umap
python scripts/export_bundle.py                                # → checkpoints/mitoguard_bundle.pt
python app/app.py --bundle checkpoints/mitoguard_bundle.pt     # demo on http://localhost:7860
pytest                                                         # unit tests
```

On a SLURM cluster: `slurm/train_lanes.sbatch` trains folds in two lanes per GPU, and
`slurm/downstream.sh` runs every later step.

## Limitations

* MIDOG++ regions are pre-selected regions of interest; whole-slide hotspot search is out of scope.
* Reliability AUROC is measured against a 20% relative-error target. For regions with very few mitoses
  that target is strict (one miss in a region with 3 mitoses is 33%).
* Research tool, not a medical device. No clinical validation.

## References

* Aubreville et al., *A comprehensive multi-domain dataset for mitotic figure detection*, Scientific Data 2023 (MIDOG++).
* Filiot et al., *Scaling self-supervised learning for histopathology with masked image modeling* (Phikon), 2023.
* Lafarge & Koelzer, *Fine-grained hard-negative mining*, MIDOG 2022 (3rd place).
* Ruifrok & Johnston, *Quantification of histochemical staining by color deconvolution*, 2001.

Data: MIDOG++ (CC0). Code: MIT.
