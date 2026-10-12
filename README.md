# MitoGuard

**Imposter-aware, scanner-robust mitotic figure detection with a count-reliability flag.**

MitoGuard counts mitotic figures in H&E tumour regions from scanners and tumour types it has never
seen, rejects the look-alike "imposter" cells that cause most false alarms, and flags regions whose
count should not be trusted.

**Results (held-out domains only, MIDOG++):**

| | YOLO11 only | **MitoGuard (+ Phikon-v2 filter)** | MitoGuard (+ Phikon filter) |
|---|---|---|---|
| Unseen tumour type: mean F1, 7 folds | 0.747 | **0.758** (beats baseline on 6/7) | 0.763 (6/7) |
| Unseen scanner: mean F1, 4 folds | 0.560 | **0.598** (3/4) | 0.619 (4/4) |
| Imposters counted as mitoses (unseen tumour) | 6.4% | **5.2%** (−19% relative) | 5.6% |
| Reliability flag AUROC, count error > 20% | 0.679 | **0.731** | 0.700 |

On the MIDOG++ paper's own test regions, the leave-one-tumour-out F1 is **0.746 vs 0.666** for the
published RetinaNet baseline (see [comparison](#comparison-with-published-results) for caveats).

## Why this matters

The mitotic count is a core input to tumour grading in human and veterinary pathology. Models trained in
one lab miscount in another, and they mostly fail by **mistaking imposters for mitoses**: dying cells,
dark nuclei and other look-alikes. In MIDOG++, imposters outnumber true mitoses (14,349 vs 11,937). A lab
cannot install a model trained elsewhere and trust its numbers. It needs a detector that holds up on its
scanner, and a warning when it does not.

## Method

```
H&E region ──► 512 px tiles ──► Stage 1: YOLO11s (classes: mitosis, imposter; H&E stain augmentation)
                                   │  keeps BOTH class scores per box
                                   │  candidate pool: p_mitosis ≥ 0.05  or  p_imposter ≥ 0.10
                                   ▼
            224 px crop ──► zoom to central 112 px ──► frozen Phikon-v2 (ViT-L, pathology FM)
                                   │  feature = [CLS token ‖ mean of the central 4×4 patch tokens]
                                   ▼
            Stage 2: 5-MLP ensemble (Platt-calibrated) → p_filter
                                   │
            learned stacker  p = σ(w·[logit p_filter, logit p_mitosis, logit p_imposter])
                                   │
             ┌─────────────────────┴──────────────────────┐
             ▼                                            ▼
   count = #{p ≥ t}                    reliability  u = Σ p(1−p) / count  →  flag if u ≥ τ
```

* **Stage 1 (candidates).** YOLO11s is fine-tuned on 512×512 tiles with two classes, mitotic figure
  and imposter. H&E stain augmentation jitters each pixel in haematoxylin–eosin optical-density space.
  Inference uses overlapping tiles and keeps **both** class scores for every box. Under domain shift
  YOLO often files true mitoses under its *imposter* class, so the candidate pool also includes those
  boxes. This raises the pool's recall ceiling from ~0.87 to ~0.96.
* **Stage 2 (imposter filter on a frozen foundation model).** Each candidate is embedded once. The key
  design point is the **cell-centred read-out**. At 0.25 µm/px a mitotic figure covers ~50 of 224
  pixels, so the usual CLS token mostly describes the surrounding tissue. Zooming to the central
  112 px and adding the mean of the ViT's central 4×4 patch tokens raises the filter's validation AP
  from 0.77 to 0.90. An MLP ensemble is trained on labelled mitoses, labelled imposters and Stage 1's
  own false positives (hard negatives).
* **Learned stacker.** A 3-feature logistic regression combines the filter with YOLO's two class
  scores. It is fitted on *validation* candidates only, because on its own training cases YOLO is
  over-confident and a stacker fitted there would learn to over-trust it.
* **Count and reliability flag.** The count is the number of kept candidates. Treating each candidate
  as Bernoulli(p), the count variance is Σ p(1−p). Normalised by the count, it is high when many
  borderline candidates support a small count.
* **No peeking.** Within every fold, 20% of the *training-domain* cases form a validation set. That set
  is used for all model selection, feature design, calibration, stacking and thresholds. The held-out
  domain is touched only once, for the final numbers. `docs/experiment_log.md` records every design
  decision together with the validation evidence behind it.

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

## Results

All numbers are on held-out domains. Every threshold, calibration, stacker and design choice was fixed
on validation cases from the training domains. Full tables: [`results/summary.md`](results/summary.md).
Per-fold rows: `results/folds.csv`. Per-region rows: `results/cases.csv.gz`.

### Unseen tumour types (primary protocol)

![F1 gain per unseen tumour type](results/figures/loto_delta_f1.png)

| Held-out tumour | YOLO11 only | + ImageNet ResNet-50 | + Phikon | **+ Phikon-v2 (proposed)** |
|---|---|---|---|---|
| Canine mast cell tumour | 0.814 | 0.804 | 0.830 | **0.826** |
| Canine lymphoma | 0.745 | 0.736 | 0.745 | **0.728** |
| Canine lung carcinoma | 0.621 | 0.620 | 0.651 | **0.635** |
| Canine soft-tissue sarcoma | 0.797 | 0.802 | 0.815 | **0.817** |
| Human breast carcinoma | 0.747 | 0.736 | 0.751 | **0.760** |
| Human melanoma | 0.799 | 0.803 | 0.817 | **0.807** |
| Human neuroendocrine tumour | 0.706 | 0.703 | 0.733 | **0.737** |
| **Mean** | 0.747 | 0.744 | 0.763 | **0.758** |

All three filters share the same candidate pool and learned stacker, so the columns differ only in the
features. **ImageNet features give no gain; pathology foundation-model features do.**

### Unseen scanners (secondary protocol)

![F1 gain per unseen scanner](results/figures/loso_delta_f1.png)

| Held-out scanner | YOLO11 only | + ImageNet ResNet-50 | + Phikon | **+ Phikon-v2** |
|---|---|---|---|---|
| 3D Histech (3 canine tumours, all unseen) | 0.451 | 0.454 | 0.467 | 0.424 |
| Aperio CS2 (breast + mast cell tumour) | 0.326 | 0.321 | 0.477 | 0.441 |
| Hamamatsu S360 (breast only: pure scanner shift) | 0.721 | 0.720 | 0.758 | 0.747 |
| Hamamatsu XR (breast, NET, melanoma) | 0.741 | 0.744 | 0.774 | 0.778 |

### What drives the gain: Stage 2 design ablation

| Stage 2 design (mean held-out F1) | Unseen tumour | Unseen scanner |
|---|---|---|
| YOLO11 only | 0.747 | 0.560 |
| ImageNet ResNet-50 filter alone (brief's design) | 0.651 | 0.635 |
| Phikon CLS-token filter alone (brief's design) | 0.668 | 0.577 |
| Phikon **cell-centred** read-out, filter alone | 0.725 | 0.628 |
| + learned stacker with YOLO scores | 0.757 | 0.614 |
| + wide candidate pool | **0.763** | **0.619** |

The filter as specified in the brief (MLP on the CLS token) is *worse* than YOLO alone. YOLO's explicit
imposter class already removes most imposters, and the CLS token of a 224 px crop describes tissue
context, not the ~50 px cell. Reading out the cell itself (zoom + centre patch tokens) is what makes the
frozen foundation model useful (filter validation AP 0.77 → 0.90).

### Why the foundation model generalises: detector-free probe

A linear probe on frozen features of all 26,286 labelled cells is trained on 6 tumour types and tested on
the 7th (mitosis vs imposter AUROC, mean over 7 held-out tumours):

| Features | AUROC | Scanner predictable from features (breast cancer only, chance 33%) |
|---|---|---|
| ImageNet ResNet-50 | 0.591 | 99% |
| ImageNet ResNet-50, zoomed | 0.643 | 99% |
| Phikon CLS | 0.754 | 100% |
| **Phikon cell-centred** | **0.876** | 100% |
| Phikon-v2 cell-centred | 0.872 | 100% |

Both feature families still encode the scanner perfectly. So the foundation model does **not**
generalise because it ignores the scanner. It generalises because its mitosis-vs-imposter direction stays
consistent across domains.

![UMAP of feature spaces](results/figures/umap_features.png)

### Count-reliability flag

![Reliability flag](results/figures/reliability.png)

u = Σ p(1−p) / count picks out regions whose count is off by more than 20% with AUROC **0.73**
(Phikon-v2), against 0.68 for the same score built on YOLO's own probabilities. Per unseen tumour it
ranges from 0.62 (soft-tissue sarcoma) to 0.88 (lymphoma).

### Stain augmentation (ablation)

H&E optical-density jitter in Stage 1 gave **no consistent benefit**: mean change −0.008 F1 for YOLO
over the 7 tumour folds. In one scanner fold (Aperio CS2), the augmented detector's candidate recall
collapsed (0.49 vs 0.77 without augmentation), on both tumour types scanned there. With a single
training run per setting we cannot separate this from seed variance, so we report it rather than explain it.

### Comparison with published results

| Held-out F1 on the official MIDOG++ test regions | MIDOG++ paper (RetinaNet)¹ | YOLO11 only | MitoGuard + Phikon | MitoGuard + Phikon-v2 |
|---|---|---|---|---|
| Tumour type unseen (mean of 7) | 0.666 | 0.738 | 0.756 | 0.746 |
| In-domain reference (mean of 7) | 0.719 | 0.790 | 0.803 | 0.806 |

* ¹ Aubreville et al. 2023, Tables 4 and 6. Same test regions and matching rule. Their numbers average 5
  cross-validation models. Our leave-one-out models also trained on the test-split regions of the
  *other* six tumours (about 25% more data). The in-domain row uses the same training data as the paper.
* A 2025 MIDOG entry (VM-UNet + stain augmentation, Percannella et al.) reports 0.736 mean F1 under
  leave-one-domain-out over whole held-out domains, with model selection on the held-out domain itself.
  Ours is **0.758–0.763** on whole held-out domains, with no selection on the held-out domain.
* MIDOG challenge leaderboard scores come from hidden test sets and are **not** comparable. We make
  no state-of-the-art claim.
* Most of the margin over the 2023 paper comes from a stronger detector (YOLO11). The two-stage filter
  adds +1.1 to +1.6 F1 on top of that.


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
* Under domain shift the bottleneck is **recall**, not precision (held-out precision ~0.87, recall ~0.68):
  unseen domains make true mitoses look *less* mitotic to every model, so thresholds tuned in-domain are
  too strict. Label-free threshold adaptation is the obvious next step.
* One training run per configuration. Fold-to-fold differences of ~0.01 F1 are within run-to-run noise.
* Some cluster nodes carry GPUs not supported by the pinned PyTorch 2.5.1/cu121 build. Those jobs were
  excluded (`--exclude=cn07`).
* Research tool, not a medical device. No clinical validation.

## References

* Aubreville et al., *A comprehensive multi-domain dataset for mitotic figure detection*, Scientific Data 2023 (MIDOG++).
* Filiot et al., *Scaling self-supervised learning for histopathology with masked image modeling* (Phikon), 2023.
* Lafarge & Koelzer, *Fine-grained hard-negative mining*, MIDOG 2022 (3rd place).
* Ruifrok & Johnston, *Quantification of histochemical staining by color deconvolution*, 2001.

Data: MIDOG++ (CC0). Code: MIT.
