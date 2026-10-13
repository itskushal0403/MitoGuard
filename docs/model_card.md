# Model card: MitoGuard v2

**Research software. Not a medical device. Not validated for clinical use.**

## Intended use

* **Task.** Count mitotic figures in a pre-selected H&E region of interest (~2 mm², level 0 at
  ~0.23–0.25 µm/px, i.e. 40×). For each region it outputs a 90% count interval, a mitotic-count category
  (≤7 / 8–14 / ≥15 per region) and a grade-ambiguity flag.
* **Intended user.** A pathologist or researcher who reviews the detections. MitoGuard is meant to
  prioritise review (flagged regions) and reduce the burden of rejecting imposters, not to replace the
  pathologist.
* **Out of scope.** Whole-slide hotspot selection, atypical mitosis subtyping, non-H&E stains,
  magnifications other than 40×, cytology, frozen sections.

## Model

| Component | Details |
|---|---|
| Candidate detectors | 2 × YOLO11s (two classes: mitosis, imposter), with and without H&E stain augmentation; candidates merged |
| Cell classifier | Ensemble of Phikon-v2 (ViT-L/16), Phikon (ViT-B/16) and Lunit DINO (ViT-S/8), each LoRA-fine-tuned (r=16) on cell-centred crops; 8-view TTA; logit averaging; Platt calibration |
| Count | candidates with calibrated p ≥ threshold (tuned on training-domain validation) |
| Interval | normalised split-conformal on Poisson-binomial spread, 90% nominal |
| Training data | MIDOG++ (CC0): 503 regions, 7 tumour types (human + canine), 4 scanner models, 5 labs; 11,937 mitoses, 14,349 imposters |
| Demo weights | trained on the official MIDOG++ training split (392 regions) |

## Performance (held-out domains, never used for any choice)

| | Unseen tumour type (7 folds) | Unseen scanner (4 folds) |
|---|---|---|
| Detection F1 [95% CI] | 0.778 [0.758, 0.795] | 0.714 [0.686, 0.740] |
| YOLO11 baseline F1 | 0.760 | 0.510 |
| Count concordance (CCC) | 0.921 | 0.849 |
| Category κ (quadratic) | 0.858 | 0.800 |
| Category accuracy, unflagged regions | 99.6% | 94.2% |
| 90% interval coverage (nominal 0.90) | 0.80 | 0.76 |
| Imposters counted as mitoses | 5.1% (YOLO11: 6.4%) | 4.7% (YOLO11: 2.0% at recall 0.36) |
| Candidate ECE | 0.017 | – |

In-domain test (official MIDOG++ test split, demo model): F1 0.809.

### Performance by subgroup (F1, held-out)

| Domain | F1 | | Domain | F1 |
|---|---|---|---|---|
| Canine mast cell tumour | 0.812 | | 3D Histech scanner (canine lung, lymphoma, sarcoma) | 0.620 |
| Canine lymphoma | 0.758 | | Aperio CS2 scanner | 0.812 |
| Canine lung carcinoma | 0.658 | | Hamamatsu S360 scanner | 0.771 |
| Canine soft-tissue sarcoma | 0.822 | | Hamamatsu XR scanner | 0.775 |
| Human breast carcinoma | 0.773 | | | |
| Human melanoma | 0.824 | | | |
| Human neuroendocrine tumour | 0.748 | | | |

## Known limitations and failure modes

* **Undercounting under shift.** On unseen domains the model misses mitoses more than it over-calls them
  (unseen tumour type: precision 0.87, recall 0.70; unseen scanner: precision 0.85, recall 0.62;
  Bland–Altman bias −4.5 to −6.4 mitoses per region). Counts near a
  category cut-off should be reviewed; the grade-ambiguity flag targets exactly these.
* **Interval coverage below nominal on new scanners** (0.76 vs 0.90). Conformal guarantees assume the new
  site resembles the calibration data, and they weaken under domain shift.
* **Weakest domain:** a scanner unseen together with unseen tumour types (3D Histech: canine lung,
  lymphoma, soft-tissue sarcoma), F1 0.62.
* **Site adaptation** (self-training on a new site's unlabelled regions) can fail when the mitosis share
  of the site differs strongly from training. The prevalence-aware variant fixes this in our experiments,
  but it was designed after observing the failure and needs confirmation on a fresh site.
* **Label noise.** Ground truth is a consensus of 2–3 pathologists; individual pathologists disagree on a
  substantial share of figures, so F1 values near 0.8 approach the reference standard's own reliability.
* **No evaluation** on whole slides, non-MIDOG scanners (e.g. Leica GT450, Philips), other species, or
  prospective data.

## Ethical considerations

MIDOG++ is a public, de-identified, CC0-licensed dataset. Any clinical deployment would require
prospective validation, regulatory review, and monitoring for performance drift at each site.
