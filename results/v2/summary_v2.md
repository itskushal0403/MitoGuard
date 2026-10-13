# MitoGuard v2 results

## Held-out F1 per domain

| protocol   | held-out   |   YOLO11 only |   MitoGuard v1 (frozen FM) |   MitoGuard v2 (fine-tuned FM) |   v2 + site adaptation (fixed budget) |   v2 + site adaptation (prevalence-aware, post-hoc) |
|:-----------|:-----------|--------------:|---------------------------:|-------------------------------:|--------------------------------------:|----------------------------------------------------:|
| loto       | CCMCT      |         0.814 |                      0.826 |                          0.812 |                                 0.823 |                                               0.839 |
| loto       | CL         |         0.745 |                      0.728 |                          0.758 |                                 0.739 |                                               0.752 |
| loto       | CLC        |         0.621 |                      0.635 |                          0.658 |                                 0.695 |                                               0.69  |
| loto       | CSTS       |         0.797 |                      0.817 |                          0.822 |                                 0.796 |                                               0.804 |
| loto       | HBC        |         0.747 |                      0.76  |                          0.773 |                                 0.786 |                                               0.792 |
| loto       | HMEL       |         0.799 |                      0.807 |                          0.824 |                                 0.821 |                                               0.823 |
| loto       | HNET       |         0.706 |                      0.737 |                          0.748 |                                 0.71  |                                               0.737 |
| loso       | 3DH        |         0.451 |                      0.424 |                          0.62  |                                 0.58  |                                               0.731 |
| loso       | CS2        |         0.326 |                      0.441 |                          0.812 |                                 0.714 |                                               0.831 |
| loso       | S360       |         0.721 |                      0.747 |                          0.771 |                                 0.786 |                                               0.797 |
| loso       | XR         |         0.741 |                      0.778 |                          0.775 |                                 0.583 |                                               0.742 |

## MitoGuard v2 model selection

Model selection for MitoGuard v2 without touching a fold's own held-out domain.

Rule A (primary): nested leave-one-domain-out selection (DomainBed-style, Gulrajani & Lopez-Paz, ICLR 2021).
  For fold k of a protocol, pick the configuration with the best mean held-out F1 on the *other* folds of
  that protocol, then report its F1 on fold k. Fold k's held-out labels never influence its own choice.
Rule B (reported for transparency): in-domain validation selection, i.e. the configuration with the best
  mean validation F1 (training-domain cases). Known to favour configurations that lean on in-domain cues.

Two families are selected separately: "no adaptation" (pure domain generalisation) and "site-adapted"
(self-training on the unlabelled target site, backbone names ending in '+st').

## Summary

| protocol   | family                                           | rule                                                    |   mean F1 |   YOLO11 |   gain | folds beating YOLO   |
|:-----------|:-------------------------------------------------|:--------------------------------------------------------|----------:|---------:|-------:|:---------------------|
| loto       | no adaptation                                    | nested LODO (primary)                                   |     0.771 |    0.747 |  0.024 | 6/7                  |
| loto       | no adaptation                                    | in-domain val (yolo+ft-ens@m05i10+fused_union)          |     0.778 |    0.747 |  0.031 | 6/7                  |
| loto       | site-adapted, fixed budget (pre-specified)       | nested LODO (primary)                                   |     0.767 |    0.747 |  0.02  | 5/7                  |
| loto       | site-adapted, fixed budget (pre-specified)       | in-domain val (yolo+ft-ens+st@m05i10+fused_union)       |     0.774 |    0.747 |  0.027 | 6/7                  |
| loto       | site-adapted, prevalence-aware budget (post-hoc) | nested LODO (primary)                                   |     0.777 |    0.747 |  0.03  | 7/7                  |
| loto       | site-adapted, prevalence-aware budget (post-hoc) | in-domain val (yolo+ft-ens+stpa@m05i10+fused_union)     |     0.783 |    0.747 |  0.036 | 7/7                  |
| loso       | no adaptation                                    | nested LODO (primary)                                   |     0.744 |    0.56  |  0.185 | 4/4                  |
| loso       | no adaptation                                    | in-domain val (yolo+ft-ens4@m05i10+fused_union)         |     0.691 |    0.56  |  0.132 | 4/4                  |
| loso       | site-adapted, fixed budget (pre-specified)       | nested LODO (primary)                                   |     0.666 |    0.56  |  0.106 | 3/4                  |
| loso       | site-adapted, fixed budget (pre-specified)       | in-domain val (yolo+ft-lunit+st@m05i10+fused_union)     |     0.698 |    0.56  |  0.139 | 4/4                  |
| loso       | site-adapted, prevalence-aware budget (post-hoc) | nested LODO (primary)                                   |     0.775 |    0.56  |  0.216 | 4/4                  |
| loso       | site-adapted, prevalence-aware budget (post-hoc) | in-domain val (yolo+ft-phikon2+stpa@m05i10+fused_union) |     0.712 |    0.56  |  0.152 | 4/4                  |

## LOTO / no adaptation: nested selection per fold

| fold   | selected                       |    f1 |   yolo |   gain |
|:-------|:-------------------------------|------:|-------:|-------:|
| CCMCT  | yolo+ft-ens@m05i10+fused_union | 0.812 |  0.814 | -0.002 |
| CL     | yolo+ft-ens@m05i10+stack_union | 0.758 |  0.745 |  0.013 |
| CLC    | yolo+ft-lunit@a05+fused_union  | 0.658 |  0.621 |  0.038 |
| CSTS   | yolo+ft-ens@m05i10+stack_union | 0.822 |  0.797 |  0.025 |
| HBC    | yolo+ft-ens@m05i10+stack_union | 0.773 |  0.747 |  0.026 |
| HMEL   | yolo+ft-ens@m05i10+stack_union | 0.824 |  0.799 |  0.024 |
| HNET   | yolo+ft-ens@a05+stack_union    | 0.748 |  0.706 |  0.041 |

## LOTO / site-adapted, fixed budget (pre-specified): nested selection per fold

| fold   | selected                              |    f1 |   yolo |   gain |
|:-------|:--------------------------------------|------:|-------:|-------:|
| CCMCT  | yolo+ft-ens+st@a05+stack_union        | 0.823 |  0.814 |  0.01  |
| CL     | yolo+ft-phikon2+st@a05+stack_union    | 0.739 |  0.745 | -0.006 |
| CLC    | yolo+ft-phikon2+st@m05i10+fused_union | 0.695 |  0.621 |  0.074 |
| CSTS   | yolo+ft-ens+st@a05+stack_union        | 0.796 |  0.797 | -0.001 |
| HBC    | yolo+ft-lunit+st@a05+fused_union      | 0.786 |  0.747 |  0.038 |
| HMEL   | yolo+ft-lunit+st@a05+fused_union      | 0.821 |  0.799 |  0.021 |
| HNET   | yolo+ft-phikon2+st@m05i10+fused_union | 0.71  |  0.706 |  0.003 |

## LOTO / site-adapted, prevalence-aware budget (post-hoc): nested selection per fold

| fold   | selected                             |    f1 |   yolo |   gain |
|:-------|:-------------------------------------|------:|-------:|-------:|
| CCMCT  | yolo+ft-ens+stpa@a05+fused_union     | 0.839 |  0.814 |  0.025 |
| CL     | yolo+ft-phikon2+stpa@a05+fused_union | 0.752 |  0.745 |  0.007 |
| CLC    | yolo+ft-phikon2+stpa@a05+fused_union | 0.69  |  0.621 |  0.07  |
| CSTS   | yolo+ft-ens+stpa@a05+stack_union     | 0.804 |  0.797 |  0.007 |
| HBC    | yolo+ft-ens+stpa@a05+fused_union     | 0.792 |  0.747 |  0.044 |
| HMEL   | yolo+ft-ens+stpa@a05+fused_union     | 0.823 |  0.799 |  0.024 |
| HNET   | yolo+ft-ens+stpa@a05+fused_union     | 0.737 |  0.706 |  0.031 |

## LOSO / no adaptation: nested selection per fold

| fold   | selected                 |    f1 |   yolo |   gain |
|:-------|:-------------------------|------:|-------:|-------:|
| 3DH    | yolo+ft-ens4@a05_union   | 0.62  |  0.451 |  0.169 |
| CS2    | yolo+ft-ens@m05i10_union | 0.812 |  0.326 |  0.486 |
| S360   | yolo+ft-ens@m05i10_union | 0.771 |  0.721 |  0.05  |
| XR     | yolo+ft-ens@m05i10_union | 0.775 |  0.741 |  0.034 |

## LOSO / site-adapted, fixed budget (pre-specified): nested selection per fold

| fold   | selected                          |    f1 |   yolo |   gain |
|:-------|:----------------------------------|------:|-------:|-------:|
| 3DH    | yolo+ft-phikon+st@a05+fused_union | 0.58  |  0.451 |  0.129 |
| CS2    | yolo+ft-lunit+st@a05+stack_union  | 0.714 |  0.326 |  0.388 |
| S360   | yolo+ft-lunit+st@m05i10_union     | 0.786 |  0.721 |  0.065 |
| XR     | yolo+ft-lunit+st@m05i10_union     | 0.583 |  0.741 | -0.158 |

## LOSO / site-adapted, prevalence-aware budget (post-hoc): nested selection per fold

| fold   | selected                      |    f1 |   yolo |   gain |
|:-------|:------------------------------|------:|-------:|-------:|
| 3DH    | yolo+ft-ens+stpa@m05i10_union | 0.731 |  0.451 |  0.28  |
| CS2    | yolo+ft-ens+stpa@m05i10_union | 0.831 |  0.326 |  0.505 |
| S360   | yolo+ft-ens+stpa@m05i10_union | 0.797 |  0.721 |  0.076 |
| XR     | yolo+ft-ens+stpa@m05i10_union | 0.742 |  0.741 |  0.001 |


## Paired bootstrap 95% confidence intervals (held-out regions)

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

## Clinical evaluation of region-level mitotic counts

Held-out regions only. Conformal intervals are calibrated on each fold's validation regions
(training domains); categories use the cut-offs <=7 / 8-14 / >=15 per ~2 mm^2 region.

## LOTO

|                                         | yolo               | yolo+phikon2-z112-ctr@m05i10+stack   | MitoGuard-v2       | MitoGuard-v2-SA     | MitoGuard-v2-SApa   |
|:----------------------------------------|:-------------------|:-------------------------------------|:-------------------|:--------------------|:--------------------|
| protocol                                | loto               | loto                                 | loto               | loto                | loto                |
| regions                                 | 503                | 503                                  | 503                | 503                 | 503                 |
| count MAE                               | 6.250497017892644  | 6.508946322067595                    | 5.90258449304175   | 6.294234592445328   | 5.9840954274353875  |
| CCC                                     | 0.913365830558174  | 0.8994453645074926                   | 0.9209630649070433 | 0.9179427401162945  | 0.9242074056272027  |
| Spearman                                | 0.9300158988783128 | 0.9240381051958307                   | 0.9264638336149409 | 0.9281262326183259  | 0.9363747263595584  |
| Bland-Altman bias                       | -4.667992047713717 | -5.236580516898608                   | -4.495029821073559 | -3.1053677932405566 | -3.761431411530815  |
| LoA low                                 | -31.04209880061787 | -33.53527067485818                   | -29.87422832364659 | -29.842976699850077 | -29.088761819313994 |
| LoA high                                | 21.706114705190437 | 23.062109641060964                   | 20.884168681499474 | 23.632241113368963  | 21.565898996252365  |
| category accuracy                       | 0.8190854870775348 | 0.8250497017892644                   | 0.831013916500994  | 0.8330019880715706  | 0.8349900596421471  |
| category kappa (quadratic)              | 0.8611770967837323 | 0.8575263851430401                   | 0.8577869890276313 | 0.8658706408699002  | 0.875285684476685   |
| conformal 90% coverage                  | 0.8091451292246521 | 0.7852882703777336                   | 0.805168986083499  | 0.8011928429423459  | 0.805168986083499   |
| naive +-1.645 sigma coverage            | 0.5387673956262425 | 0.5506958250497018                   | 0.5705765407554672 | 0.5566600397614314  | 0.5646123260437376  |
| median interval width                   | 12.611483693775215 | 11.884745969668842                   | 12.119675276331446 | 13.217798814682482  | 12.32800083667044   |
| flagged (interval straddles a cut-off)  | 0.5964214711729622 | 0.5646123260437376                   | 0.5447316103379721 | 0.5765407554671969  | 0.5467196819085487  |
| category accuracy | flagged             | 0.7                | 0.6936619718309859                   | 0.6934306569343066 | 0.7137931034482758  | 0.7054545454545454  |
| category accuracy | not flagged         | 0.9950738916256158 | 0.9954337899543378                   | 0.9956331877729258 | 0.9953051643192489  | 0.9912280701754386  |
| share of category errors caught by flag | 0.989010989010989  | 0.9886363636363636                   | 0.9882352941176471 | 0.9880952380952381  | 0.9759036144578314  |

Per fold:

| system                             | fold   |   coverage |     q |   kappa |   CCC |
|:-----------------------------------|:-------|-----------:|------:|--------:|------:|
| yolo                               | CCMCT  |      0.7   | 3.467 |   0.913 | 0.944 |
| yolo                               | CL     |      0.473 | 2.668 |   0.715 | 0.674 |
| yolo                               | CLC    |      0.591 | 3.038 |   0.584 | 0.718 |
| yolo                               | CSTS   |      0.96  | 3.181 |   0.811 | 0.972 |
| yolo                               | HBC    |      0.913 | 3.942 |   0.891 | 0.911 |
| yolo                               | HMEL   |      0.755 | 3.56  |   0.877 | 0.968 |
| yolo                               | HNET   |      0.909 | 3.542 |   0.767 | 0.979 |
| yolo+phikon2-z112-ctr@m05i10+stack | CCMCT  |      0.7   | 3.396 |   0.926 | 0.961 |
| yolo+phikon2-z112-ctr@m05i10+stack | CL     |      0.364 | 2.516 |   0.5   | 0.575 |
| yolo+phikon2-z112-ctr@m05i10+stack | CLC    |      0.591 | 2.838 |   0.55  | 0.73  |
| yolo+phikon2-z112-ctr@m05i10+stack | CSTS   |      0.97  | 3.36  |   0.862 | 0.97  |
| yolo+phikon2-z112-ctr@m05i10+stack | HBC    |      0.867 | 2.879 |   0.87  | 0.923 |
| yolo+phikon2-z112-ctr@m05i10+stack | HMEL   |      0.714 | 3.314 |   0.891 | 0.97  |
| yolo+phikon2-z112-ctr@m05i10+stack | HNET   |      0.945 | 3.664 |   0.822 | 0.981 |
| MitoGuard-v2                       | CCMCT  |      0.64  | 3.004 |   0.881 | 0.94  |
| MitoGuard-v2                       | CL     |      0.418 | 2.239 |   0.441 | 0.708 |
| MitoGuard-v2                       | CLC    |      0.591 | 3.029 |   0.588 | 0.741 |
| MitoGuard-v2                       | CSTS   |      0.95  | 3.372 |   0.841 | 0.972 |
| MitoGuard-v2                       | HBC    |      0.92  | 3.493 |   0.916 | 0.943 |
| MitoGuard-v2                       | HMEL   |      0.878 | 3.494 |   0.902 | 0.988 |
| MitoGuard-v2                       | HNET   |      0.873 | 3.155 |   0.749 | 0.982 |
| MitoGuard-v2-SA                    | CCMCT  |      0.6   | 2.798 |   0.902 | 0.963 |
| MitoGuard-v2-SA                    | CL     |      0.327 | 2.413 |   0.528 | 0.662 |
| MitoGuard-v2-SA                    | CLC    |      0.795 | 2.96  |   0.621 | 0.853 |
| MitoGuard-v2-SA                    | CSTS   |      0.86  | 3.349 |   0.79  | 0.943 |
| MitoGuard-v2-SA                    | HBC    |      0.98  | 3.493 |   0.944 | 0.964 |
| MitoGuard-v2-SA                    | HMEL   |      0.796 | 3.038 |   0.902 | 0.983 |
| MitoGuard-v2-SA                    | HNET   |      0.873 | 2.859 |   0.788 | 0.96  |
| MitoGuard-v2-SApa                  | CCMCT  |      0.72  | 2.772 |   0.926 | 0.968 |
| MitoGuard-v2-SApa                  | CL     |      0.327 | 2.032 |   0.637 | 0.679 |
| MitoGuard-v2-SApa                  | CLC    |      0.795 | 3.049 |   0.641 | 0.828 |
| MitoGuard-v2-SApa                  | CSTS   |      0.89  | 3.378 |   0.789 | 0.952 |
| MitoGuard-v2-SApa                  | HBC    |      0.92  | 3.164 |   0.924 | 0.961 |
| MitoGuard-v2-SApa                  | HMEL   |      0.816 | 3.18  |   0.902 | 0.979 |
| MitoGuard-v2-SApa                  | HNET   |      0.891 | 2.849 |   0.863 | 0.975 |

## LOSO

|                                         | yolo                | yolo+phikon2-z112-ctr@m05i10+stack   | MitoGuard-v2        | MitoGuard-v2-SA     | MitoGuard-v2-SApa   |
|:----------------------------------------|:--------------------|:-------------------------------------|:--------------------|:--------------------|:--------------------|
| protocol                                | loso                | loso                                 | loso                | loso                | loso                |
| regions                                 | 503                 | 503                                  | 503                 | 503                 | 503                 |
| count MAE                               | 14.733598409542743  | 14.129224652087474                   | 8.326043737574553   | 14.159045725646124  | 6.312127236580517   |
| CCC                                     | 0.45191915152222223 | 0.5059189079169953                   | 0.8490047483218707  | 0.6881008965184782  | 0.9331215913893743  |
| Spearman                                | 0.7055285489920178  | 0.7046915478218913                   | 0.8746927476545968  | 0.6776415496197495  | 0.9039494029901521  |
| Bland-Altman bias                       | -14.411530815109344 | -13.42544731610338                   | -6.389662027833002  | -3.817097415506958  | -2.371769383697813  |
| LoA low                                 | -70.71413310334108  | -68.18068004153173                   | -40.651551530790115 | -56.227810001207736 | -27.47467548333548  |
| LoA high                                | 41.8910714731224    | 41.32978540932498                    | 27.87222747512411   | 48.59361517019382   | 22.731136715939854  |
| category accuracy                       | 0.6620278330019881  | 0.6620278330019881                   | 0.7992047713717694  | 0.6361829025844931  | 0.8210735586481114  |
| category kappa (quadratic)              | 0.5520220832451741  | 0.5623912503107134                   | 0.7995435564477602  | 0.5763316007529278  | 0.8536674515693495  |
| conformal 90% coverage                  | 0.5924453280318092  | 0.6341948310139165                   | 0.7594433399602386  | 0.6023856858846919  | 0.8071570576540755  |
| naive +-1.645 sigma coverage            | 0.33399602385685884 | 0.36182902584493043                  | 0.5526838966202783  | 0.34393638170974156 | 0.5606361829025845  |
| median interval width                   | 9.067618768030961   | 10.12879549861563                    | 11.707543789217397  | 12.910877885924037  | 12.614941252562936  |
| flagged (interval straddles a cut-off)  | 0.5347912524850894  | 0.5785288270377733                   | 0.5864811133200796  | 0.5745526838966203  | 0.6023856858846919  |
| category accuracy | flagged             | 0.5427509293680297  | 0.5326460481099656                   | 0.6983050847457627  | 0.4844290657439446  | 0.7326732673267327  |
| category accuracy | not flagged         | 0.7991452991452992  | 0.839622641509434                    | 0.9423076923076923  | 0.8411214953271028  | 0.955               |
| share of category errors caught by flag | 0.7235294117647059  | 0.8                                  | 0.8811881188118812  | 0.8142076502732241  | 0.9                 |

Per fold:

| system                             | fold   |   coverage |     q |   kappa |   CCC |
|:-----------------------------------|:-------|-----------:|------:|--------:|------:|
| yolo                               | 3DH    |      0.402 | 2.896 |   0.383 | 0.33  |
| yolo                               | CS2    |      0.37  | 3.712 |   0.374 | 0.288 |
| yolo                               | S360   |      0.94  | 3.993 |   0.927 | 0.913 |
| yolo                               | XR     |      0.87  | 3.4   |   0.87  | 0.954 |
| yolo+phikon2-z112-ctr@m05i10+stack | 3DH    |      0.407 | 2.883 |   0.333 | 0.281 |
| yolo+phikon2-z112-ctr@m05i10+stack | CS2    |      0.45  | 3.515 |   0.523 | 0.457 |
| yolo+phikon2-z112-ctr@m05i10+stack | S360   |      0.96  | 4.223 |   0.985 | 0.953 |
| yolo+phikon2-z112-ctr@m05i10+stack | XR     |      0.942 | 3.539 |   0.888 | 0.981 |
| MitoGuard-v2                       | 3DH    |      0.508 | 2.822 |   0.642 | 0.634 |
| MitoGuard-v2                       | CS2    |      0.87  | 3.6   |   0.939 | 0.966 |
| MitoGuard-v2                       | S360   |      1     | 3.409 |   0.942 | 0.986 |
| MitoGuard-v2                       | XR     |      0.935 | 3.524 |   0.857 | 0.978 |
| MitoGuard-v2-SA                    | 3DH    |      0.462 | 2.46  |   0.566 | 0.507 |
| MitoGuard-v2-SA                    | CS2    |      0.68  | 3.331 |   0.816 | 0.841 |
| MitoGuard-v2-SA                    | S360   |      0.98  | 3.414 |   0.851 | 0.954 |
| MitoGuard-v2-SA                    | XR     |      0.61  | 3.59  |   0.458 | 0.743 |
| MitoGuard-v2-SApa                  | 3DH    |      0.653 | 2.883 |   0.818 | 0.855 |
| MitoGuard-v2-SApa                  | CS2    |      0.93  | 3.5   |   0.933 | 0.987 |
| MitoGuard-v2-SApa                  | S360   |      1     | 3.698 |   0.956 | 0.99  |
| MitoGuard-v2-SApa                  | XR     |      0.864 | 3.298 |   0.794 | 0.957 |

