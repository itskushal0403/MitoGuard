# MitoGuard v2 model selection

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
