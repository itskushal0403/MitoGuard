# Candidate-level calibration on held-out tumour types (LOTO)

| system                          |   held-out candidates |    ECE |   Brier |
|:--------------------------------|----------------------:|-------:|--------:|
| YOLO11 p_mitosis (raw)          |                 49279 | 0.0599 |  0.0789 |
| v1 frozen Phikon-v2 filter      |                 49279 | 0.0337 |  0.0837 |
| v2 fine-tuned Phikon-v2         |                 91901 | 0.0171 |  0.0393 |
| v2 fine-tuned + site adaptation |                 91901 | 0.0146 |  0.0462 |