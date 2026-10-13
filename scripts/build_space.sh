#!/bin/bash
# Assemble the Hugging Face Space folder (CPU demo) from the repo + exported lite bundle.
#   bash scripts/build_space.sh /path/to/space_dir   (checkpoints/ must already contain the lite bundle + YOLO weights)
set -e
S=${1:-/scratch/m25csa016/mitoguard_space}
R=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $S/mitoguard $S/examples $S/figures
cp $R/app/space/app.py $S/app.py
cp $R/app/space/README.md $S/README.md
cp $R/app/space/requirements.txt $S/requirements.txt
cp $R/mitoguard/*.py $S/mitoguard/
cp $R/app/examples/*.jpg $S/examples/
cp $R/results/v2/figures/v2_f1.png $R/results/v2/figures/calibration.png $R/results/figures/umap_features.png $R/docs/figures/pipeline.png $S/figures/
ls -R $S | head -40
