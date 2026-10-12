#!/bin/bash
# Everything after Stage 1 training. Idempotent: re-run whenever new YOLO runs have finished.
#   sbatch --gres=gpu:1 -c 16 --mem=96G -p mtech -o $MITOGUARD_WORK/logs/downstream_%j.out slurm/downstream.sh
source /csehome/m25csa016/cv_projects/MitoGuard/slurm/env.sh
set -e
BACKBONES=${BACKBONES:-"phikon phikon2 resnet50 phikon_tta"}
python scripts/detect_all.py                                     # new runs only
python scripts/embed_crops.py --backbones $BACKBONES --rebuild-bank
RUNS=$(ls $MITOGUARD_WORK/cands | sed 's/\.pkl$//')
python scripts/run_stage2.py --runs $RUNS --backbones $BACKBONES --overwrite
python scripts/evaluate.py --backbones $BACKBONES
python scripts/make_report.py
OMP_NUM_THREADS=16 python scripts/feature_analysis.py --backbones resnet50 phikon phikon2 phikon_tta --umap
if [ -f $MITOGUARD_WORK/stage2/indomain_all_stain__phikon.pkl ]; then python scripts/export_bundle.py; fi
