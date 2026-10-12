#!/bin/bash
# Final downstream pass over every YOLO run that has candidates (run on the login node; submits a job chain).
#   bank -> 3 parallel embedding jobs -> 3 parallel Stage 2 jobs -> evaluation + report (CPU)
set -e
source /csehome/m25csa016/cv_projects/MitoGuard/slurm/env.sh
W=$MITOGUARD_WORK; L=$W/logs
python scripts/embed_crops.py --bank-only --rebuild-bank
ALL=$(ls $W/cands | sed 's/\.pkl$//' | tr '\n' ' ')
STAIN=$(ls $W/cands | grep _stain | sed 's/\.pkl$//' | tr '\n' ' ')
GPU="-p ${PART:-test} --exclude=cn07 --gres=gpu:1 -c 12 --mem=48G -t 00:55:00"
emb() { sbatch --parsable $GPU --job-name=emb -o $L/emb_%j.out --wrap "source slurm/env.sh; python scripts/embed_crops.py --backbones $1"; }
E1=$(emb "phikon2-z112-ctr"); E2=$(emb "phikon-z112-ctr phikon"); E3=$(emb "resnet50 resnet50-z112")
s2() { sbatch --parsable $GPU -d afterok:$1 --job-name=s2 -o $L/s2_%j.out --wrap "source slurm/env.sh; $2"; }
S1=$(s2 $E1 "python scripts/run_stage2.py --backbones phikon2-z112-ctr --pool m05i10 --runs $ALL --overwrite && python scripts/run_stage2.py --backbones phikon2-z112-ctr --pool m05 --runs $STAIN --overwrite")
S2=$(s2 $E2 "python scripts/run_stage2.py --backbones phikon-z112-ctr --pool m05i10 --runs $ALL --overwrite && python scripts/run_stage2.py --backbones phikon-z112-ctr phikon --pool m05 --runs $STAIN --overwrite")
S3=$(s2 $E3 "python scripts/run_stage2.py --backbones resnet50 resnet50-z112 --pool m05 --runs $STAIN --overwrite && python scripts/run_stage2.py --backbones resnet50 --pool m05i10 --runs $STAIN --overwrite")
EV=$(sbatch --parsable -p test -c 24 --mem=96G -t 00:59:00 -d afterok:$S1:$S2:$S3 --job-name=evaluate -o $L/evaluate_%j.out \
  --wrap "source slurm/env.sh; python scripts/evaluate.py --workers 12 --pools m05 m05i10 --backbones phikon2-z112-ctr phikon-z112-ctr phikon resnet50 resnet50-z112 && python scripts/make_report.py && python scripts/export_bundle.py && OMP_NUM_THREADS=24 python scripts/feature_analysis.py --backbones resnet50 resnet50-z112 phikon phikon-z112-ctr phikon2-z112-ctr --umap")
echo "embed $E1 $E2 $E3 | stage2 $S1 $S2 $S3 | evaluate $EV"
