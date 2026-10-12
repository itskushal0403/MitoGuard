#!/bin/bash
# Two concurrent YOLO training lanes on one GPU (each run is dataloader-bound, so the A30 is shared).
# Usage: sbatch slurm/train_lanes.sbatch "fold1:variant fold2:variant ..." "fold3:variant ..."
#   variant = stain | nostain
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=120G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=yolo_lanes
#SBATCH --output=/scratch/m25csa016/mitoguard_work/logs/%x_%j.out
source /csehome/m25csa016/cv_projects/MitoGuard/slurm/env.sh
EPOCHS=${EPOCHS:-30}
lane() {
  for spec in $1; do
    fold=${spec%%:*}; variant=${spec##*:}
    if [ -f $MITOGUARD_WORK/yolo/${fold}_${variant}/DONE ]; then echo "skip $spec"; continue; fi
    flag=""; [ "$variant" = "nostain" ] && flag="--no-stain-aug"
    echo "$(date) start $spec"
    python scripts/train_yolo.py --fold $fold $flag --epochs $EPOCHS --workers 13 \
      > $MITOGUARD_WORK/logs/train_${fold}_${variant}.log 2>&1
    rc=$?
    [ $rc -eq 0 ] && touch $MITOGUARD_WORK/yolo/${fold}_${variant}/DONE
    echo "$(date) done $spec exit=$rc"
  done
}
lane "$1" &
lane "$2" &
wait
