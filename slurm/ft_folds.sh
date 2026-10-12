#!/bin/bash
# Fine-tune a Stage 2 backbone on every fold (one GPU job per fold) and convert the scores.
#   bash slurm/ft_folds.sh <backbone> <partition> [extra finetune args...]
source /csehome/m25csa016/cv_projects/MitoGuard/slurm/env.sh
BB=$1; PART=${2:-test}; shift 2
L=$MITOGUARD_WORK/logs
TL=$([ "$PART" = "test" ] && echo 00:59:00 || echo 06:00:00)
FOLDS="loto_CCMCT loto_CL loto_CLC loto_CSTS loto_HBC loto_HMEL loto_HNET loso_3DH loso_CS2 loso_S360 loso_XR indomain_all"
for f in $FOLDS; do
  [ -f $MITOGUARD_WORK/ft/${f}__${BB}.npz ] && { echo "skip $f"; continue; }
  sbatch -p $PART --exclude=cn07 --gres=gpu:1 -c 8 --mem=64G -t $TL --job-name=ft_$BB -o $L/ft_${BB}_${f}_%j.out \
    --wrap "source slurm/env.sh; python scripts/finetune_stage2.py --fold $f --backbone $BB $* && python scripts/ft_to_stage2.py --folds $f --backbone $BB" | awk -v f=$f '{print f, $4}'
done
