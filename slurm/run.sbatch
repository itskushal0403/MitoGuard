#!/bin/bash
# Generic GPU job: sbatch slurm/run.sbatch <python script> [args...]
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/m25csa016/mitoguard_work/logs/%x_%j.out
source /csehome/m25csa016/cv_projects/MitoGuard/slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$@"
