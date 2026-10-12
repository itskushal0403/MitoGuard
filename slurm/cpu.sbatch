#!/bin/bash
# Generic CPU job: sbatch slurm/cpu.sbatch <python script> [args...]
#SBATCH --partition=test
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m25csa016/mitoguard_work/logs/%x_%j.out
source /csehome/m25csa016/cv_projects/MitoGuard/slurm/env.sh
python "$@"
