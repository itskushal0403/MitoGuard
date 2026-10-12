# Source from SLURM jobs: activates the project environment.
export PYTHONNOUSERSITE=1
export PATH=/scratch/m25csa016/envs/mitoguard/bin:$PATH
export MIDOG_ROOT=/scratch/m25csa016/midogpp
export MITOGUARD_WORK=/scratch/m25csa016/mitoguard_work
export HF_HOME=/scratch/m25csa016/hf_cache
export TORCH_HOME=/scratch/m25csa016/torch_cache
export YOLO_CONFIG_DIR=/scratch/m25csa016/mitoguard_work/.ultralytics
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PROJECT=/csehome/m25csa016/cv_projects/MitoGuard
cd $PROJECT
