#!/bin/bash
# MitoGuard v2 final pass (CPU job): convert fine-tuned scores, build ensembles, evaluate all systems,
# nested model selection, bootstrap CIs, clinical metrics, report figures, demo bundle.
#   sbatch -p test -c 24 --mem=96G -t 00:59:00 slurm/final_v2.sh
source /csehome/m25csa016/cv_projects/MitoGuard/slurm/env.sh
set -e
W=$MITOGUARD_WORK
FOLDS="loto_CCMCT loto_CL loto_CLC loto_CSTS loto_HBC loto_HMEL loto_HNET loso_3DH loso_CS2 loso_S360 loso_XR indomain_all"
python scripts/make_ensemble.py --members phikon2 phikon lunit --name ens
python scripts/make_ensemble.py --members phikon2 phikon lunit kaikob --name ens4
python scripts/make_ensemble.py --members phikon2+st phikon+st lunit+st --name ens+st
python scripts/make_ensemble.py --members phikon2+stpa phikon+stpa lunit+stpa --name ens+stpa
for bb in phikon2 phikon lunit kaikob ens ens4 phikon2+st phikon+st lunit+st ens+st phikon2+stpa phikon+stpa lunit+stpa ens+stpa; do
  avail=$(for f in $FOLDS; do [ -f $W/ft/${f}__${bb}.npz ] && echo $f; done | tr '\n' ' ')
  [ -n "$avail" ] && python scripts/ft_to_stage2.py --folds $avail --backbone $bb > /dev/null
done
FT="ft-phikon2 ft-phikon ft-lunit ft-kaikob ft-ens ft-ens4 ft-phikon2+st ft-phikon+st ft-lunit+st ft-ens+st ft-phikon2+stpa ft-phikon+stpa ft-lunit+stpa ft-ens+stpa"
V1="phikon2-z112-ctr"   # v1 headline system, for comparison (v1 ablations stay in results/)
export MITOGUARD_RESULTS=$PROJECT/results/v2
python scripts/evaluate.py --workers 15 --pools m05i10 a05 --backbones $V1 $FT --out $MITOGUARD_RESULTS
python scripts/select_v2.py
python scripts/bootstrap_ci.py --results $MITOGUARD_RESULTS --systems MitoGuard-v2 MitoGuard-v2-SA MitoGuard-v2-SApa "yolo+phikon2-z112-ctr@m05i10+stack"
python scripts/clinical_eval.py --systems yolo "yolo+phikon2-z112-ctr@m05i10+stack" MitoGuard-v2 MitoGuard-v2-SA MitoGuard-v2-SApa
python scripts/make_report_v2.py
python scripts/calibration.py
