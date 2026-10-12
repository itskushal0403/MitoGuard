#!/bin/bash
# Pull-based YOLO training queue shared by any number of SLURM jobs on any partition.
#   queue file : $MITOGUARD_WORK/queue/runs.txt   (one "fold:variant" per line, in priority order)
#   claim      : mkdir $MITOGUARD_WORK/queue/claims/<run>  (atomic) + owner job id, under a flock
#   stale claim: owner job no longer in squeue and run not finished -> reclaimed and resumed from last.pt
#   finished   : PENDING marker removed from the run directory (DONE already exists for queued runs)
# Usage: queue_worker.sh <max_runs>   (1 for 1-hour test-partition jobs; large for mtech jobs)
source /csehome/m25csa016/cv_projects/MitoGuard/slurm/env.sh
Q=$MITOGUARD_WORK/queue; MAX=${1:-1000}; EPOCHS=${EPOCHS:-30}; WORKERS=${WORKERS:-12}
ME=${SLURM_JOB_ID:-$$}_${LANE:-0}
claim() {
  (
    flock 9
    while read -r spec; do
      run="${spec%%:*}_${spec##*:}"; d=$MITOGUARD_WORK/yolo/$run; c=$Q/claims/$run
      [ -f $d/PENDING ] || continue                       # finished
      if [ -d $c ]; then
        owner=$(cat $c/owner 2>/dev/null); job=${owner%%_*}
        squeue -h -j "$job" 2>/dev/null | grep -q . && continue   # owner alive
        rm -rf $c                                          # stale (timed out / crashed)
      fi
      mkdir $c && echo $ME > $c/owner && echo "$spec" && break
    done < $Q/runs.txt
  ) 9>$Q/lock
}
n=0
while [ $n -lt $MAX ]; do
  spec=$(claim); [ -z "$spec" ] && { echo "$(date) queue empty"; break; }
  fold=${spec%%:*}; variant=${spec##*:}; run=${fold}_${variant}
  flag=""; [ "$variant" = "nostain" ] && flag="--no-stain-aug"
  echo "$(date) [$ME] start $spec on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  python scripts/train_yolo.py --fold $fold $flag --epochs $EPOCHS --workers $WORKERS --resume \
    >> $MITOGUARD_WORK/logs/train_${run}.log 2>&1
  rc=$?
  [ $rc -eq 0 ] && rm -f $MITOGUARD_WORK/yolo/$run/PENDING && rm -rf $Q/claims/$run
  echo "$(date) [$ME] done $spec exit=$rc"
  n=$((n + 1))
done
