#!/bin/bash
#SBATCH --partition=all
#SBATCH --constraint=A6000
#SBATCH --exclude=grogu-4-13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:25:00
#SBATCH --no-requeue
#SBATCH --output=/grogu/user/junhong3/lpwm-work/experiments/fused-integration/m1/slurm/%x-%j.out
# Milestone-1 pair job for ONE seed:  sbatch --job-name=m1-s<SEED> [--exclude=grogu-4-13,<blocklist>] run_m1_seed.sh <SEED> <ORDER>
#   ORDER = reference,fused_cl  (A then D)   or   fused_cl,reference  (D then A)
# Gate 1: the allocated GPU must be exactly an NVIDIA RTX A6000 (and not grogu-4-13) -> otherwise exit 3, node is appended to the blocklist.
# Gate 2: the frozen code snapshot must match its sha256 manifest -> otherwise exit 4.
# Every outcome (also failures, time limits) is written to the persistent ledger m1/ledger/<jobid>.json.
set -u
export M1_EXPECTED_GPU="${M1_EXPECTED_GPU:-NVIDIA RTX A6000}"   # the study's one GPU model; the gate below enforces it
SEED=$1; ORDER=$2
W=/grogu/user/junhong3/lpwm-work; E=$W/experiments/fused-integration; M=$E/m1; PY=$W/venvs/lpwm-stn/bin/python
T=$E/tree-m1; CFG=$W/data/static_bair128_vgg.json
export TORCH_HOME=$W/cache/torch
LIMIT_S=${M1_LIMIT_S:-5100}   # must match the --time given at submission
JOB=${SLURM_JOB_ID:-nojob}; NODE=$(hostname -s); START=$(date +%s); STATUS=running; RC=-1; GPULINE=""
mkdir -p $M/ledger $M/runs $M/slurm
LED=$M/ledger/$JOB.json
write_ledger() {
  local END; END=$(date +%s)
  $PY - "$LED" "$JOB" "$SEED" "$ORDER" "$NODE" "$STATUS" "$RC" "$START" "$END" "$GPULINE" "$LIMIT_S" <<'PYEOF'
import json, sys
led, job, seed, order, node, status, rc, start, end, gpu, limit = sys.argv[1:12]
json.dump({"job": job, "kind": "m1_seed", "seed": int(seed), "order": order, "node": node, "gpu_line": gpu, "status": status, "exit_code": int(rc),
           "start_epoch": int(start), "end_epoch": int(end), "elapsed_s": int(end) - int(start), "elapsed_gpu_hours": (int(end) - int(start)) / 3600, "limit_s": int(limit)}, open(led, "w"), indent=1)
PYEOF
}
trap 'write_ledger' EXIT
trap 'STATUS=terminated_by_signal; exit 143' TERM
write_ledger
source $T/benchmarks/stn/m1_hwcheck.sh
HW=$(m1_check_gpu); HRC=$?
echo "$HW"; GPULINE=$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader,nounits 2>&1 | head -1)
if [ $HRC -ne 0 ]; then STATUS=hw_mismatch; RC=3; echo "$NODE" >> $M/blocklist.txt; exit 3; fi
if ! (cd $T && sha256sum -c $M/tree-m1.sha256 --quiet >/dev/null 2>&1); then STATUS=snapshot_modified; RC=4; echo "frozen snapshot does not match its manifest"; exit 4; fi
if [ -e $M/STOP ]; then STATUS=stopped_by_flag; RC=5; echo "STOP flag present"; exit 5; fi
( flock 9; $PY $M/jobs/m1_budget_gate.py $M/ledger $JOB $LIMIT_S 8.0 ) 9>$M/ledger.lock; GRC=$?
if [ $GRC -ne 0 ]; then STATUS=budget_gate; RC=5; echo "blocked by the aggregate GPU-hour cap"; exit 5; fi
OUT=$M/runs/seed${SEED}-${JOB}; mkdir -p $OUT/scripts
cp $0 $OUT/scripts/ 2>/dev/null; cp $CFG $OUT/scripts/; cp $M/tree-m1.sha256 $OUT/scripts/
{ echo "job $JOB seed $SEED order $ORDER node $NODE start_epoch $START"; nvidia-smi -L; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.max.sm,power.draw --format=csv; echo "manifest sha256 of manifest: $(sha256sum $M/tree-m1.sha256 | cut -c1-16)"; sha256sum $CFG $W/data/bair_256_ours_subset/manifest.json; } > $OUT/env.txt 2>&1
$PY $T/benchmarks/stn/m1_train_pair.py --repo-root $T --config $CFG --seed $SEED --epochs 8 --order $ORDER --lpips-head $W/weights/eval/lpips/vgg.pth --out $OUT --expect-val-images 1200 > $OUT/driver.log 2>&1
RC=$?
if [ $RC -eq 0 ]; then STATUS=ok; else STATUS=failed; fi
echo "driver exit=$RC"; tail -5 $OUT/driver.log | cut -c1-300; echo "OUT=$OUT"
exit $RC
