#!/bin/bash
#SBATCH --job-name=compile-probe
#SBATCH --partition=all
#SBATCH --constraint=rtx3090
#SBATCH --exclude=grogu-4-13,grogu-4-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/grogu/user/junhong3/lpwm-work/experiments/compile-probe/slurm/%x-%j.out
# torch.compile probe: A (reference) then D (fused_cl), one process each, same verified RTX 3090. Per path: eager baseline,
# dynamo explain, compiled training steps, eager drift check, compiled-vs-eager inference numerics (compile_probe.py).
# Runs the frozen milestone-1 tree (production files identical to the pushed code). Nothing in the repository changes.
# Every outcome is written to the ledger compile-probe/ledger/<jobid>.json.
set -u
export M1_EXPECTED_GPU="NVIDIA GeForce RTX 3090"
W=/grogu/user/junhong3/lpwm-work; P=$W/experiments/compile-probe; PY=$W/venvs/lpwm-stn/bin/python
T=$W/experiments/fused-integration/tree-m1; CFG=$W/data/static_bair128_vgg.json
export TORCH_HOME=$W/cache/torch
export TORCH_LOGS="graph_breaks,recompiles"
LIMIT_S=5400   # must match --time
JOB=${SLURM_JOB_ID:-nojob}; NODE=$(hostname -s); START=$(date +%s); STATUS=running; RC=-1; GPULINE=""
mkdir -p $P/ledger $P/runs $P/slurm
LED=$P/ledger/$JOB.json
write_ledger() {
  local END; END=$(date +%s)
  $PY - "$LED" "$JOB" "$NODE" "$STATUS" "$RC" "$START" "$END" "$GPULINE" "$LIMIT_S" <<'PYEOF'
import json, sys
led, job, node, status, rc, start, end, gpu, limit = sys.argv[1:10]
json.dump({"job": job, "kind": "compile_probe", "node": node, "gpu_line": gpu, "status": status, "exit_code": int(rc),
           "start_epoch": int(start), "end_epoch": int(end), "elapsed_s": int(end) - int(start),
           "elapsed_gpu_hours": (int(end) - int(start)) / 3600, "limit_s": int(limit)}, open(led, "w"), indent=1)
PYEOF
}
trap 'write_ledger' EXIT
trap 'STATUS=terminated_by_signal; exit 143' TERM
write_ledger
source $T/benchmarks/stn/m1_hwcheck.sh
HW=$(m1_check_gpu); HRC=$?
echo "$HW"; GPULINE=$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader,nounits 2>&1 | head -1)
if [ $HRC -ne 0 ]; then STATUS=hw_mismatch; RC=3; exit 3; fi
OUT=$P/runs/probe-$JOB; mkdir -p $OUT/eval/lpips $OUT/scripts
cp $W/weights/eval/lpips/vgg.pth $OUT/eval/lpips/vgg.pth     # LossLPIPS reads eval/lpips/vgg.pth relative to the cwd
cp $0 $P/tools/compile_probe.py $OUT/scripts/ 2>/dev/null
{ echo "job $JOB node $NODE start_epoch $START"; nvidia-smi -L;
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.max.sm --format=csv; } > $OUT/env.txt 2>&1
cd $OUT
RC=0
for PATHNAME in reference fused_cl; do
  export TORCHINDUCTOR_CACHE_DIR=$OUT/cache-$PATHNAME/inductor TRITON_CACHE_DIR=$OUT/cache-$PATHNAME/triton   # cold, per path
  $PY $P/tools/compile_probe.py --repo-root $T --config $CFG --path $PATHNAME --out $OUT/probe-$PATHNAME.json > $OUT/probe-$PATHNAME.log 2>&1
  R=$?; echo "$PATHNAME exit=$R"; grep '^\[' $OUT/probe-$PATHNAME.log | cut -c1-300
  if [ $R -ne 0 ]; then RC=$R; fi
done
if [ $RC -eq 0 ]; then STATUS=ok; else STATUS=failed; fi
echo "OUT=$OUT"
exit $RC
