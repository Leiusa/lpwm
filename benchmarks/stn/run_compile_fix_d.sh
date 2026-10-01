#!/bin/bash
#SBATCH --job-name=compile-fix-d
#SBATCH --partition=all
#SBATCH --constraint=rtx3090
#SBATCH --exclude=grogu-4-13,grogu-4-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/grogu/user/junhong3/lpwm-work/experiments/compile-probe/slurm/%x-%j.out
# Path D on the fixed tree (tree-fix), whole-model torch.compile with the custom Triton ops kept out of the traced graph
# (compile_probe.py --disable-custom-ops: torch.compiler.disable around triton_backend.stn_crop/stn_paste and composite_fused,
# applied from the probe script; no repository file changes). Outcome -> compile-probe/ledger/<jobid>.json.
set -u
export M1_EXPECTED_GPU="NVIDIA GeForce RTX 3090"
W=/grogu/user/junhong3/lpwm-work; P=$W/experiments/compile-probe; PY=$W/venvs/lpwm-stn/bin/python
NEW=$P/tree-fix; CFG=$W/data/static_bair128_vgg.json
export TORCH_HOME=$W/cache/torch
LIMIT_S=3600   # must match --time
JOB=${SLURM_JOB_ID:-nojob}; NODE=$(hostname -s); START=$(date +%s); STATUS=running; RC=-1; GPULINE=""
mkdir -p $P/ledger $P/runs $P/slurm
LED=$P/ledger/$JOB.json
write_ledger() {
  local END; END=$(date +%s)
  $PY - "$LED" "$JOB" "$NODE" "$STATUS" "$RC" "$START" "$END" "$GPULINE" "$LIMIT_S" <<'PYEOF'
import json, sys
led, job, node, status, rc, start, end, gpu, limit = sys.argv[1:10]
json.dump({"job": job, "kind": "compile_fix_d", "node": node, "gpu_line": gpu, "status": status, "exit_code": int(rc),
           "start_epoch": int(start), "end_epoch": int(end), "elapsed_s": int(end) - int(start),
           "elapsed_gpu_hours": (int(end) - int(start)) / 3600, "limit_s": int(limit)}, open(led, "w"), indent=1)
PYEOF
}
trap 'write_ledger' EXIT
trap 'STATUS=terminated_by_signal; exit 143' TERM
write_ledger
source $NEW/benchmarks/stn/m1_hwcheck.sh
HW=$(m1_check_gpu); HRC=$?
echo "$HW"; GPULINE=$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader,nounits 2>&1 | head -1)
if [ $HRC -ne 0 ]; then STATUS=hw_mismatch; RC=3; exit 3; fi
if ! (cd $NEW && sha256sum -c $P/tree-fix.sha256 --quiet >/dev/null 2>&1); then STATUS=snapshot_modified; RC=4; echo "tree-fix does not match its manifest"; exit 4; fi
OUT=$P/runs/fixd-$JOB; mkdir -p $OUT/eval/lpips $OUT/scripts
cp $W/weights/eval/lpips/vgg.pth $OUT/eval/lpips/vgg.pth     # LossLPIPS reads eval/lpips/vgg.pth relative to the cwd
cp $0 $P/tools/compile_probe.py $OUT/scripts/ 2>/dev/null
{ echo "job $JOB node $NODE start_epoch $START"; nvidia-smi -L;
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.max.sm --format=csv; } > $OUT/env.txt 2>&1
cd $OUT
export TORCH_LOGS="graph_breaks,recompiles"
export TORCHINDUCTOR_CACHE_DIR=$OUT/cache-fused_cl/inductor TRITON_CACHE_DIR=$OUT/cache-fused_cl/triton   # cold
$PY $P/tools/compile_probe.py --repo-root $NEW --config $CFG --path fused_cl --disable-custom-ops --out $OUT/probe-fused_cl.json > $OUT/probe-fused_cl.log 2>&1
RC=$?; echo "fused_cl probe exit=$RC"; grep '^\[' $OUT/probe-fused_cl.log | cut -c1-300
if [ $RC -eq 0 ]; then STATUS=ok; else STATUS=failed; fi
echo "OUT=$OUT"
exit $RC
