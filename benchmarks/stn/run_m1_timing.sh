#!/bin/bash
#SBATCH --job-name=m1-timing
#SBATCH --partition=all
#SBATCH --constraint=rtx3090
#SBATCH --exclude=grogu-4-13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/grogu/user/junhong3/lpwm-work/experiments/fused-integration/m1/slurm/%x-%j.out
# Sizing run (not part of the study's results): real-flow training step time of A (reference) and D (fused_cl) at B=16 on the study GPU,
# to set the pair jobs' time limit. Same gates and ledger as the study jobs. Counts against the GPU-hour cap.
set -u
export M1_EXPECTED_GPU="${M1_EXPECTED_GPU:-NVIDIA GeForce RTX 3090}"
W=/grogu/user/junhong3/lpwm-work; E=$W/experiments/fused-integration; M=$E/m1; PY=$W/venvs/lpwm-stn/bin/python
T=$E/tree-m1; CFG=$W/data/static_bair128_vgg.json
export TORCH_HOME=$W/cache/torch
JOB=${SLURM_JOB_ID:-nojob}; NODE=$(hostname -s); START=$(date +%s); STATUS=running; RC=-1; GPULINE=""
mkdir -p $M/ledger $M/runs $M/slurm
LED=$M/ledger/$JOB.json
write_ledger() {
  local END; END=$(date +%s)
  $PY - "$LED" "$JOB" "timing" "$NODE" "$STATUS" "$RC" "$START" "$END" "$GPULINE" <<'PYEOF'
import json, sys
led, job, kind, node, status, rc, start, end, gpu = sys.argv[1:10]
json.dump({"job": job, "kind": "m1_" + kind, "node": node, "gpu_line": gpu, "status": status, "exit_code": int(rc), "start_epoch": int(start), "end_epoch": int(end),
           "elapsed_s": int(end) - int(start), "elapsed_gpu_hours": (int(end) - int(start)) / 3600, "limit_s": 900}, open(led, "w"), indent=1)
PYEOF
}
trap 'write_ledger' EXIT
trap 'STATUS=terminated_by_signal; exit 143' TERM
write_ledger
source $T/benchmarks/stn/m1_hwcheck.sh
HW=$(m1_check_gpu); HRC=$?
echo "$HW"; GPULINE=$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader,nounits 2>&1 | head -1)
if [ $HRC -ne 0 ]; then STATUS=hw_mismatch; RC=3; echo "$NODE" >> $M/blocklist.txt; exit 3; fi
if ! (cd $T && sha256sum -c $M/tree-m1.sha256 --quiet >/dev/null 2>&1); then STATUS=snapshot_modified; RC=4; exit 4; fi
OUT=$M/runs/timing-${JOB}; mkdir -p $OUT
for P in reference fused_cl; do
  $PY $E/final/full_stack_scan.py --repo-root $T --config $CFG --path $P --batch-size 16 --warmup-steps 10 --timed-steps 40 --seed 0 --out $OUT/scan-$P.json > $OUT/scan-$P.log 2>&1
  echo "$P exit=$?"
done
$PY - <<PYEOF
import json
a = json.load(open("$OUT/scan-reference.json")); d = json.load(open("$OUT/scan-fused_cl.json"))
ta, td = a["event_ms_median"], d["event_ms_median"]
# A6000 pair, measured: total wall = 1.06 x (8 epochs x 750 steps x step time) for each path
pair_min = 1.06 * 6000 * (ta + td) / 1000 / 60
print(f"3090 step ms: A {ta:.1f} (status {a['status']}, path_check {a['path_check']['ok']}), D {td:.1f} (status {d['status']}, path_check {d['path_check']['ok']}) | D vs A {100*(td/ta-1):+.1f}%")
print(f"peak allocated MB: A {a['steady_peak_allocated_mb']:.0f}, D {d['steady_peak_allocated_mb']:.0f}; reserved A {a['steady_peak_reserved_mb']:.0f}, D {d['steady_peak_reserved_mb']:.0f}")
print(f"PREDICTED pair wall time on this GPU: {pair_min:.1f} min (A6000 measured 61)")
json.dump({"step_ms_A": ta, "step_ms_D": td, "predicted_pair_min": pair_min}, open("$OUT/prediction.json", "w"))
PYEOF
STATUS=ok; RC=0
echo "OUT=$OUT"
