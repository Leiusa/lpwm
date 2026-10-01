#!/bin/bash
#SBATCH --job-name=m1-smoke
#SBATCH --partition=all
#SBATCH --constraint=A6000
#SBATCH --exclude=grogu-4-13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/grogu/user/junhong3/lpwm-work/experiments/fused-integration/m1/slurm/%x-%j.out
# Milestone-1 smoke test (no science): validates the whole pipeline on tiny data (both trainings + four evaluations, runtime
# evidence, hardware gate) and re-evaluates the existing seed-2 A6000 checkpoints on the full validation set to check the new
# evaluation script against the earlier recorded numbers. Passing writes SMOKE_PASS into the smoke output directory.
set -u
export M1_EXPECTED_GPU="${M1_EXPECTED_GPU:-NVIDIA RTX A6000}"   # the study's one GPU model; the gate below enforces it
W=/grogu/user/junhong3/lpwm-work; E=$W/experiments/fused-integration; M=$E/m1; PY=$W/venvs/lpwm-stn/bin/python
T=$E/tree-m1; CFG=$W/data/static_bair128_vgg.json
export TORCH_HOME=$W/cache/torch
JOB=${SLURM_JOB_ID:-nojob}; NODE=$(hostname -s); START=$(date +%s); STATUS=running; RC=-1; GPULINE=""; SEED=-1; ORDER=smoke
mkdir -p $M/ledger $M/runs $M/slurm
LED=$M/ledger/$JOB.json
write_ledger() {
  local END; END=$(date +%s)
  $PY - "$LED" "$JOB" "smoke" "$NODE" "$STATUS" "$RC" "$START" "$END" "$GPULINE" <<'PYEOF'
import json, sys
led, job, kind, node, status, rc, start, end, gpu = sys.argv[1:10]
json.dump({"job": job, "kind": "m1_" + kind, "node": node, "gpu_line": gpu, "status": status, "exit_code": int(rc), "start_epoch": int(start), "end_epoch": int(end),
           "elapsed_s": int(end) - int(start), "elapsed_gpu_hours": (int(end) - int(start)) / 3600}, open(led, "w"), indent=1)
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
OUT=$M/runs/smoke-${JOB}; mkdir -p $OUT/scripts; cp $0 $OUT/scripts/
{ echo "job $JOB node $NODE"; nvidia-smi -L; sha256sum $M/tree-m1.sha256 $CFG; } > $OUT/env.txt 2>&1
# tiny data: first 3 train episodes, first 2 validation episodes (symlinks into the real subset)
SD=$M/smoke_data; rm -rf $SD; mkdir -p $SD/train $SD/val
for e in $(ls $W/data/bair_256_ours_subset/train | sort -n | head -3); do ln -s $W/data/bair_256_ours_subset/train/$e $SD/train/$e; done
for e in $(ls $W/data/bair_256_ours_subset/val | sort -n | head -2); do ln -s $W/data/bair_256_ours_subset/val/$e $SD/val/$e; done
NVAL=$(( $(ls $SD/val | wc -l) * 30 ))
GATE=0; [ "$M1_EXPECTED_GPU" = "NVIDIA RTX A6000" ] && GATE=1
echo "== unit checks of the hardware decision logic"
$PY - <<'PYEOF' > $OUT/unit_checks.log 2>&1; UC=$?
import sys
sys.path.insert(0, "/grogu/user/junhong3/lpwm-work/experiments/fused-integration/tree-m1/benchmarks/stn")
import m1_common as MC
A6, R3, ADA, TI = "NVIDIA RTX A6000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX 6000 Ada Generation", "NVIDIA GeForce RTX 3080 Ti"
ok = [[A6, "GPU-abc", "49140"]]
f = lambda node, rows, tname, exp: MC.check_gpu_facts(node, rows, tname, expected=exp)
assert f("grogu-1-25", ok, A6, A6) == []
assert f("grogu-4-13", ok, A6, A6)                                         # excluded node
assert f("grogu-0-25", [[ADA, "GPU-abc", "49140"]], ADA, A6)              # Ada variant is not an A6000
assert f("grogu-1-25", [[TI, "GPU-abc", "12288"]], TI, A6)                # 3080 Ti is not an A6000
assert f("grogu-1-25", ok + ok, A6, A6)                                    # two visible GPUs
assert f("grogu-4-8", [[R3, "GPU-b", "24576"]], R3, A6)                   # a 3090 is rejected when A6000 is expected
assert f("grogu-4-8", [[R3, "GPU-b", "24576"]], R3, R3) == []             # and accepted when the study expects a 3090
print("hardware decision logic: all 7 cases behave")
PYEOF
cat $OUT/unit_checks.log
echo "== tiny pair (1 epoch each on 3 train / 2 val episodes) + four evaluations"
$PY $T/benchmarks/stn/m1_train_pair.py --repo-root $T --config $CFG --seed 0 --epochs 1 --order reference,fused_cl --lpips-head $W/weights/eval/lpips/vgg.pth \
    --out $OUT/tiny --smoke --root-override $SD --expect-val-images $NVAL > $OUT/tiny.log 2>&1
TRC=$?; tail -6 $OUT/tiny.log | cut -c1-300
echo "== four evaluations of the existing seed-2 A6000 checkpoints on the full validation set"
S2=$E/fsq-seed2-3967367
$PY $T/benchmarks/stn/m1_eval.py --repo-root $T --config $CFG --ckpt-a $S2/220926_211825_bair_gdlp_reference_p1/saves/bair_gdlp_reference_p1.pth \
    --ckpt-d $S2/220926_215331_bair_gdlp_fused_cl_p1/saves/bair_gdlp_fused_cl_p1.pth --out $OUT/eval_seed2_existing.json --expect-val-images 1200 > $OUT/eval_seed2_existing.log 2>&1
ERC=$?; tail -6 $OUT/eval_seed2_existing.log | cut -c1-300
$PY $T/benchmarks/stn/m1_smoke_check.py --smoke-dir $OUT --unit-rc $UC --tiny-rc $TRC --eval-rc $ERC --gate-recorded $GATE --recorded $E/final_recorded_seed2.json > $OUT/smoke_check.log 2>&1
CRC=$?; cat $OUT/smoke_check.log
echo "== budget gate + file lock (as used by the seed jobs)"
( flock 9; $PY $M/jobs/m1_budget_gate.py $M/ledger $JOB 5100 8.0 ) 9>$M/ledger.lock; GRC=$?
echo "budget gate rc=$GRC (expected 0)"
RC=$CRC; if [ $GRC -ne 0 ]; then RC=1; echo "SMOKE_FAIL: budget gate/flock check"; fi
if [ $RC -eq 0 ]; then STATUS=ok; else STATUS=failed; fi
echo "OUT=$OUT"
exit $RC
