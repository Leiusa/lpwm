#!/bin/bash
#SBATCH --job-name=compile-fix
#SBATCH --partition=all
#SBATCH --constraint=rtx3090
#SBATCH --exclude=grogu-4-13,grogu-4-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/grogu/user/junhong3/lpwm-work/experiments/compile-probe/slurm/%x-%j.out
# torch.compile fixes (patch sizes as Python int; List[int] instead of torch.Size in lpwm_stn/reference.py), one RTX 3090:
#   1. regression tests on the OLD tree (tree-m1) and the FIXED tree (tree-fix): test_stn.py (bit-exact vs golden, cpu+cuda),
#      test_decoder_fused_integration.py, test_particle_dec_channels_last.py
#   2. neutrality: the same seeded eager computations in both trees, per path, compared byte for byte (compile_fix_neutrality.py)
#   3. the compile probe (compile_probe.py, unchanged) on the fixed tree, A then D
# Nothing is edited here; tree-fix is read-only and verified against its manifest. Outcome -> compile-probe/ledger/<jobid>.json.
set -u
export M1_EXPECTED_GPU="NVIDIA GeForce RTX 3090"
W=/grogu/user/junhong3/lpwm-work; P=$W/experiments/compile-probe; PY=$W/venvs/lpwm-stn/bin/python
OLD=$W/experiments/fused-integration/tree-m1; NEW=$P/tree-fix; CFG=$W/data/static_bair128_vgg.json
export TORCH_HOME=$W/cache/torch
LIMIT_S=5400   # must match --time
JOB=${SLURM_JOB_ID:-nojob}; NODE=$(hostname -s); START=$(date +%s); STATUS=running; RC=-1; GPULINE=""
mkdir -p $P/ledger $P/runs $P/slurm
LED=$P/ledger/$JOB.json
write_ledger() {
  local END; END=$(date +%s)
  $PY - "$LED" "$JOB" "$NODE" "$STATUS" "$RC" "$START" "$END" "$GPULINE" "$LIMIT_S" <<'PYEOF'
import json, sys
led, job, node, status, rc, start, end, gpu, limit = sys.argv[1:10]
json.dump({"job": job, "kind": "compile_fix", "node": node, "gpu_line": gpu, "status": status, "exit_code": int(rc),
           "start_epoch": int(start), "end_epoch": int(end), "elapsed_s": int(end) - int(start),
           "elapsed_gpu_hours": (int(end) - int(start)) / 3600, "limit_s": int(limit)}, open(led, "w"), indent=1)
PYEOF
}
trap 'write_ledger' EXIT
trap 'STATUS=terminated_by_signal; exit 143' TERM
write_ledger
source $OLD/benchmarks/stn/m1_hwcheck.sh
HW=$(m1_check_gpu); HRC=$?
echo "$HW"; GPULINE=$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader,nounits 2>&1 | head -1)
if [ $HRC -ne 0 ]; then STATUS=hw_mismatch; RC=3; exit 3; fi
if ! (cd $NEW && sha256sum -c $P/tree-fix.sha256 --quiet >/dev/null 2>&1); then STATUS=snapshot_modified; RC=4; echo "tree-fix does not match its manifest"; exit 4; fi
OUT=$P/runs/fix-$JOB; mkdir -p $OUT/eval/lpips $OUT/scripts $OUT/tests
cp $W/weights/eval/lpips/vgg.pth $OUT/eval/lpips/vgg.pth     # LossLPIPS reads eval/lpips/vgg.pth relative to the cwd
cp $0 $P/tools/compile_probe.py $P/tools/compile_fix_neutrality.py $P/compile_fix.patch $P/tree-fix.sha256 $OUT/scripts/ 2>/dev/null
{ echo "job $JOB node $NODE start_epoch $START"; nvidia-smi -L;
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.max.sm --format=csv; } > $OUT/env.txt 2>&1
cd $OUT
RC=0

echo "== 1. regression tests (old vs fixed tree)"
for TREE in old new; do
  if [ $TREE = old ]; then TD=$OLD; else TD=$NEW; fi
  for T in "tests/stn/test_stn.py --device cpu" "tests/stn/test_stn.py --device cuda" \
           "tests/composite/test_decoder_fused_integration.py" "tests/composite/test_particle_dec_channels_last.py"; do
    NAME=$(echo "$T" | sed 's#tests/##; s#[/ ]#_#g; s#\.py##; s#--device_##')
    ( cd $OUT && timeout 1200 $PY $TD/$T ) > $OUT/tests/$TREE-$NAME.log 2>&1
    R=$?; echo "$TREE $NAME exit=$R | $(tail -1 $OUT/tests/$TREE-$NAME.log | cut -c1-160)"
  done
done

echo "== 2. neutrality (old vs fixed tree, eager, byte for byte)"
for PATHNAME in reference fused_cl; do
  $PY $P/tools/compile_fix_neutrality.py run --repo-root $OLD --config $CFG --path $PATHNAME --out $OUT/neutral-old-$PATHNAME.pt > $OUT/neutral-old-$PATHNAME.log 2>&1
  R1=$?
  $PY $P/tools/compile_fix_neutrality.py run --repo-root $NEW --config $CFG --path $PATHNAME --out $OUT/neutral-new-$PATHNAME.pt > $OUT/neutral-new-$PATHNAME.log 2>&1
  R2=$?
  if [ $R1 -eq 0 ] && [ $R2 -eq 0 ]; then
    $PY $P/tools/compile_fix_neutrality.py compare $OUT/neutral-old-$PATHNAME.pt $OUT/neutral-new-$PATHNAME.pt --out $OUT/neutrality-$PATHNAME.json
    R3=$?
  else
    R3=9; tail -5 $OUT/neutral-old-$PATHNAME.log $OUT/neutral-new-$PATHNAME.log | cut -c1-300
  fi
  echo "$PATHNAME neutrality run old=$R1 new=$R2 compare=$R3"
  if [ $R3 -ne 0 ]; then RC=8; fi
done

echo "== 3. compile probe on the fixed tree"
export TORCH_LOGS="graph_breaks,recompiles"
for PATHNAME in reference fused_cl; do
  export TORCHINDUCTOR_CACHE_DIR=$OUT/cache-$PATHNAME/inductor TRITON_CACHE_DIR=$OUT/cache-$PATHNAME/triton   # cold, per path
  $PY $P/tools/compile_probe.py --repo-root $NEW --config $CFG --path $PATHNAME --out $OUT/probe-$PATHNAME.json > $OUT/probe-$PATHNAME.log 2>&1
  R=$?; echo "$PATHNAME probe exit=$R"; grep '^\[' $OUT/probe-$PATHNAME.log | cut -c1-300
  if [ $R -ne 0 ]; then RC=$R; fi
done
unset TORCH_LOGS
if [ $RC -eq 0 ]; then STATUS=ok; else STATUS=failed; fi
echo "OUT=$OUT"
exit $RC
