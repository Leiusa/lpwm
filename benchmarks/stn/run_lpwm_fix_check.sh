#!/bin/bash
# numpy-int fix (particle decoder traceable by torch.compile) + backward-by-module profile, LPWM, one GPU, sequential.
#
#   cd ~/dlp-lpwm-optimization && bash lpwm/benchmarks/stn/run_lpwm_fix_check.sh cfg_lpwm_C_s1_tf32.json
#
# Old tree = the commit just before the fix ("Patch sizes as plain int"), checked out once as a git worktree in
# ./lpwm_nofix; new tree = ./lpwm. Both are run with the same scripts (from ./lpwm) and the same config.
#   1. neutrality: eager, fp32 matmul, old vs new -> bitwise_identical / within_repeatability / DIFFERS
#   2. compiled step time + graph breaks, old vs new, torch.compile default and reduce-overhead, TF32 matmul
#   3. eager per-module forward + BACKWARD profile (new tree, depth 3, TF32 matmul)
set -euo pipefail
CFG=${1:?usage: run_lpwm_fix_check.sh CONFIG.json}
S=lpwm/benchmarks/stn
FIX=$(git -C lpwm log --format=%H -1 --grep='^Patch sizes as plain int')
[ -n "$FIX" ] || { echo "fix commit not found in ./lpwm (git pull first)"; exit 1; }
if [ ! -d lpwm_nofix ]; then
  git -C lpwm worktree add --detach ../lpwm_nofix "$FIX^"
fi
git -C lpwm_nofix checkout -q --detach "$FIX^"
OUT=fixcheck_$(date +%m%d_%H%M)
mkdir -p "$OUT"
{ echo "config $CFG"; echo "new $(git -C lpwm rev-parse --short HEAD)  old $(git -C lpwm_nofix rev-parse --short HEAD)";
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; } | tee "$OUT/info.txt"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "another process is using the GPU; stop it first (timings would be wrong)"; exit 1
fi

echo "== 1/3 neutrality (eager)"
python $S/lpwm_fix_neutrality.py run --repo-root lpwm_nofix --config "$CFG" --out "$OUT/neutral_old.pt" 2>&1 | tee "$OUT/neutral_old.log"
python $S/lpwm_fix_neutrality.py run --repo-root lpwm --config "$CFG" --out "$OUT/neutral_new.pt" 2>&1 | tee "$OUT/neutral_new.log"
python $S/lpwm_fix_neutrality.py compare "$OUT/neutral_old.pt" "$OUT/neutral_new.pt" --out "$OUT/neutrality.json" 2>&1 | tee "$OUT/neutrality.log"
rm -f "$OUT"/neutral_*.pt

echo "== 2/3 compiled step, old vs new"
for MODE in default reduce-overhead; do
  for TREE in lpwm_nofix lpwm; do
    python $S/lpwm_profile.py --config "$CFG" --repo-root $TREE --tf32-matmul --compile $MODE --steps 20 \
      --profile-steps 1 --out "$OUT/compile_${MODE}_${TREE}" 2>&1 | grep -v Warning | tee "$OUT/compile_${MODE}_${TREE}.log"
    rm -f "$OUT/compile_${MODE}_${TREE}/trace.json"
  done
done

echo "== 3/3 eager forward + backward by module"
python $S/lpwm_profile.py --config "$CFG" --repo-root lpwm --tf32-matmul --depth 3 --out "$OUT/eager_bwd" 2>&1 \
  | tee "$OUT/eager_bwd.log"
rm -f "$OUT/eager_bwd/trace.json"

echo
echo "== summary"
cat "$OUT/neutrality.log" | head -1
for f in "$OUT"/compile_*.log; do echo "$(basename "$f" .log): $(grep -a '^step ' "$f") | $(grep -a 'dynamo:' "$f" | sed 's/.*dynamo: //')"; done
echo "all results in $OUT/  (send me: $OUT/*.log and $OUT/*/lpwm_profile.json)"
