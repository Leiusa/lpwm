#!/bin/bash
# cuDNN algorithm selection check, LPWM, one GPU, sequential (option A). Measurement only; nothing in the repo changes.
#
#   cd ~/dlp-lpwm-optimization && bash lpwm/benchmarks/stn/run_lpwm_cudnn_check.sh cfg_lpwm_C_s1_tf32.json [CAP_MIB=2048]
#
# All runs: config's STN path, TF32 matmul, same seed and batches. train_lpwm.py today = benchmark off, deterministic on.
# First attempt: benchmark on under reduce-overhead ran out of memory (94.5 GB GH200; cuDNN picked algorithms with large
# workspaces, which CUDA graphs keep in their private pool). So benchmark runs here cap the cuDNN workspace with
# CUDNN_CONV_WSCAP_DBG (MiB), and one uncapped benchmark run uses compile mode default (no CUDA graphs).
#   1. compiled step, one process per row (name  mode  cudnn flags  workspace cap):
#        ro_base          reduce-overhead  benchmark off, deterministic on   -       (today)
#        ro_nondet        reduce-overhead  benchmark off, deterministic off  -
#        ro_bench_cap     reduce-overhead  benchmark on,  deterministic on   $CAP MiB
#        ro_both_cap      reduce-overhead  benchmark on,  deterministic off  $CAP MiB
#        ro_base_again    reduce-overhead  as ro_base (drift check)
#        def_base         default          benchmark off, deterministic on   -
#        def_bench        default          benchmark on,  deterministic on   -       (uncapped)
#   2. eager per-module forward + backward: base vs both (benchmark + nondeterministic, capped)
set -uo pipefail
CFG=${1:?usage: run_lpwm_cudnn_check.sh CONFIG.json [WORKSPACE_CAP_MIB]}
CAP=${2:-2048}
S=lpwm/benchmarks/stn
OUT=cudnncheck_$(date +%m%d_%H%M)
mkdir -p "$OUT"
{ echo "config $CFG  workspace cap $CAP MiB"; echo "code $(git -C lpwm rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; } | tee "$OUT/info.txt"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "another process is using the GPU; stop it first (timings would be wrong)"; exit 1
fi
B="--cudnn-benchmark"; N="--cudnn-nondeterministic"
ROWS=(
  "ro_base|reduce-overhead||"
  "ro_nondet|reduce-overhead|$N|"
  "ro_bench_cap|reduce-overhead|$B|$CAP"
  "ro_both_cap|reduce-overhead|$B $N|$CAP"
  "ro_base_again|reduce-overhead||"
  "def_base|default||"
  "def_bench|default|$B|"
)
run() {  # name mode flags cap -> log; prints the step line or FAILED
  local name=$1 mode=$2 flags=$3 cap=$4 env=()
  [ -n "$cap" ] && env=(env CUDNN_CONV_WSCAP_DBG=$cap)
  local cmode=(); [ "$mode" != eager ] && cmode=(--compile "$mode" --steps 30 --profile-steps 1)
  [ "$mode" = eager ] && cmode=(--depth 3)
  if ! "${env[@]}" python $S/lpwm_profile.py --config "$CFG" --tf32-matmul "${cmode[@]}" $flags --out "$OUT/$name" \
      > "$OUT/$name.log" 2>&1; then
    echo "$name: FAILED ($(grep -a -m1 -o 'OutOfMemoryError\|Error[^:]*' "$OUT/$name.log" | head -1))"
  fi
  rm -f "$OUT/$name/trace.json"
}

echo "== 1/2 compiled step"
for r in "${ROWS[@]}"; do
  IFS='|' read -r name mode flags cap <<< "$r"
  run "$name" "$mode" "$flags" "$cap"
  echo "$name: $(grep -a '^step ' "$OUT/$name.log" || echo '-')"
done

echo "== 2/2 eager forward + backward by module"
run eager_base eager "" ""
run eager_both_cap eager "$B $N" "$CAP"
for V in eager_base eager_both_cap; do
  if [ -f "$OUT/$V/lpwm_profile.json" ]; then
    python $S/lpwm_profile_tables.py "$OUT/$V/lpwm_profile.json" --top 20 > "$OUT/$V.txt"
  else
    { echo "FAILED:"; tail -n 3 "$OUT/$V.log"; } > "$OUT/$V.txt"
  fi
done

{ cat "$OUT/info.txt"; echo; echo "== compiled step, median of 30 steps (TF32 matmul)"
  for r in "${ROWS[@]}"; do
    IFS='|' read -r name mode flags cap <<< "$r"
    echo "$name [$mode${flags:+ $flags}${cap:+ cap=$cap}]: $(grep -a '^step ' "$OUT/$name.log" || { echo FAILED; tail -n 1 "$OUT/$name.log"; })"
  done
  for V in eager_base eager_both_cap; do echo; echo "== $V"; cat "$OUT/$V.txt"; done
} > "$OUT/report.txt"
cat "$OUT/report.txt"
echo "download $OUT/report.txt and send it to me"
