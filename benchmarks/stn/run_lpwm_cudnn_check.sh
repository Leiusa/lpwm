#!/bin/bash
# cuDNN algorithm selection check, LPWM, one GPU, sequential (option A). Measurement only; nothing in the repo changes.
#
#   cd ~/dlp-lpwm-optimization && bash lpwm/benchmarks/stn/run_lpwm_cudnn_check.sh cfg_lpwm_C_s1_tf32.json
#
# All runs: config's STN path, TF32 matmul, same seed and batches. train_lpwm.py today = benchmark off, deterministic on.
#   1. compiled step (torch.compile reduce-overhead = the standard training path), 4 combinations
#        base   benchmark off, deterministic on   (today)
#        bench  benchmark on,  deterministic on
#        nondet benchmark off, deterministic off
#        both   benchmark on,  deterministic off
#      then base again (drift check: how much the same setting moves between runs)
#   2. eager per-module forward + backward for base and both (which DLP modules change)
set -euo pipefail
CFG=${1:?usage: run_lpwm_cudnn_check.sh CONFIG.json}
S=lpwm/benchmarks/stn
OUT=cudnncheck_$(date +%m%d_%H%M)
mkdir -p "$OUT"
{ echo "config $CFG"; echo "code $(git -C lpwm rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; } | tee "$OUT/info.txt"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "another process is using the GPU; stop it first (timings would be wrong)"; exit 1
fi
flags() {
  case $1 in
    base*) echo "" ;;
    bench) echo "--cudnn-benchmark" ;;
    nondet) echo "--cudnn-nondeterministic" ;;
    both) echo "--cudnn-benchmark --cudnn-nondeterministic" ;;
  esac
}

echo "== 1/2 compiled step (reduce-overhead)"
for V in base bench nondet both base_again; do
  if ! python $S/lpwm_profile.py --config "$CFG" --tf32-matmul --compile reduce-overhead --steps 30 --profile-steps 1 \
    $(flags $V) --out "$OUT/compiled_$V" > "$OUT/compiled_$V.log" 2>&1; then
    echo "$V: FAILED, last lines of $OUT/compiled_$V.log:"; tail -n 8 "$OUT/compiled_$V.log"
  fi
  rm -f "$OUT/compiled_$V/trace.json"
  echo "$V: $(grep -a '^step ' "$OUT/compiled_$V.log" || true)"
done

echo "== 2/2 eager forward + backward by module (base vs both)"
for V in base both; do
  if python $S/lpwm_profile.py --config "$CFG" --tf32-matmul --depth 3 $(flags $V) --out "$OUT/eager_$V" > "$OUT/eager_$V.log" 2>&1; then
    python $S/lpwm_profile_tables.py "$OUT/eager_$V/lpwm_profile.json" --top 20 > "$OUT/eager_$V.txt"
  else
    { echo "FAILED, last lines of $OUT/eager_$V.log:"; tail -n 8 "$OUT/eager_$V.log"; } | tee "$OUT/eager_$V.txt"
  fi
  rm -f "$OUT/eager_$V/trace.json"
done

{ echo "== compiled step (reduce-overhead, TF32 matmul), median of 30 steps"
  for V in base bench nondet both base_again; do
    echo "$V: $(grep -a '^step ' "$OUT/compiled_$V.log" || { echo FAILED; tail -n 3 "$OUT/compiled_$V.log"; })"; done
  for V in base both; do echo; echo "== eager $V"; cat "$OUT/eager_$V.txt"; done
} > "$OUT/report.txt"
cat "$OUT/report.txt"
echo "download $OUT/report.txt and send it to me"
