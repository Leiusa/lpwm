#!/bin/bash
# cuDNN algorithm selection: last speed point + short quality check (LPWM, one GPU, sequential).
#
#   cd ~/dlp-lpwm-optimization && bash lpwm/benchmarks/stn/run_lpwm_cudnn_quality.sh cfg_lpwm_C_s1_tf32.json [SEEDS="1"]
#
# 1. speed (lpwm_profile.py, compile mode default, TF32 matmul, median of 30 steps), same session:
#      def_base   benchmark off, deterministic on   (today)
#      def_bench  benchmark on,  deterministic on
#      def_both   benchmark on,  deterministic off  (not measured before)
# 2. quality, per seed: two train_lpwm.py runs from the given config, differing ONLY in the cuDNN keys
#      base: cudnn_benchmark=false, cudnn_deterministic=true     new: cudnn_benchmark=true, cudnn_deterministic=false
#    both: torch_compile "default" (reduce-overhead + benchmark runs out of memory), 4 epochs x 400 steps,
#    log_step_timing on; everything else as in the config (seed, STN path, TF32 matmul, eval).
#    Then compare_runs.py (per-epoch step time, train loss/PSNR, val metrics) and lpwm_compare_grid.py (fixed val
#    episodes, prediction + decomposition images) on the final checkpoints. No pass/fail threshold is applied.
set -uo pipefail
CFG=${1:?usage: run_lpwm_cudnn_quality.sh CONFIG.json [SEEDS]}
SEEDS=${2:-1}
S=lpwm/benchmarks/stn
export TORCH_HOME=${TORCH_HOME:-$HOME/dlp-lpwm-optimization/weights/torch}
OUT=cudnnq_$(date +%m%d_%H%M)
mkdir -p "$OUT"
{ echo "config $CFG  seeds $SEEDS"; echo "code $(git -C lpwm rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; } | tee "$OUT/info.txt"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "another process is using the GPU; stop it first"; exit 1
fi

echo "== 1/2 speed (compile default)"
for r in "def_base|" "def_bench|--cudnn-benchmark" "def_both|--cudnn-benchmark --cudnn-nondeterministic"; do
  IFS='|' read -r name flags <<< "$r"
  python $S/lpwm_profile.py --config "$CFG" --tf32-matmul --compile default --steps 30 --profile-steps 1 $flags \
    --out "$OUT/$name" > "$OUT/$name.log" 2>&1 || echo "$name: FAILED"
  rm -f "$OUT/$name/trace.json"
  echo "$name: $(grep -a '^step ' "$OUT/$name.log" || tail -n 1 "$OUT/$name.log")" | tee -a "$OUT/speed.txt"
done

echo "== 2/2 quality"
for SEED in $SEEDS; do
  for V in base new; do
    C="$OUT/cfg_s${SEED}_$V.json"
    python - "$CFG" "$C" "$SEED" "$V" <<'EOF'
import json, sys
src, dst, seed, v = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = json.load(open(src))
c.update(seed=seed, torch_compile="default", num_epochs=4, max_steps_per_epoch=400, log_step_timing=True,
         cudnn_benchmark=(v == "new"), cudnn_deterministic=(v != "new"), run_prefix=f"_lpwm_cudnnq_s{seed}_{v}")
json.dump(c, open(dst, "w"), indent=1)
EOF
    echo "-- seed $SEED $V: $(date +%H:%M)"
    python lpwm/train_lpwm.py -d "$C" > "$OUT/train_s${SEED}_$V.log" 2>&1 || echo "seed $SEED $V: training FAILED (see $OUT/train_s${SEED}_$V.log)"
    grep -a "path check" "$OUT/train_s${SEED}_$V.log" | head -1
  done
  python $S/compare_runs.py "$OUT/train_s${SEED}_base.log" "$OUT/train_s${SEED}_new.log" --labels base,new \
    > "$OUT/compare_s${SEED}.txt" 2>&1
  RA=$(ls -d *_bair_gddlp_lpwm_cudnnq_s${SEED}_base | tail -n 1)
  RB=$(ls -d *_bair_gddlp_lpwm_cudnnq_s${SEED}_new | tail -n 1)
  python $S/lpwm_compare_grid.py --config "$OUT/cfg_s${SEED}_base.json" \
    --ckpt-a "$RA/saves/bair_gddlp_lpwm_cudnnq_s${SEED}_base.pth" --ckpt-b "$RB/saves/bair_gddlp_lpwm_cudnnq_s${SEED}_new.pth" \
    --labels base,new --tf32-a 1 --tf32-b 1 --out "$OUT/grid_s${SEED}" > "$OUT/grid_s${SEED}.log" 2>&1 \
    || echo "seed $SEED: compare grid FAILED (see $OUT/grid_s${SEED}.log)"
done

{ cat "$OUT/info.txt"; echo; echo "== speed (compile default, TF32 matmul, median of 30 steps)"; cat "$OUT/speed.txt"
  for SEED in $SEEDS; do
    echo; echo "== quality seed $SEED"; cat "$OUT/compare_s${SEED}.txt"
    echo; grep -a "^episode" "$OUT/grid_s${SEED}.log"
  done
} > "$OUT/report.txt"
cat "$OUT/report.txt"
echo "download $OUT/report.txt (and look at $OUT/grid_s*/pred_ep*.png, decomp_ep*.png)"
