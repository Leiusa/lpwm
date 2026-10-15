#!/bin/bash
# Single-image DLP: where does the optimized training step still spend its time? (dlp_profile.py, one GPU, sequential)
#
#   cd ~/dlp-lpwm-optimization && bash lpwm/benchmarks/stn/run_dlp_profile.sh static_bair128_vgg_lambda.json [OVERRIDE_JSON]
#   OVERRIDE_JSON is applied to every row, e.g. '{"n_kp_enc": "all"}' = no particle filtering (Tal's setting).
#
# Rows (all: tf32_matmul off, cudnn deterministic on, channels_last off; everything else from the config):
#   A_eager     stn_backend=reference, no fused composite, eager             (original; per-module fwd/bwd)
#   C_eager_cb  triton + fused composite + cudnn_benchmark, eager             (per-module fwd/bwd of the optimized math)
#   C_def_cb    C_eager_cb + torch_compile "default"                          (kernel breakdown after compile)
#   C_ro_cb     C_eager_cb + torch_compile "reduce-overhead"                  (the recommended config, 49.9 ms in the matrix)
# Each row reports: step time (GPU events, = [step-timing]), wall time per step of the real loop (with train_dlp.py's
# per-step .item() reads), without them, without the DataLoader, GPU busy %, GPU time by kernel category.
set -uo pipefail
CFG=${1:?usage: run_dlp_profile.sh CONFIG.json [OVERRIDE_JSON]}
COMMON=${2:-"{}"}
S=lpwm/benchmarks/stn
export TORCH_HOME=${TORCH_HOME:-$HOME/dlp-lpwm-optimization/weights/torch}
OUT=results/dlp_prof_$(date +%m%d_%H%M)
mkdir -p "$OUT"
{ echo "config $CFG  override $COMMON"; echo "code $(git -C lpwm rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; } | tee "$OUT/info.txt"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "another process is using the GPU; stop it first"; exit 1
fi
ROWS=(
  'A_eager|eager|{"stn_backend": "reference", "fused_composite": false}'
  'C_eager_cb|eager|{"stn_backend": "triton", "fused_composite": true, "cudnn_benchmark": true}'
  'C_def_cb|default|{"stn_backend": "triton", "fused_composite": true, "cudnn_benchmark": true}'
  'C_ro_cb|reduce-overhead|{"stn_backend": "triton", "fused_composite": true, "cudnn_benchmark": true}'
)
for r in "${ROWS[@]}"; do
  IFS='|' read -r name mode over <<< "$r"
  python - "$CFG" "$OUT/cfg_$name.json" "$over" "$COMMON" <<'EOF'
import json, sys
src, dst, over, common = sys.argv[1:5]
c = json.load(open(src))
c.update(particle_dec_channels_last=False, cudnn_benchmark=False, cudnn_deterministic=True, tf32_matmul=False)
c.update(json.loads(common))
c.update(json.loads(over))
json.dump(c, open(dst, "w"), indent=1)
EOF
  echo "-- $name ($mode, $(date +%H:%M))"
  python $S/dlp_profile.py --config "$OUT/cfg_$name.json" --mode "$mode" --out "$OUT/$name" > "$OUT/$name.log" 2>&1 \
    || echo "$name: FAILED ($(grep -a -m1 -o 'OutOfMemoryError\|[A-Za-z]*Error' "$OUT/$name.log" | head -1))"
  rm -f "$OUT/$name/trace.json"
  grep -a -A3 '^-- 1\.' "$OUT/$name.log" | head -4
done
{ cat "$OUT/info.txt"
  for r in "${ROWS[@]}"; do
    name=${r%%|*}
    echo; echo "=================== $name"
    sed -n '/^== /,$p' "$OUT/$name.log" 2>/dev/null || echo FAILED
  done
} > "$OUT/report.txt"
echo "download $OUT/report.txt and send it to me"
