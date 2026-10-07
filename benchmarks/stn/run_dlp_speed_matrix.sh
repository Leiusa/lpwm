#!/bin/bash
# Single-image DLP (train_dlp.py) speed matrix: which switches make a training step faster on THIS GPU.
# Each row is a short real training run (same config, seed, data order), one at a time on one GPU.
#
#   cd ~/dlp-lpwm-optimization && bash lpwm/benchmarks/stn/run_dlp_speed_matrix.sh static_bair128_vgg_lambda.json [STEPS=300] [OVERRIDE_JSON]
#   OVERRIDE_JSON is applied to every run. '{"n_kp_enc": "all"}' = no particle filtering: encoder and decoder use all
#   prior particles (n_kp_enc set to the prior count models.py derives: min(n_kp_per_patch * (image_size // patch_size)^2,
#   n_kp_prior)).
#
# Every run: 2 epochs x STEPS steps, warmup_epoch=1 (epoch 0 = warmup, epoch 1 = normal training, the one reported),
# log_step_timing on, eval_im_metrics off; everything else from the config. Reported: epoch-1 median step time and
# peak allocated memory from the [step-timing] line, and the first-step path check (which path really ran).
#   A          stn_backend=reference, no fused composite, eager                 (original)
#   C          stn_backend=triton + fused_composite, eager
#   C_comp     C + torch_compile "default"
#   C_ro       C + torch_compile "reduce-overhead"
#   C_comp_cb  C + torch_compile "default" + cudnn_benchmark
#   C_comp_cb_tf32  C_comp_cb + tf32_matmul                                     (precision change)
#   C_ro_cb    C + torch_compile "reduce-overhead" + cudnn_benchmark             (ran out of memory in LPWM)
#   A_again    A repeated (drift check)
# Speed only: no quality is measured here.
set -uo pipefail
CFG=${1:?usage: run_dlp_speed_matrix.sh CONFIG.json [STEPS]}
STEPS=${2:-300}
COMMON=${3:-"{}"}
export TORCH_HOME=${TORCH_HOME:-$HOME/dlp-lpwm-optimization/weights/torch}
OUT=results/dlp_speed_$(date +%m%d_%H%M)
mkdir -p "$OUT"
{ echo "config $CFG  steps/epoch $STEPS  override $COMMON"; echo "code $(git -C lpwm rev-parse --short HEAD)"; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; } | tee "$OUT/info.txt"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "another process is using the GPU; stop it first"; exit 1
fi
ROWS=(
  'A|{"stn_backend": "reference", "fused_composite": false}'
  'C|{"stn_backend": "triton", "fused_composite": true}'
  'C_comp|{"stn_backend": "triton", "fused_composite": true, "torch_compile": "default"}'
  'C_ro|{"stn_backend": "triton", "fused_composite": true, "torch_compile": "reduce-overhead"}'
  'C_comp_cb|{"stn_backend": "triton", "fused_composite": true, "torch_compile": "default", "cudnn_benchmark": true}'
  'C_comp_cb_tf32|{"stn_backend": "triton", "fused_composite": true, "torch_compile": "default", "cudnn_benchmark": true, "tf32_matmul": true}'
  'C_ro_cb|{"stn_backend": "triton", "fused_composite": true, "torch_compile": "reduce-overhead", "cudnn_benchmark": true}'
  'A_again|{"stn_backend": "reference", "fused_composite": false}'
)
for r in "${ROWS[@]}"; do
  name=${r%%|*}; over=${r#*|}
  python - "$CFG" "$OUT/cfg_$name.json" "$name" "$STEPS" "$over" "$COMMON" <<'EOF'
import json, sys
src, dst, name, steps, over, common = sys.argv[1:7]
c = json.load(open(src))
c.update(num_epochs=2, warmup_epoch=1, max_steps_per_epoch=int(steps), log_step_timing=True, eval_im_metrics=False,
         particle_dec_channels_last=False, torch_compile=False, cudnn_benchmark=False, cudnn_deterministic=True,
         tf32_matmul=False, run_prefix=f"_speed_{name}")
c.update(json.loads(common))
c.update(json.loads(over))
if c.get("n_kp_enc") == "all":   # same rule as models.py DLP.__init__
    total = c.get("n_kp_per_patch", 1) * (c["image_size"] // c["patch_size"]) ** 2
    c["n_kp_enc"] = min(total, c.get("n_kp_prior", total))
json.dump(c, open(dst, "w"), indent=1)
EOF
  echo "-- $name ($(date +%H:%M))"
  python lpwm/train_dlp.py -d "$OUT/cfg_$name.json" > "$OUT/train_$name.log" 2>&1 \
    || echo "$name: FAILED ($(grep -a -m1 -o 'OutOfMemoryError\|[A-Za-z]*Error' "$OUT/train_$name.log" | head -1))"
  mv -n ./*_speed_"$name" "$OUT/" 2>/dev/null   # the run directory train_dlp.py created
  line=$(grep -a '\[step-timing\] epoch=1 ' "$OUT/train_$name.log" | tail -n 1)
  med=$(echo "$line" | grep -o 'median_ms=[0-9.]*' | cut -d= -f2)
  peak=$(echo "$line" | grep -o 'peak_alloc_mb=[0-9.]*' | cut -d= -f2)
  echo "$name|${med:-FAILED}|${peak:-}" >> "$OUT/rows.txt"
  echo "   median ${med:-FAILED} ms, peak ${peak:-?} MB"
done

python - "$OUT" <<'EOF' | tee "$OUT/report.txt"
import sys, re, glob, os
out = sys.argv[1]
print(open(os.path.join(out, "info.txt")).read().strip())
rows = [l.strip().split("|") for l in open(os.path.join(out, "rows.txt")) if l.strip()]
ref = next((float(m) for n, m, _ in rows if n == "A" and m != "FAILED"), None)
print(f"\n{'run':16s} {'step ms (epoch 1 median)':>24s} {'vs A':>8s} {'peak GB':>8s}")
for n, m, p in rows:
    if m == "FAILED":
        print(f"{n:16s} {'FAILED':>24s}"); continue
    rel = f"{100 * (float(m) - ref) / ref:+.1f}%" if ref else "-"
    print(f"{n:16s} {float(m):24.1f} {rel:>8s} {float(p) / 1e3 if p else float('nan'):8.1f}")
print("\nparticles (from the model summary in each log):")
for n, *_ in rows[:1]:
    f = os.path.join(out, f"train_{n}.log")
    txt = open(f, errors="replace").read() if os.path.exists(f) else ""
    for key in ("Prior Keypoint Filtering", "Posterior Particles (Encoder)", "Posterior Particles (Decoder)", "Filtering Heuristic"):
        m = re.search(re.escape(key) + r"[^\n]*", txt)
        print("  " + (m.group(0)[:120] if m else key + ": not found"))
print("\npath checks:")
for n, *_ in rows:
    f = os.path.join(out, f"train_{n}.log")
    m = re.search(r"path check \(first step\) ok: (.*)", open(f, errors="replace").read()) if os.path.exists(f) else None
    print(f"  {n}: {m.group(1)[:200] if m else 'none'}")
EOF
echo "download $OUT/report.txt"
