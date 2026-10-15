#!/bin/bash
# Single-image DLP: does the recommended fast config train as well as the original? Paired real training runs
# (train_dlp.py), one GPU, sequential.
#
#   cd ~/dlp-lpwm-optimization && bash lpwm/benchmarks/stn/run_dlp_quality.sh static_bair128_vgg_lambda.json \
#       [SEEDS="1 2"] [EPOCHS=10] [STEPS=400] [OVERRIDE_JSON='{"n_kp_enc": "all"}']
#
# Per seed, two runs from identical initial weights and data order (config key seed), differing ONLY in the speed keys:
#   A  stn_backend=reference, fused_composite=false, eager, cudnn_benchmark=false            (original)
#   C  stn_backend=triton,    fused_composite=true,  torch_compile "reduce-overhead", cudnn_benchmark=true (recommended)
# Both: EPOCHS epochs x STEPS steps, eval every epoch (validation ELBO + LPIPS/PSNR/SSIM on the validation set,
# deterministic, uncompiled model), figures every epoch, log_step_timing on, tf32_matmul off, cudnn deterministic on,
# channels_last off; OVERRIDE_JSON (default: no particle filtering, Tal's setting) and everything else from the config.
# Report: compare_runs.py per seed (per-epoch step time, memory, train loss/PSNR, validation metrics) and the final-epoch
# C - A differences per seed and their mean. No pass/fail threshold is applied; look at the figures as well:
#   OUT/<run dir>/figures   (reconstructions, particles, bounding boxes, decompositions; A and C side by side)
set -uo pipefail
CFG=${1:?usage: run_dlp_quality.sh CONFIG.json [SEEDS] [EPOCHS] [STEPS] [OVERRIDE_JSON]}
SEEDS=${2:-"1 2"}
EPOCHS=${3:-10}
STEPS=${4:-400}
COMMON=${5:-'{"n_kp_enc": "all"}'}
S=lpwm/benchmarks/stn
export TORCH_HOME=${TORCH_HOME:-$HOME/dlp-lpwm-optimization/weights/torch}
OUT=results/dlp_quality_$(date +%m%d_%H%M)
mkdir -p "$OUT"
{ echo "config $CFG  seeds $SEEDS  epochs $EPOCHS x $STEPS steps  override $COMMON"; echo "code $(git -C lpwm rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; } | tee "$OUT/info.txt"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "another process is using the GPU; stop it first"; exit 1
fi
for SEED in $SEEDS; do
  for V in A C; do
    python - "$CFG" "$OUT/cfg_s${SEED}_$V.json" "$SEED" "$V" "$EPOCHS" "$STEPS" "$COMMON" <<'EOF'
import json, sys
src, dst, seed, v, epochs, steps, common = sys.argv[1:8]
c = json.load(open(src))
c.update(json.loads(common))
c.update(seed=int(seed), num_epochs=int(epochs), max_steps_per_epoch=int(steps), eval_epoch_freq=1, eval_im_metrics=True,
         log_step_timing=True, tf32_matmul=False, cudnn_deterministic=True, particle_dec_channels_last=False,
         run_prefix=f"_q_s{seed}_{v}")
if v == "A":
    c.update(stn_backend="reference", fused_composite=False, torch_compile=False, cudnn_benchmark=False)
else:
    c.update(stn_backend="triton", fused_composite=True, torch_compile="reduce-overhead", cudnn_benchmark=True)
if c.get("n_kp_enc") == "all":   # same rule as models.py DLP.__init__
    total = c.get("n_kp_per_patch", 1) * (c["image_size"] // c["patch_size"]) ** 2
    c["n_kp_enc"] = min(total, c.get("n_kp_prior", total))
if c.get("warmup_epoch", 0) >= c["num_epochs"]:
    sys.exit(f"warmup_epoch {c['warmup_epoch']} >= num_epochs {c['num_epochs']}: raise EPOCHS")
json.dump(c, open(dst, "w"), indent=1)
print(f"   seed {seed} {v}: warmup_epoch {c.get('warmup_epoch')}, n_kp_enc {c.get('n_kp_enc')}, batch {c['batch_size']}")
EOF
    [ -f "$OUT/cfg_s${SEED}_$V.json" ] || exit 1
    echo "-- seed $SEED $V ($(date +%H:%M))"
    python lpwm/train_dlp.py -d "$OUT/cfg_s${SEED}_$V.json" > "$OUT/train_s${SEED}_$V.log" 2>&1 \
      || echo "seed $SEED $V: FAILED ($(grep -a -m1 -o 'OutOfMemoryError\|[A-Za-z]*Error' "$OUT/train_s${SEED}_$V.log" | head -1))"
    mv -n ./*_q_s"${SEED}"_"$V" "$OUT/" 2>/dev/null   # the run directory train_dlp.py created
    grep -a "path check" "$OUT/train_s${SEED}_$V.log" | head -1
    grep -a "init_weights_sha256" "$OUT/train_s${SEED}_$V.log" | head -1 | grep -o "init_weights_sha256: [0-9a-f]*"
  done
  python $S/compare_runs.py "$OUT/train_s${SEED}_A.log" "$OUT/train_s${SEED}_C.log" --labels A,C > "$OUT/compare_s${SEED}.txt" 2>&1
done

python - "$OUT" $SEEDS <<'EOF' > "$OUT/summary.txt"
import os, sys
sys.path.insert(0, "lpwm/benchmarks/stn")
from compare_runs import parse
out, seeds = sys.argv[1], sys.argv[2:]
cols = ("step_ms", "epoch_wall_s", "peak_gb", "train_loss", "train_psnr", "val_psnr", "val_ssim", "val_lpips")
print("final epoch, C - A per seed (val_lpips, train_loss, step_ms, epoch_wall_s, peak_gb: lower is better)")
print(f"{'seed':>4s} " + " ".join(f"{c:>13s}" for c in cols))
diffs = {c: [] for c in cols}
for s in seeds:
    runs = [parse(os.path.join(out, f"train_s{s}_{v}.log"))[0] for v in ("A", "C")]
    if not all(runs):
        print(f"{s:>4s} incomplete"); continue
    e = min(max(runs[0]), max(runs[1]))
    row = []
    for c in cols:
        a, b = runs[0].get(e, {}).get(c), runs[1].get(e, {}).get(c)
        if a is None or b is None:
            row.append(f"{'-':>13s}"); continue
        diffs[c].append(b - a)
        row.append(f"{b - a:+13.4f}")
    print(f"{s:>4s} " + " ".join(row) + f"   (epoch {e}; A {runs[0][e].get('val_psnr', float('nan')):.3f} dB val PSNR)")
print(f"{'mean':>4s} " + " ".join(f"{sum(v) / len(v):+13.4f}" if v else f"{'-':>13s}" for v in diffs.values()))
EOF
{ cat "$OUT/info.txt"; echo; cat "$OUT/summary.txt"
  for SEED in $SEEDS; do echo; echo "=================== seed $SEED"; cat "$OUT/compare_s${SEED}.txt"; done
} > "$OUT/report.txt"
cat "$OUT/report.txt"
echo "download $OUT/report.txt and look at the figures in $OUT/*_q_s*_A/figures vs $OUT/*_q_s*_C/figures"
