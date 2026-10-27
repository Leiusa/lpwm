#!/bin/bash
# LPWM (video) multi-GPU speed: original vs optimized, 1 vs N GPUs, through the real training script
# (train_lpwm_accelerate.py, HuggingFace Accelerate DDP). Measurement only. Run from the repository root, inside a
# GPU allocation with all GPUs of one model (record the names below), nothing else running on them:
#
#   bash benchmarks/stn/run_lpwm_multigpu.sh CONFIG.json [GPU_COUNTS="1 2"] [EPOCHS=2] [STEPS=200] [OVERRIDE_JSON='{}']
#
# Rows, for each GPU count N (batch_size from the config is PER PROCESS, so the global batch is N x batch_size):
#   A     stn_backend=reference, fused_composite=false, eager, cudnn_benchmark=false   (original)
#   C     stn_backend=triton, fused_composite=true, torch_compile "default", cudnn_benchmark=true
#   C_ro  C with torch_compile "reduce-overhead" (CUDA graphs; may not work with DDP -> reported as FAILED)
# All rows: seed 0, warmup_epoch 0 (no recompilation when the warmup flag flips mid-run), log_step_timing on,
# eval_im_metrics off, tf32_matmul off, cudnn deterministic on; OVERRIDE_JSON is applied to every row
# (e.g. '{"n_kp_enc": "all"}' = no particle filtering).
# LPWM needs much more memory per GPU than single-image DLP: lower batch_size (per process) in OVERRIDE_JSON if
# needed, e.g. '{"batch_size": 2}'.
# Report: last-epoch [step-timing] of the main process (epoch 0 pays compilation and cuDNN trials), sequences/s over
# all GPUs, scaling efficiency = sequences/s(N) / (N x sequences/s(1)) per row, peak memory per GPU. Two clocks: "step"
# = GPU step after the batch is on the device (excludes data loading), "wall" = epoch wall time / steps (includes
# data loading, logging); a large gap = the input pipeline (CPU) limits. DDP sync check: [final-weights] hashes
# of all processes must be equal.
set -uo pipefail
CFG=${1:?usage: run_lpwm_multigpu.sh CONFIG.json [GPU_COUNTS] [EPOCHS] [STEPS] [OVERRIDE_JSON]}
COUNTS=${2:-"1 2"}
EPOCHS=${3:-2}
STEPS=${4:-200}
COMMON=${5:-"{}"}
OUT=results/lpwm_multigpu_$(date +%m%d_%H%M)
mkdir -p "$OUT"
{ echo "config $CFG  gpu counts $COUNTS  epochs $EPOCHS x $STEPS steps  override $COMMON"
  echo "code $(git rev-parse --short HEAD)  host $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  python -c "import torch, accelerate; print('torch', torch.__version__, '| accelerate', accelerate.__version__)"
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader; } | tee "$OUT/info.txt"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | wc -l)" != 1 ]; then
  echo "the visible GPUs are not all the same model; stop"; exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "another process is using the GPUs; stop it first"; exit 1
fi
ROWS=(
  'A|{"stn_backend": "reference", "fused_composite": false, "torch_compile": false, "cudnn_benchmark": false}'
  'C|{"stn_backend": "triton", "fused_composite": true, "torch_compile": "default", "cudnn_benchmark": true}'
  'C_ro|{"stn_backend": "triton", "fused_composite": true, "torch_compile": "reduce-overhead", "cudnn_benchmark": true}'
)
for N in $COUNTS; do
  if [ "$N" -gt "$NGPU" ]; then echo "only $NGPU GPUs visible, skipping N=$N"; continue; fi
  for r in "${ROWS[@]}"; do
    IFS='|' read -r name over <<< "$r"
    tag="${name}_${N}gpu"
    python - "$CFG" "$OUT/cfg_$tag.json" "$over" "$COMMON" "$EPOCHS" "$STEPS" "$tag" <<'EOF'
import json, sys
src, dst, over, common, epochs, steps, tag = sys.argv[1:8]
c = json.load(open(src))
c.update(json.loads(common))
c.update(json.loads(over))
c.update(seed=0, warmup_epoch=0, num_epochs=int(epochs), max_steps_per_epoch=int(steps), log_step_timing=True,
         eval_im_metrics=False, tf32_matmul=False, cudnn_deterministic=True, run_prefix=f"_mg_{tag}")
json.dump(c, open(dst, "w"), indent=1)
EOF
    echo "-- $tag ($(date +%H:%M))"
    PORT=$((29500 + RANDOM % 1000))
    accelerate launch --num_processes "$N" --num_machines 1 --mixed_precision no --dynamo_backend no \
        --main_process_port "$PORT" train_lpwm_accelerate.py -d "$OUT/cfg_$tag.json" > "$OUT/train_$tag.log" 2>&1 \
      || echo "$tag: FAILED ($(grep -a -m1 -o 'OutOfMemoryError\|[A-Za-z]*Error' "$OUT/train_$tag.log" | head -1))"
    mv -n ./*_mg_"$tag" "$OUT/" 2>/dev/null   # the run directory the training script created
    grep -a "path check" "$OUT/train_$tag.log" | head -"$N"
    grep -a "\[step-timing\]" "$OUT/train_$tag.log" | tail -1
    hashes=$(grep -a "\[final-weights\]" "$OUT/train_$tag.log" | grep -o 'sha256=[0-9a-f]*' | sort -u)
    nh=$(grep -a -c "\[final-weights\]" "$OUT/train_$tag.log")
    if [ "$nh" = "$N" ] && [ "$(echo "$hashes" | wc -l | tr -d " ")" = 1 ]; then
      echo "$tag: final weights identical on all $N processes ($hashes)" | tee -a "$OUT/sync_check.txt"
    else
      echo "$tag: SYNC CHECK FAILED: $nh hash lines, distinct: $(echo $hashes)" | tee -a "$OUT/sync_check.txt"
    fi
  done
done

python - "$OUT" "$COUNTS" <<'EOF' | tee "$OUT/summary.txt"
import os, re, sys
out, counts = sys.argv[1], sys.argv[2].split()
def last_timing(tag):
    p = os.path.join(out, f"train_{tag}.log")
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p, errors="replace") if "[step-timing]" in l]
    if not lines:
        return None
    return {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.]+)", lines[-1])}
print("last epoch, main process (sequences/s over all GPUs; efficiency = seq/s(N) / (N x seq/s(1)))")
print("step = GPU step without data loading; wall = epoch wall time / steps, with data loading")
print(f"{'row':8s} {'GPUs':>4s} {'step ms':>9s} {'seq/s':>9s} {'efficiency':>10s} {'wall ms':>9s} "
      f"{'wall seq/s':>10s} {'wall eff':>8s} {'peak GB/GPU':>11s} {'train loss':>10s}")
for row in ("A", "C", "C_ro"):
    base = last_timing(f"{row}_1gpu")
    for n in counts:
        t = last_timing(f"{row}_{n}gpu")
        if t is None:
            print(f"{row:8s} {n:>4s} {'FAILED / missing':>30s}"); continue
        eff = t["sequences_per_s"] / (int(n) * base["sequences_per_s"]) if base else float("nan")
        wall_ms = 1000 * t["epoch_wall_s"] / t["iters"]
        wall_ips = t["sequences_per_s"] * t["median_ms"] / wall_ms
        base_wall_ips = base["sequences_per_s"] * base["median_ms"] / (1000 * base["epoch_wall_s"] / base["iters"]) \
            if base else float("nan")
        wall_eff = wall_ips / (int(n) * base_wall_ips)
        print(f"{row:8s} {n:>4s} {t['median_ms']:9.1f} {t['sequences_per_s']:9.2f} {eff:10.2f} {wall_ms:9.1f} "
              f"{wall_ips:10.2f} {wall_eff:8.2f} {t['peak_alloc_mb'] / 1e3:11.2f} {t['train_loss']:10.4f}")
EOF
echo "report: $OUT/summary.txt (logs: $OUT/train_*.log)"
