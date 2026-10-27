# Multi-GPU DLP Training with the Speedup: How to Run and Verify It

Author: Junhong Lin (rpadcmu@gmail.com), with advisor Tal Daniel. Branch: `release/dlp-multigpu` (based on
`release/dlp-speedup`, commit `ec9ecbf`). Last updated 2026-10-27.

This guide covers single-image DLP training on several GPUs of one machine with `train_dlp_accelerate.py`
(HuggingFace Accelerate, PyTorch DDP), using the opt-in speed keys described in `DLP_SPEEDUP_GUIDE.md`. As there,
every change is opt-in: with the new config keys absent, the script behaves exactly as before.

---

## 0. TL;DR

- **The optimized code works on multiple GPUs.** On 2 GPUs, the original configuration (A), the recommended
  configuration with torch.compile `default` (C) and with `reduce-overhead` (C_ro) all train; every process takes the
  requested fast path, and the final weights are bit-identical on both GPUs (gradients are synchronized).
- **Scaling is close to linear:** efficiency 0.99 (A) and 0.95 (C, C_ro) on 2 GPUs. With 2 GPUs, C gives
  89.6 images/s vs 21.9 for the original on 1 GPU (4.1×).
- **Recommended config** (same keys as single-GPU; on multi-GPU we suggest compile `default`, see 4.1):

```json
"stn_backend": "triton",
"fused_composite": true,
"torch_compile": "default",
"cudnn_benchmark": true
```

```bash
accelerate launch --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no \
    train_dlp_accelerate.py -d configs/YOUR_CONFIG.json
```

---

## 1. What is in this branch (on top of `release/dlp-speedup`)

| Path | What changed |
|---|---|
| `train_dlp_accelerate.py` | The opt-in keys of `train_dlp.py` (`seed`, `stn_backend`, `fused_composite`, `torch_compile`, `tf32_matmul`, `cudnn_benchmark`, `cudnn_deterministic`, `max_steps_per_epoch`, `log_step_timing`) plus `num_workers`; first-step path check on every process; per-epoch step timing with images/s over all GPUs; final-weight hash per process (DDP sync check); respects a `CUDA_VISIBLE_DEVICES` already set by a scheduler |
| `benchmarks/stn/run_dlp_multigpu.sh` | Original vs optimized at 1 and N GPUs through the real training script: scaling efficiency, data-loading overhead, sync check |

No model code changed: `models.py`, `modules/`, `lpwm_stn/` and `utils/` are the same as on `release/dlp-speedup`.

---

## 2. How to launch

### 2.1 Command
```bash
accelerate launch --num_processes N --num_machines 1 --mixed_precision no --dynamo_backend no \
    train_dlp_accelerate.py -d configs/YOUR_CONFIG.json
```
- `--dynamo_backend no`: compilation is controlled by the config key `torch_compile` (the script compiles the
  DDP-wrapped model itself, with the Triton ops kept out of the graph, as in `train_dlp.py`). Letting Accelerate
  compile as well would compile twice and without those boundaries.
- `--mixed_precision no`: the Triton kernels are fp32 only (`DLP_SPEEDUP_GUIDE.md`, 3.2); AMP would make them fall
  back or raise.
- `--main_process_port`: set a free port if another Accelerate job runs on the same machine.

### 2.2 Which GPUs
- If `CUDA_VISIBLE_DEVICES` is already set (e.g. by Slurm), the script uses it. Otherwise it sets it from the `GPUS`
  list at the top of `train_dlp_accelerate.py` (as before).
- Use GPUs of **one model** only; DDP runs at the speed of the slowest GPU.

### 2.3 Batch size, learning rate, data loading
- `batch_size` in the config is **per process**: the global batch is `N × batch_size`. The script prints both.
- The learning rate is **not** rescaled for the larger global batch (unchanged behaviour). Decide whether to adjust it.
- `num_workers` (new key, default 4 = previous value) is the number of DataLoader workers **per process**. Each process
  also needs one CPU core of its own, so request at least `N × (num_workers + 1)` CPU cores.

### 2.4 Slurm example
```bash
salloc --gres=gpu:2 -c 8 --mem=28G --time=2:00:00     # 2 processes × (2 workers + 1) = 6 cores, plus margin
# inside the allocation (from the repository root):
accelerate launch --num_processes 2 --num_machines 1 --mixed_precision no --dynamo_backend no \
    train_dlp_accelerate.py -d configs/YOUR_CONFIG.json
```
Slurm grants the job only when one node has the GPUs, the CPU cores **and** the memory free at the same time; a node
that shows idle GPUs can still be unable to host the job. If the job stays pending, lower `-c` / `--mem` (and
`num_workers` accordingly) or pick another node.

---

## 3. Verifying a multi-GPU run

### 3.1 In every run (set `log_step_timing: true`)
1. **Path check, once per process:**
   `path check (first step) ok on process i: backend=triton, fused_composite_calls=1, ...` — one line for each
   process. A process that did not take the requested path raises an error at the first step.
2. **`[step-timing]` per epoch** (main process): median step ms, `images_per_s` over all GPUs, peak memory,
   `epoch_wall_s`.
3. **Sync check at the end:** `[final-weights] process i/N sha256=...` — one line per process; **all hashes must be
   equal**. DDP keeps the replicas identical only if the gradients were all-reduced at every step, so equal hashes
   after training show that synchronization worked (including with Triton ops, torch.compile and CUDA graphs).
   The hash is printed only when `log_step_timing` is on.

### 3.2 Scaling measurement — `benchmarks/stn/run_dlp_multigpu.sh`
```bash
bash benchmarks/stn/run_dlp_multigpu.sh configs/YOUR_CONFIG.json "1 2" 2 200 '{"num_workers": 2}'
#                                        config                  GPU counts, epochs, steps/epoch, override
```
- Runs rows A (original), C (recommended, compile `default`) and C_ro (`reduce-overhead`) at each GPU count, short real
  trainings (seed 0, `warmup_epoch` 0); reports the last epoch (the first one pays compilation and cuDNN trials).
- Stops if the visible GPUs are of different models or if another process uses them.
- Output `results/dlp_multigpu_*/`: `summary.txt` (table below), `sync_check.txt` (one line per run: "final weights
  identical on all N processes" or "SYNC CHECK FAILED"), the logs and configs of every run.
- Columns: **step** = GPU time of a training step after the batch is on the device; **wall** = epoch wall time / steps,
  which includes data loading and logging. Efficiency = images/s(N) / (N × images/s(1)). If wall efficiency is much
  lower than step efficiency, the input pipeline (CPU) is the limit, not the GPUs.

---

## 4. Results

**Setup:** 2× Quadro RTX 6000 (Turing, 24 GB; no TF32 tensor cores), one node, torch 2.6.0+cu126, Accelerate 1.15.0;
single-image BAIR config `static_bair128_vgg.json` (128×128, 90 particles, batch 16 per GPU; not in this repository),
override `{"num_workers": 2}`, 2 epochs × 200 steps, last epoch, one run per row. Code: the two commits of this
branch (tested as `df1ee7d` on `lambda-sync`, with identical model code).

| Row | GPUs | Step (ms) | Images/s | Efficiency | Wall (ms) | Wall efficiency | Peak memory / GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (original) | 1 | 730.6 | 21.9 | — | 733.0 | — | 11.16 GB |
| A (original) | 2 | 737.4 | 43.4 | **0.99** | 739.9 | **0.99** | 11.20 GB |
| C (recommended) | 1 | 339.9 | 47.1 | — | 341.1 | — | 6.79 GB |
| C (recommended) | 2 | 357.3 | 89.6 | **0.95** | 361.4 | **0.94** | 6.84 GB |
| C_ro | 1 | 336.6 | 47.5 | — | 380.9 | — | 6.64 GB |
| C_ro | 2 | 353.2 | 90.6 | **0.95** | 409.3 | **0.93** | 7.90 GB |

- **Functional:** all 6 runs completed; path check correct on every process; final weights identical on both GPUs for
  A, C and C_ro.
- **C vs A:** 2.15× on 1 GPU, the same on 2 GPUs; peak memory −39%.
- C's efficiency (0.95) is a little lower than A's (0.99) because its step is shorter, so the gradient all-reduce takes
  a larger share of it. Step and wall efficiencies agree: the input pipeline was not the limit here.

### 4.1 `reduce-overhead` vs `default` on multi-GPU
On this GPU C_ro's step is the same as C's, but its wall time is higher: with CUDA graphs, the last batch of every
epoch (the plotted one, which requests `return_alpha_masks`) takes a different path and the graphs are recorded again,
costing 8–18 s per epoch. On a GPU where the step is launch-bound (e.g. single-image DLP on GH200, where
`reduce-overhead` was clearly faster), `reduce-overhead` may still pay off; measure with 3.2 before choosing it.

---

## 5. Known issues and limits

- **CUDA-graph re-recording at the end of each epoch** with `reduce-overhead` (4.1). Not a multi-GPU bug; also present
  on one GPU.
- **Process 1 also creates an (empty) run directory** (`prepare_logdir` runs on every process). Logs, plots and
  checkpoints are written by the main process only; the extra directory can be deleted.
- **Measured on one GPU model only** (Quadro RTX 6000, no TF32): absolute times on Ampere or newer GPUs will differ;
  the efficiency on 2 GPUs of one node is the transferable part. More than 2 GPUs and multi-node were not tested.
- **Short runs only** (200 steps per epoch): no long training / quality comparison on multi-GPU.
- **LPWM** (`train_lpwm_accelerate.py`) does not have the opt-in keys yet; this guide is for single-image DLP.
