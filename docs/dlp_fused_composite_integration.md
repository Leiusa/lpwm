# DLP fused paste + composite: opt-in integration and measurements (stage version)

Scope: **static DLP** (`timestep_horizon == 1`, single GPU, `train_dlp.py`). The LPWM dynamics model was not
evaluated. Everything here is opt-in and off by default. This document records what was built, how it was
checked, what was measured, and what is *not* established.

## What was added

| Piece | Where |
|---|---|
| Autograd wrapper around the standalone fused forward/backward kernels (kernel math unchanged) | `lpwm_stn/composite_autograd.py` |
| `fused_composite=False` constructor argument on `DLPDecoder` and `DLP`; a branch at the top of `DLPDecoder.decode_objects` | `modules/modules.py`, `models.py` |
| Optional config keys in the training entry: `seed`, `stn_backend`, `fused_composite`, `log_step_timing` | `train_dlp.py` |
| Measurement and experiment harness | `benchmarks/stn/` (`attribute_memory.py`, `fusion_ceiling.py`, `fused_model_compare.py`, `train_dlp_compare.py`, `prepare_bair_subset.py`, plus edits to `bench_e2e.py`, `build_model.py`) |
| Same-machine Triton-vs-reference check for crop/paste | `tests/stn/live_check.py` |
| Tests | `tests/composite/test_composite_autograd.py`, `tests/composite/test_decoder_fused_integration.py`, `benchmarks/stn/test_harness.py` |

### Behaviour

* Off (default): the decoder runs exactly the previous code path. `state_dict` is unchanged (no new parameters or
  buffers), so old checkpoints load with `strict=True`.
* On: `paste -> alpha/depth composite -> reduce` runs in one fused kernel and the per-particle
  `[B, K, 4, H, W]` canvas is never formed. Only the five inputs are saved for backward.
* **Unsupported input raises `ValueError` with the reason; there is no silent fallback.** Supported: fp32, contiguous,
  CUDA, `dec_objects [B,K,4,p,p]` with square patches, `translation is None`. AMP/fp16 is therefore rejected.
* Double backward (`create_graph=True`) raises. The `dec_objects` gradient is accumulated with atomics and is not
  bit-reproducible run to run (nor is torch's own `grid_sampler` backward).
* `train_dlp.py` logs the actual path at startup and checks the first step: if the config asks for
  `fused_composite`/`stn_backend` and the step did not use it, the run stops with a `RuntimeError`.
  `decoder_module.fused_composite_calls` counts fused executions.

### Enabling it

```json
{"stn_backend": "triton", "fused_composite": true, "seed": 0, "log_step_timing": true}
```

## Checks performed (and their limits)

Two lines are kept apart: (A) does the wrapper/integration introduce new error, (B) how far are the fused and the
old Triton chain from the original PyTorch reference. No pass/fail threshold was derived from any error seen here.

* **A. Wiring (asserted).** Wrapper vs the standalone kernels: forward outputs and the four non-atomic gradients
  bit-identical over 12 fixture cases x alpha_masks True/False x cotangent scenarios; the atomic `dec_objects`
  gradient agrees within a bound fixed in advance (1e-6 relative L2; observed 6.7e-8, the standalone kernel's own
  run-to-run spread is 6.5e-8). Decoder integration: 6/6, existing alpha_masks API tests 9/9. Switch off vs pure
  `91fcae1`: loss, reconstruction and `dec_objects` bit-identical; gradients differ only at the run-to-run level.
  A wiring test found and fixed one real defect (`@once_differentiable` silently dropped second-order terms).
* **B. Total difference vs the original reference (reported only).** Whole model, same weights/input/random state:
  fused vs Triton chain: loss rel. diff <= 1.5e-7, reconstruction rel. L2 ~2e-8, gradient rel. L2 2.5e-7 to 4.7e-7 (B=16, B=1);
  Triton chain vs reference (already present before this work): gradient rel. L2 2e-4 to 5e-4. No frozen acceptance
  rule applies to these inputs (element-wise envelopes exist only for the 47 fixture combinations), so no verdict
  is declared.
* **Known failures, kept as they are.** `tests/stn/live_check.py` (Triton crop/paste vs reference, same GPU,
  32 cases): 30 pass, 2 fail with the original thresholds unchanged: `bair_prior/crop.scale` (`grad[kp]`,
  `grad[z_scale]`) and `obj3d128_prior/crop.scale` (`grad[kp]`). For the first, the error is concentrated in one
  particle whose sampling coordinate lies ~3e-6 px from a bilinear kink; this supports, but does not prove, that
  it is kink sensitivity rather than a kernel defect. The forward composite kernel also still has an unaddressed
  flush-to-zero difference in `tl.math.div_rn`.

## Measurements (NVIDIA RTX A6000, comparisons always paired inside one Slurm allocation)

Paths: **reference** = PyTorch STN, **triton** = Triton crop/paste, **fused** = Triton crop + fused composite.

| Setting | reference | triton | fused |
|---|---:|---:|---:|
| Synthetic static DLP, BAIR-derived, B=1: step ms / peak MB | 42.9 / 722 | 41.5 / 639 | 40.8 / 563 |
| Synthetic static DLP, BAIR-derived, B=16: step ms / peak MB | 289.8 / 9,528 | 242.6 / 8,239 | 229.8 / 7,817 |
| Standalone composite chain, B=16 (patches -> outputs + grads): ms / whole-chain peak MB | (59.6 / 1,987) | 18.06 / 1,798 | 3.98 / 12 |
| `train_dlp.py`, `shapes` (synthetic, mse), B=32: CUDA-event step ms / peak MB | 63.6 / 2,007 | 59.5 / 1,876 | 58.6 / 1,824 |
| **`train_dlp.py`, real BAIR frames, vgg loss, B=16: CUDA-event step ms** | 331.4 | 284.2 | 271.0 |
| BAIR: wall per iteration ms (epoch wall / 750) | 332.4 | 285.5 | 272.3 |
| BAIR: throughput images/s | 48.1 | 56.0 | 58.8 |
| BAIR: peak allocated MB | 10,760 | 9,468 | 8,337 |

CUDA-event time covers forward + backward + update only; wall time also includes data loading and logging (about
1.0 to 1.3 ms more per step here). At B=1 the step is CPU-bound and the gain is small.

### BAIR short training (3 epochs, 2 passes in opposite order, same seed/init/data order)

* Validation loss (final epoch, two passes): reference 12.853 / 13.036, triton 12.975 / 12.926, fused 12.933 / 13.000.
  Whole validation set (1,200 images) PSNR: 22.15 to 22.40 dB across all six runs; the spread within one path
  (up to 0.14 dB) exceeds the spread between paths.
* **Observation kept open:** the training loss of both Triton paths is above the reference by +1.0% to +2.2%
  (two-pass means, epoch 0/1/2: reference 23.62/14.66/12.79, triton 23.95/14.85/12.92, fused 24.14/14.84/12.92).
  Only two runs per path exist and the reference differs from itself by up to 1.2% between passes; validation
  loss and validation MSE show no consistent direction. It has not been diagnosed.
* Reconstruction grids (target, reference, triton, fused) show no colour shift or corruption on any path.

## Not established

* Convergence or final quality equivalence: only 3 epochs, one seed, 400 training episodes.
* Behaviour at other batch sizes, resolutions or particle counts; dynamics (LPWM) models; AMP.
* Why the fusion memory saving differs between experiments (-422 MB synthetic vs -1,131 MB in the BAIR run): data,
  training state and tensor lifetimes all differ, so it is not attributed to the loss.
* The result is a "fixed prefix subset with an internal validation split", not a random sample and not the official
  validation set.

## BAIR data used (personal directory only)

* Source: Hugging Face `taldatech/bair_256`, file `bair_256_ours.tar.gz` (126,620,002,163 bytes, CC-BY-SA-4.0),
  revision `1ba345924da397c0017e63ba916bd57d82455ffd`. `prepare_bair_subset.py data` streams the archive prefix,
  keeps the first 400 complete train episodes as `train/` and the next 40 as `val/`, reads at most a byte cap
  (1.26 GB was read) and stops instead of continuing if the subset cannot be cut. The manifest holds per-file
  sha256 and the aggregate `64917d7eeb7da3826e9d1524b6c3f2d70ec0ea0fb5f79d4722ba7a8fc086f431`.
* `datasets/bair_ds.py:BAIRImage` uses `str(int(folder))`, so zero-padded archive folders (`05498`) are not found;
  `prepare_bair_subset.py unpad` renames them (episode numbers unchanged) and records this in the manifest.
* vgg loss weights (`prepare_bair_subset.py weights`): torchvision `vgg16-397923af.pth` (553,433,881 B) and the
  LPIPS head `vgg.pth` (7,289 B, md5 `d507d7349b931f0638a25a48a722f98a`).
* Static config: `configs/bair.json` without the dynamics-only keys, `timestep_horizon=1`, `batch_size=16`,
  `eval_im_metrics=false`, `root` set to the personal subset; every loss setting is the original one.

```python
import json
b, s = json.load(open("configs/bair.json")), json.load(open("configs/shapes.json"))
cfg = {k: v for k, v in b.items() if k in s}     # drops the 18 keys only the dynamics model reads
cfg.update(timestep_horizon=1, batch_size=16, eval_im_metrics=False, root="<personal subset>/")
```

## Reproducing

```bash
python benchmarks/stn/prepare_bair_subset.py data    --out <dir> --train-episodes 400 --val-episodes 40
python benchmarks/stn/prepare_bair_subset.py unpad   --out <dir>
python benchmarks/stn/prepare_bair_subset.py weights --torch-home <dir>/torch --lpips-dir <dir>/eval/lpips
TORCH_HOME=<dir>/torch python benchmarks/stn/train_dlp_compare.py --repo-root . --config <static bair cfg> \
    --epochs 3 --seed 0 --smoke-first --lpips-head <dir>/eval/lpips/vgg.pth --out <run dir>
python benchmarks/stn/fused_model_compare.py numerics|bench|smoke ...      # whole-model numerics / timing / smoke
python tests/composite/test_composite_autograd.py; python tests/composite/test_decoder_fused_integration.py
```

Environment additions on Grogu (declared in `requirements.txt`): `imageio`, `opencv-python-headless`, `h5py`,
`lazy-loader`, `piqa`, `scikit-image`, `scipy`, `tifffile`; torch 2.6.0+cu126, triton 3.2.0 and numpy 1.26.4 unchanged.
Raw results, logs, per-run manifests and script hashes were written outside the repository under
`/grogu/user/junhong3/lpwm-work/experiments/fused-integration/` (`run-3959243/`, `train-compare-3959260/`,
`bair-compare-3959347/`) and `.../data/`.
