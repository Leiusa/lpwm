# DLP Training Speedup: What Changed and How to Apply It to Your DLP Code

Author: Junhong Lin (rpadcmu@gmail.com), with advisor Tal Daniel. Branch: `release/dlp-speedup` (based on
`research/stn-composite-fusion`, commit `00accf5`). Last updated 2026-10-07.

This guide is written so that the optimizations can be ported into **a different version of the DLP code** whose key
parts (the STN crop in the particle encoders, the STN paste in the decoder, and the depth-weighted alpha compositing)
are the same as in this repository. Every change is opt-in: with the new config keys absent, the code behaves exactly
as before.

---

## 0. TL;DR

**Recommended config for single-image DLP training** (no precision-reducing change):

```json
"stn_backend": "triton",
"fused_composite": true,
"torch_compile": "reduce-overhead",
"cudnn_benchmark": true
```

**Measured speed** (GH200, single-image DLP, BAIR 128×128, batch 16, 90 particles, `train_dlp.py`, median step time of
epoch 1 after 300 warm-up steps; one run per row; the original repeated twice differs by 1%):

| Configuration | Step (ms) | vs. original | Peak memory |
|---|---:|---:|---:|
| Original (reference STN, eager) | 102.0 | — | 10.8 GB |
| + Triton STN + fused composite | 96.0 | −5.9% | 8.7 GB |
| + torch.compile `default` | 67.1 | −34% | 7.8 GB |
| + torch.compile `reduce-overhead` | 55.7 | −45% | 7.6 GB |
| + `reduce-overhead` + `cudnn_benchmark` (**recommended**) | **49.9** | **−51%** | **7.4 GB** |
| + `default` + `cudnn_benchmark` + TF32 matmul | 61.8 | −39% | 6.8 GB (TF32 adds nothing here) |

**Correctness** at 256 particles without filtering (Section 6): the Triton kernels are as close to a float64 ground
truth as the original fp32 code is, in every stress scenario (heavy overlap, border, sub-pixel and full-image glimpses,
depth ties, all particles off). In the whole model, the gradient difference between optimized and original (median
relative 1.6e-4) is 6× smaller than what a 1e-6 input perturbation causes, and ~100× smaller than what PyTorch's
default TF32 convolutions (which the original code already trains with) cause.

**Important caveats** (Section 8): speedups are GPU- and config-specific (measure on your GPU with Section 6.3); the
256-particle (no filtering) speed is being measured separately; a long-run quality comparison at 256 particles has not
been done yet.

---

## 1. What is in this branch

| Path | What it is | Needed for porting? |
|---|---|---|
| `lpwm_stn/` | Self-contained package: reference STN (the original PyTorch code, unchanged numerics), Triton crop/paste kernels, fused paste+composite kernels, backend registry | **Yes, copy as is** |
| `utils/compile_utils.py` | `compile_for_training()` (torch.compile with the Triton ops kept out of the graph) and `compile_mode_from_config()` | **Yes, copy** (adjust 1 import, Section 4.6) |
| `modules/modules.py` | 3 call sites switched to `lpwm_stn` (2× crop, 1× paste), fused-composite branch in `DLPDecoder`, `return_alpha_masks` plumbing, optional channels-last in `ObjectDecoderCNN` | Port the edits (Sections 4.2–4.5) |
| `models.py` | `DLP.__init__` gets `fused_composite` / `particle_dec_channels_last`; `return_alpha_masks` passed through | Port the edits (Section 4.4–4.5) |
| `utils/util_func.py` | `spatial_transform`, `affine_grid_sample`, `create_masks_*` now re-exported from `lpwm_stn` | Optional (Section 4.7) |
| `train_dlp.py`, `train_lpwm.py` | Opt-in config keys, compile wrapper, cuDNN switches, first-step path check, step timing | Port the edits (Section 4.6) |
| `benchmarks/stn/dlp_kernel_stress_check.py` | Kernel correctness vs float64 ground truth (+ whole model) | Run after porting (6.1) |
| `benchmarks/stn/dlp_model_grad_diag.py` | Whole-model gradient consistency vs rounding-level yardsticks | Run after porting (6.2) |
| `benchmarks/stn/run_dlp_speed_matrix.sh` | Speed of each configuration through the real training script | Run after porting (6.3) |
| `benchmarks/stn/build_model.py` | Builds `DLP(...)` from a config by matching constructor argument names (used by the two Python checks) | Copy with the checks |

---

## 2. The optimizations, one by one

### 2.1 Triton STN crop (encoder glimpses) — `stn_backend: "triton"`
- **Where:** `ParticleAttributeEncoder.forward` and `ParticleFeaturesEncoder.forward` crop one glimpse per particle.
- **Original:** `x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)` materializes a full copy of the image **per particle**
  (`[bs*n_kp, ch, H, W]`, the largest allocation in the encoder), then `affine_grid` + `grid_sample`.
- **New:** one Triton kernel computes the affine coordinates and bilinear samples directly from the shared image, with
  a hand-written backward (gradients w.r.t. image, centers and scales). No repeated image, no dense grid.
- **Precision:** fp32, same math; differences are rounding-level (Section 6).
- **Fallback:** unsupported inputs silently use the original PyTorch code (see 3.2).

### 2.2 Triton STN paste (decoder) — `stn_backend: "triton"`
- **Where:** `DLPDecoder.translate_patches` places each decoded glimpse on the canvas (inverse transform).
- **New:** Triton forward/backward, no dense grid. Used when `fused_composite` is off.

### 2.3 Fused paste + depth-weighted alpha composite — `fused_composite: true`
- **Where:** `DLPDecoder.decode_objects` (paste every particle's RGBA glimpse to full resolution, then
  `get_objects_alpha_rgb_with_depth`).
- **Original:** forms a `[bs, n_kp, 4, H, W]` canvas stack (one full-resolution canvas per particle), then reduces it.
- **New:** one kernel pastes and composites while reducing over particles; the per-particle canvas is never formed.
  Backward is a separate Triton kernel. This is where most of the memory saving comes from, and it grows with the
  number of particles.
- **Unsupported input raises `ValueError`** (no silent fallback), so a run can never silently take the wrong path.

### 2.4 `return_alpha_masks` (memory, always safe)
The per-particle alpha stack (`alpha_masks`, `[bs, n_kp, 1, H, W]`) is only used for plotting; it is never read by the
loss. The training script now requests it only for the batch that is plotted. All other values are bit-identical.

### 2.5 torch.compile — `torch_compile: "reduce-overhead"` (or `"default"`)
- Compiles the **training forward** only; the uncompiled model is kept for validation, plotting and checkpoints (same
  parameters, same `state_dict` keys).
- The Triton ops are wrapped with `torch.compiler.disable`; without this, Dynamo traces into the Triton launchers with
  symbolic strides and crashes. These boundaries cause graph breaks (about 19 in LPWM), which cost no measurable time.
- `reduce-overhead` adds CUDA graphs. For single-image DLP (short steps) it is clearly faster than `default`
  (55.7 vs 67.1 ms); for LPWM (long steps) the two are equal.
- Cost: each compilation takes 1.5–3 minutes, and a recompilation happens once when the `warmup` flag flips. Negligible
  for long runs, noticeable in short tests.
- Precision: fp32; fusion changes rounding order only (max |diff| ≈ 5e-4 on [0, 1] pixel outputs).

### 2.6 cuDNN algorithm selection — `cudnn_benchmark: true` (keep `cudnn_deterministic: true`)
- The original scripts set `torch.backends.cudnn.benchmark = False` and `deterministic = True` at import time, so cuDNN
  picks convolution algorithms by heuristic. With `benchmark=True` it times the candidates for each shape once and keeps
  the fastest.
- Gain: −9 to −10% on top of compile for single-image DLP; −13% in LPWM, almost all in **conv backward** of the
  glimpse CNNs (in LPWM the attribute-encoder backward went 75 → 42 ms).
- Precision: unchanged (convolutions are TF32 by PyTorch default either way). Cost: bit-exact run-to-run
  reproducibility (the selected algorithm can differ between runs). `deterministic=False` added only ~2 ms more, so we
  recommend keeping it `true`.
- Memory: about +7 GB during the first epoch only (algorithm trials), then identical.

### 2.7 Not recommended (measured)
| Option | Result | Why not |
|---|---|---|
| `tf32_matmul: true` | Single-image DLP: no gain (61.3 → 61.8 ms). LPWM: −36% (transformers) | The only precision-reducing option; useless for DLP |
| `particle_dec_channels_last: true` | Faster on RTX 3090, **slower** on GH200 (33.5 → 38.5 ms) due to layout copies | GPU-dependent; changes conv algorithms |
| LPWM: `reduce-overhead` + `cudnn_benchmark` | Out of memory on a 94.5 GB GH200 (cause unknown; not workspace size) | Use `default` + `cudnn_benchmark` for LPWM. Single-image DLP is fine |

---

## 3. Requirements and limits

### 3.1 Software and hardware
- NVIDIA GPU with CUDA; tested on RTX 3090 (torch 2.6.0, Triton 3.2) and GH200 (torch 2.7.1+cu128, Triton 3.3.1).
- `pip install triton` matching your torch (normally pulled in by torch). Triton ≥ 3.3 needs the `tl.constexpr(...)`
  fix that is already in `lpwm_stn/composite_backward.py`.

### 3.2 When the fast path is used
| Op | Fast path requires | Otherwise |
|---|---|---|
| `stn_crop` | CUDA, fp32 for image/centers/scales, `padding_mode="border"`, 4-D image, `kp` `[bs, n_kp, 2]`, 1–16 channels, `patch_size² ≤ 65536` | Falls back to the original PyTorch code |
| `stn_paste` | CUDA, fp32, patches `[bs, n_kp, ch, p, p]` square, 1–16 channels | Falls back to the original PyTorch code |
| `composite_fused` | CUDA, fp32, **contiguous** tensors on one device; `dec_objects [B, K, 4, p, p]` square; `z_kp [B, K, 2]`; `z_scale` `None` or `[B, K, 2]`; `obj_on [B, K]`; `z_depth [B, K, 1]`; no `translation` | **Raises `ValueError`** with the reason |

Because crop and paste fall back silently, use the first-step path check (4.6) and the logs to confirm the fast path is
really taken. bf16/fp16 inputs (AMP) are not supported by the kernels and would fall back / raise.

---

## 4. Porting into your DLP code, step by step

**Before you start, check that your version computes the same thing (4.1).** The Triton kernels implement the exact
semantics of `lpwm_stn/reference.py` and `lpwm_stn/composite_reference.py`. If your `spatial_transform` or compositing
differs (different epsilon, clamping, `align_corners`, padding mode, an extra term), the kernels will compute something
different from your code. Section 6 detects that, but check first.

### 4.1 Semantics to compare against your code
**`spatial_transform`** (`lpwm_stn/reference.py`, identical to the original `utils/util_func.py`):
```python
theta = torch.zeros(2, 3, device=image.device).repeat(image.shape[0], 1, 1)
theta[:, 0, 0] = z_scale[:, 1] if not inverse else 1 / (z_scale[:, 1] + eps)        # eps = 1e-9
theta[:, 1, 1] = z_scale[:, 0] if not inverse else 1 / (z_scale[:, 0] + eps)
theta[:, 0, -1] = z_pos[:, 1] if not inverse else - z_pos[:, 1] / (z_scale[:, 1] + eps)
theta[:, 1, -1] = z_pos[:, 0] if not inverse else - z_pos[:, 0] / (z_scale[:, 0] + eps)
grid = F.affine_grid(theta, out_dims, align_corners=False)
out = F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode=padding_mode)
```
- Crop: `inverse=False`, `padding_mode='border'`, `z_scale = sigmoid(raw_scale)` or `patch_size / img_size` if `None`.
- Paste: `inverse=True`, `padding_mode='zeros'` (the default), `z_scale = sigmoid(scale)` unless `scale_normalized`.

**Compositing** (`lpwm_stn/composite_reference.py`, identical to `get_objects_alpha_rgb_with_depth`):
```python
a_obj = obj_on[:, :, None, None, None] * a_obj
rgba_obj = a_obj * rgb_obj
importance_map = a_obj * torch.sigmoid(-z_depth[:, :, :, None, None])
importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + 1e-5)
out_rgb = (rgba_obj * importance_map).sum(dim=1)
alpha_mask = 1.0 - (importance_map * a_obj).sum(dim=1)        # background mask
```
If your compositing differs, enable only `stn_backend: "triton"` (crop/paste) and leave `fused_composite` off.

### 4.2 Copy the package
Copy `lpwm_stn/` into your repository root (next to `models.py`). It has no dependency on the rest of the code base.
Importing it registers two backends: `"reference"` (default, the original code) and `"triton"`.

### 4.3 Encoders: replace the two crop blocks
In `ParticleAttributeEncoder.forward` and `ParticleFeaturesEncoder.forward` (search for
`x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)` followed by `spatial_transform(..., inverse=False, padding_mode='border')`):

```python
# BEFORE
batch_size, _, _, img_size = x.shape
_, n_kp, _ = kp.shape
x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)
x_repeated = x_repeated.view(-1, *x.shape[1:])
if z_scale is None:
    z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
else:
    z_scale = torch.sigmoid(z_scale)   # assume unnormalized z_scale
z_pos = kp.reshape(-1, kp.shape[-1])
z_scale = z_scale.view(-1, z_scale.shape[-1])
out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False, padding_mode='border')

# AFTER
batch_size = x.shape[0]
_, n_kp, _ = kp.shape
cropped_objects = stn_crop(x, kp, self.patch_size, z_scale=z_scale, padding_mode='border')
# [batch_size * n_kp, ch, patch_size, patch_size]
```
`z_scale` is passed **unnormalized** (the sigmoid happens inside). Keep any later use of `batch_size` / `n_kp`.

At the top of `modules/modules.py`:
```python
from lpwm_stn import create_masks_fast, create_masks_with_scale, stn_crop, stn_paste
from lpwm_stn.composite_autograd import composite_fused, fused_composite_supported
```
(and remove `spatial_transform`, `create_masks_fast`, `create_masks_with_scale` from the `utils.util_func` import if
they are imported there, to avoid shadowing).

### 4.4 Decoder: replace the paste and add the fused branch
**`DLPDecoder.translate_patches`** — replace the body:
```python
def translate_patches(self, kp_batch, patches_batch, scale=None, translation=None, scale_normalized=False):
    return stn_paste(kp_batch, patches_batch, self.feature_map_size, scale=scale,
                     translation=translation, scale_normalized=scale_normalized)
```
(use whatever canvas size variable your version uses; here it is `self.feature_map_size`).

**`DLPDecoder.__init__`** — new arguments and attributes:
```python
def __init__(self, ..., fused_composite=False, particle_dec_channels_last=False):
    ...
    self.fused_composite = bool(fused_composite)
    self.fused_composite_calls = 0   # lets the training script prove which path ran
```
(`particle_dec_channels_last` is optional and not recommended; you can omit it, see 2.7.)

**New method** in `DLPDecoder`:
```python
def _decode_objects_fused(self, z_kp, z_features, obj_on, z_scale, translation, z_depth, z_ctx,
                          return_alpha_masks):
    # same glimpses as get_objects_alpha_rgb, but paste + composite run in one fused kernel
    z_kp = z_kp.contiguous()          # deterministic=True can pass views from torch.chunk
    obj_on = obj_on.contiguous()
    z_depth = z_depth.contiguous()
    if z_scale is not None:
        z_scale = z_scale.contiguous()
    dec_objects = self.particle_dec(z_features, context=z_ctx)          # [bs * n_kp, 4, p, p]
    dec_objects = dec_objects.view(-1, z_kp.shape[1], *dec_objects.shape[1:])  # [bs, n_kp, 4, p, p]
    if translation is not None:
        ok, reason = False, "translation is not None (the fused kernel has no translation input)"
    else:
        ok, reason = fused_composite_supported(dec_objects, z_kp, z_scale, obj_on, z_depth, self.feature_map_size)
    if not ok:
        raise ValueError(f"fused_composite=True but the input is unsupported: {reason}")
    alpha_masks, bg_mask, dec_objects_trans = composite_fused(
        dec_objects, z_kp, z_scale, obj_on, z_depth, self.feature_map_size,
        return_alpha_masks=return_alpha_masks)
    self.fused_composite_calls += 1
    return dec_objects, dec_objects_trans, alpha_masks, bg_mask
```
Note the return order matches the original `decode_objects`: `(dec_objects, dec_objects_trans, alpha_masks, bg_mask)`,
where `dec_objects_trans` here is the **composited RGB** `[bs, 3, H, W]` (as in the original, after compositing).
If your `particle_dec` call takes different arguments (e.g. no `context`), mirror your own `get_objects_alpha_rgb`.

**`DLPDecoder.decode_objects`** — branch at the top:
```python
def decode_objects(self, z_kp, z_features, obj_on, z_scale=None, translation=None, z_depth=None,
                   z_ctx=None, return_alpha_masks=True):
    if self.fused_composite:
        return self._decode_objects_fused(z_kp, z_features, obj_on, z_scale, translation, z_depth, z_ctx,
                                          return_alpha_masks)
    ...  # original code unchanged
```

**`DLP.__init__`** (`models.py`) — accept the flags and pass them to `DLPDecoder(...)`:
```python
def __init__(self, ..., fused_composite=False, particle_dec_channels_last=False):
    ...
    self.decoder_module = DLPDecoder(..., fused_composite=fused_composite,
                                     particle_dec_channels_last=particle_dec_channels_last)
```
The `state_dict` is unchanged: checkpoints load in both directions.

### 4.5 `return_alpha_masks` plumbing (optional, memory)
Add `return_alpha_masks=True` to `DLP.forward`, `DLP.decode_all`, `DLPDecoder.forward`, `DLPDecoder.decode_all`,
`DLPDecoder.decode_objects` and pass it down. In `get_objects_alpha_rgb_with_depth`:
```python
if return_alpha_masks:
    a_obj = importance_map * a_obj
else:
    a_obj = None
return a_obj, alpha_mask, dec_objects_trans
```
In the training loop, request masks only for the batch that is plotted:
```python
plot_this_epoch = (epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1)
for batch_idx, batch in enumerate(pbar):
    need_masks = plot_this_epoch and batch_idx == len(dataloader) - 1
    model_output = train_model(x, ..., return_alpha_masks=need_masks)
```
Anything else that reads `model_output['alpha_masks']` (validation, evaluation, plotting) keeps the default `True`.

### 4.6 Training script
Copy `utils/compile_utils.py`. It wraps `lpwm_stn.triton_backend.stn_crop/stn_paste` and
`modules.modules.composite_fused` with `torch.compiler.disable`. **If your decoder imports `composite_fused` in a
module other than `modules.modules`, change that line** (it must patch the name the decoder actually calls).

In `train_dlp.py` (see this branch's version for the complete code):
```python
import lpwm_stn
from utils.compile_utils import compile_for_training, compile_mode_from_config

# after loading the config; absent keys = original behaviour
stn_backend = config.get('stn_backend')            # 'reference' (default) or 'triton'
if stn_backend is not None:
    lpwm_stn.set_backend(stn_backend)
fused_composite = bool(config.get('fused_composite', False))
compile_mode = compile_mode_from_config(config)    # None, 'default', 'reduce-overhead'
torch.backends.cudnn.benchmark = bool(config.get('cudnn_benchmark', False))
torch.backends.cudnn.deterministic = bool(config.get('cudnn_deterministic', True))

model = DLP(..., fused_composite=fused_composite).to(device)
train_model = model if compile_mode is None else compile_for_training(model, compile_mode)
# use train_model for the training forward ONLY; keep `model` for validation, plotting, checkpoints

# in the training loop, before the forward:
if compile_mode == 'reduce-overhead':
    torch.compiler.cudagraph_mark_step_begin()   # CUDA graphs: a new iteration starts
model_output = train_model(x, warmup=warmup, with_loss=True, return_alpha_masks=need_masks, ...)
```
Requirements for `reduce-overhead`: fixed batch shapes (`drop_last=True` in the DataLoader, which the original script
already uses) and the `cudagraph_mark_step_begin()` call every iteration.

**First-step path check** (strongly recommended; this branch's `train_dlp.py` has it): after the first iteration,
raise if `model.decoder_module.fused_composite_calls != int(fused_composite)`, if `lpwm_stn.get_backend_name()` differs
from the requested backend, or if the cuDNN flags differ from the config. The script prints a line like
`path check (first step) ok: backend=triton, fused_composite_calls=1, ..., cudnn_benchmark=True` into the log.

### 4.7 `utils/util_func.py` (optional)
This branch re-exports `affine_grid_sample`, `spatial_transform`, `create_masks_fast`, `create_masks_with_scale` from
`lpwm_stn` so that every STN op goes through one backend registry. This is not needed for the speedup. If you skip it,
make sure the names used in 4.3/4.4 come from `lpwm_stn`.

---

## 5. Config keys

| Key | Values | Default | Effect |
|---|---|---|---|
| `stn_backend` | `"reference"`, `"triton"` | `"reference"` | Triton crop/paste |
| `fused_composite` | bool | `false` | Fused paste + composite in the decoder |
| `torch_compile` | `false`, `true`/`"default"`, `"reduce-overhead"` | `false` | Compile the training forward |
| `cudnn_benchmark` | bool | `false` | Time conv algorithms, keep the fastest |
| `cudnn_deterministic` | bool | `true` | Keep `true` |
| `tf32_matmul` | bool | `false` | TF32 for matmul/linear (precision change; no gain for DLP) |
| `particle_dec_channels_last` | bool | `false` | Not recommended (2.7) |
| `seed` | int | none | Fixes initial weights, data order and sampling noise |
| `log_step_timing` | bool | `false` | `[step-timing]` line per epoch: median step ms, peak memory |
| `max_steps_per_epoch` | int | none | Cap steps per epoch (short tests) |

---

## 6. Verifying the port on your code

Run these after porting, on the GPU you train on. All three are measurement only. Run them one at a time (nothing else
on the GPU), from the repository root (or point `--repo-root` at it).

### 6.1 Kernel correctness — `benchmarks/stn/dlp_kernel_stress_check.py`
```bash
python benchmarks/stn/dlp_kernel_stress_check.py --config YOUR_CONFIG.json --out results/kernel_stress
```
- **Part 1** (independent of your model): crop, paste and fused composite at 256 particles, batch 4, 128×128 image,
  16×16 glimpses, scenarios `random`, `overlap` (all 256 glimpses on the same pixels), `border`, `tiny` (sub-pixel),
  `huge` (whole image), `ties` (equal depths, `obj_on` exactly 0/1), `all_off`. Every output and input gradient is
  compared with the original implementation evaluated in **float64** (ground truth).
- **Rule (fixed in the script):** a tensor passes if finite and `err_new ≤ 10 × err_ref32 + 1e-7`, where `err_ref32` is
  how far the original fp32 code is from float64. A wrong formula, index or race gives errors of 1e-3…1.
- **Part 2** (whole model) compares to float64 too, but the DLP model mixes in float32 tensors internally, so the
  float64 run usually fails; the script then prints "no verdict" and the fp32 differences. Use 6.2 instead.
- **Our result (GH200):** part 1 PASS in every scenario; worst `err_new / err_ref32` = 2.79 (`huge`, `bg_mask`,
  1.6e-7 vs 5.8e-8); `overlap` 1.11.
- **If you changed the semantics** (4.1), part 1 still compares `lpwm_stn`'s own reference vs its kernels; to compare
  against *your* original functions, temporarily point the `reference.*` calls in part 1 at them.

### 6.2 Whole-model gradient consistency — `benchmarks/stn/dlp_model_grad_diag.py`
```bash
python benchmarks/stn/dlp_model_grad_diag.py --config YOUR_CONFIG.json --n-kp-enc all --out results/grad_diag
```
`--n-kp-enc all` = no particle filtering (encoder and decoder use all prior particles); `""` = the config's value.
Same weights and batch, `model.eval()` + `deterministic=True` + `with_loss=True` (no sampling noise), TF32 off. Runs:
`orig`, `orig_again`, `orig_pert` (input × (1 + 1e-6·N(0,1))), `orig_tf32` (cuDNN TF32 on, i.e. how the original
trains), `opt`, `opt_again`, `stn_only`, `fused_only`. For every parameter gradient it prints the relative L2
difference to `orig` and the gradient's own norm.
- **Rule (fixed in the script):** consistent if for every tensor
  `relL2(opt) ≤ 10 × max(relL2(orig_pert), relL2(orig_again)) + 1e-7`.
- **Our result** (BAIR config, 256/256 particles, batch 4): **CONSISTENT, 0 of 315 tensors outside**. Median relL2:
  `opt` 1.6e-4, `orig_pert` 1.0e-3, `orig_tf32` 1.5e-2, `stn_only` 1.6e-4, `fused_only` 4.1e-4. The largest relative
  difference was the `prior_encoder.enc.conv_out` gradient, whose norm is 1.8e-8 (numerically zero) and which differs
  by 94% between two runs of the *original* — noise, not a bug. (Side observation: in this setting the prior encoder
  receives almost no gradient.)

### 6.3 Speed on your GPU — `benchmarks/stn/run_dlp_speed_matrix.sh`
```bash
bash benchmarks/stn/run_dlp_speed_matrix.sh YOUR_CONFIG.json 300 '{"n_kp_enc": "all"}'
```
Runs 8 short real `train_dlp.py` trainings (2 epochs × 300 steps; epoch 1 reported): original, Triton+fused,
+compile default, +reduce-overhead, +cudnn_benchmark (default and reduce-overhead), +TF32, original again (drift).
Writes `results/dlp_speed_*/report.txt` with step time, change vs original, peak memory, the particle counts the model
used, and each run's first-step path check. The third argument is applied to every run (omit it to use your config as
is). The script runs `python lpwm/train_dlp.py` relative to the parent directory of the repository; adjust that
line if your layout differs.

### 6.4 In every real run
Check the log for the `path check (first step) ok: ...` line and the `[step-timing]` lines. If you set `seed`, the
`init_weights_sha256` and `first_batch_sha256` values let you prove two runs started from the same state.

---

## 7. Background results (for context)

| Setting | Result | Evidence |
|---|---|---|
| Single-image DLP, RTX 3090, original vs Triton+fused+channels-last | Step −23.7% (347 → 265 ms), peak memory −22%; val PSNR difference −0.029 dB, 95% CI [−0.140, +0.082] | 6 paired seeds |
| Single-image DLP, GH200 | Table in Section 0 | 1 run per config, drift 1% |
| LPWM (BAIR, 17 frames, batch 5), GH200 | 820 → ~390 ms/step (−52%) with Triton+fused+compile+TF32; 94.4 → ~67 GB; ~347 ms with cuDNN benchmark (`default` compile) | TF32: 3 seeds + visual check, no consistent direction |
| Correctness at 256 particles | Section 6.1 / 6.2 | GH200 |

---

## 8. Caveats and open items
1. **GPU-specific.** Every number here is from RTX 3090 or GH200. On GH200 the Triton STN alone gives only −6% because
   the original `grid_sample` is already fast there; compile and cuDNN benchmark give most of the gain. Measure on your
   GPU (6.3).
2. **Particle count.** The −51% is with 90 particles. Without filtering (all 256) the balance changes (more
   paste/composite and decoder conv work, relatively less launch overhead); this configuration is being measured.
3. **Full-run time.** The speedup is per training step. Validation, image saving and checkpointing are not
   accelerated, so a whole run saves somewhat less. It does not drift over time: compute per step is fixed.
4. **Quality.** Correctness is established at the kernel and gradient level (Section 6), and single-image DLP quality
   was checked with 6 seeds at 90 particles on RTX 3090. A **long paired run at 256 particles on a complex dataset
   (original vs recommended, plus a second seed of the original as the yardstick) has not been done yet.**
5. **Reproducibility.** With `cudnn_benchmark: true`, two runs with the same seed are no longer bit-identical. For a
   bit-exact comparison set it to `false`.
6. **Precision.** The recommended config contains no precision-reducing change. TF32 matmul is the only such option
   and is not recommended for DLP.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `requested optimization path not in effect` / path check `RuntimeError` | The requested backend, fused composite or cuDNN flags did not take effect; check `lpwm_stn.set_backend` runs before the first forward and the model was built with `fused_composite=True` |
| `fused_composite=True but the input is unsupported: ...` | See the reason; typical: non-fp32 input (AMP), non-contiguous tensors in your own call path, `translation` is used, or `z_scale` shape `[B, K, 1]` |
| Dynamo / `SymInt` / stride assertion inside a Triton launcher under compile | The ops are not wrapped: `compile_for_training` must patch the module attribute your code actually calls (4.6) |
| Triton compile error mentioning `constexpr` on Triton ≥ 3.3 | Use this branch's `lpwm_stn/composite_backward.py` |
| Out of memory with `reduce-overhead` + `cudnn_benchmark` | Seen in LPWM only; use `torch_compile: "default"` |
| No speedup from `stn_backend: "triton"` | Inputs fell back (3.2): not CUDA/fp32, or `padding_mode` not `border` for crop |
| Slow first epoch | Compilation (1.5–3 min) and cuDNN algorithm trials; reported times use epoch 1+ |
