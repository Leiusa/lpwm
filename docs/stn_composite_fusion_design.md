# Paste + alpha/depth composite — reference isolation and fixture design

Pre-work for `research/stn-composite-fusion`. **No kernel exists yet and no branch
has been created.** This document fixes what the reference *is*, so fixtures can be
built from it before any fused implementation is written.

## 1. The exact chain

Two adjacent methods on `DLPDecoder` ([modules/modules.py:5294](../modules/modules.py#L5294) and
[:5306](../modules/modules.py#L5306)), called back-to-back from `decode_objects` at
[:5326](../modules/modules.py#L5326):

```
particle_dec(z_features, context=z_ctx)      -> dec_objects        [B*K, 4, p, p]
  reshape                                    -> dec_objects        [B, K, 4, p, p]
  translate_patches -> stn_paste             -> dec_objects_trans  [B, K, 4, H, W]   <-- big
  split(dim=2, [1,3])                        -> a_obj [B,K,1,H,W], rgb_obj [B,K,3,H,W]
--- get_objects_alpha_rgb_with_depth ---
  a_obj          = obj_on[:,:,None,None,None] * a_obj
  rgba_obj       = a_obj * rgb_obj                                  [B,K,3,H,W]      <-- big
  importance_map = a_obj * sigmoid(-z_depth[:,:,:,None,None])       [B,K,1,H,W]      <-- big
  importance_map = importance_map / (importance_map.sum(1, keepdim=True) + eps)
  dec_objects_trans = (rgba_obj * importance_map).sum(dim=1)        [B,3,H,W]   reduced
  alpha_mask        = 1.0 - (importance_map * a_obj).sum(dim=1)     [B,1,H,W]   reduced
  a_obj             = importance_map * a_obj                        [B,K,1,H,W] NOT reduced
```

`eps = 1e-5`, applied to the **sum** before division. Reduction is `dim=1` (particles).
Preserve both the epsilon placement and the reduction order: they are numerically load-bearing.

## 2. Contract

**Differentiable inputs:** `dec_objects` (the decoded RGBA glimpses), `z_kp`, `z_scale`,
`obj_on`, `z_depth`. `z_ctx` enters upstream through `particle_dec` and is outside this unit.

**Outputs:** `a_obj` → `alpha_masks` `[B,K,1,H,W]`, `alpha_mask` → `bg_mask` `[B,1,H,W]`,
`dec_objects_trans` `[B,3,H,W]`.

**Non-differentiable / structural:** `H = W = feature_map_size = image_size`; `eps`;
`padding_mode='zeros'` on the paste (inverse direction).

## 3. Intermediate sizes — where the memory goes

bair at the config's own `bs=5` (`B = bs*T = 5*17 = 85`, `K = 90`, `H = W = 128`, fp32):

| tensor | shape | bytes |
|---|---|---:|
| `dec_objects_trans` (post-paste) | `[85, 90, 4, 128, 128]` | **1.60 GB** |
| `rgba_obj` | `[85, 90, 3, 128, 128]` | 1.20 GB |
| `importance_map` | `[85, 90, 1, 128, 128]` | 0.40 GB |
| `a_obj` (returned) | `[85, 90, 1, 128, 128]` | 0.40 GB |

Each is retained for backward. This is why bair OOMs at bs=5 on a 24 GB card and why the
reference OOM at bs=2 landed exactly on `rgba_obj = a_obj * rgb_obj`
([modules/modules.py:5313](../modules/modules.py#L5313)).

## 4. The decisive structural finding

`alpha_masks` — the one `[B,K,...]` output that is **not** reduced — **never reaches the loss.**

`calc_dyn_elbo` binds it at [models.py:1363](../models.py#L1363) and `calc_static_elbo` at
[:1844](../models.py#L1844); an AST check confirms neither function ever reads the local
again. Every real consumer is visualization or evaluation: bounding-box plotting and
segmentation maps in `train_*.py` and `eval/eval_model.py`, all inside logging branches.

So the per-particle stack carries **no gradient path into the loss**, yet is materialized and
retained on every training step.

**This is the memory win, and it is also a decision that is not mine to make.** Making
`alpha_masks` optional changes the decoder's public output contract, which the standing
constraints protect. Three options, in increasing intrusiveness:

1. Fuse the two reduced outputs only, still materializing `alpha_masks`. Safe, contract
   unchanged, but keeps a `[B,K,1,H,W]` tensor — recovers roughly the 1.6 GB paste output
   and 1.2 GB `rgba_obj`, not the 0.4 GB mask.
2. Compute `alpha_masks` lazily, behind a flag defaulting to today's behaviour. Contract
   preserved by default; training can opt out.
3. Drop it from the training path entirely. Largest saving, changes Encoder-visible output.

I recommend **(2)**, but it needs your explicit authorization before implementation.

## 5. Fixture design

Same discipline as the existing STN fixtures: generate from the **untouched PyTorch chain**,
never from the fused kernel.

**Reference wrapper** — a free function reproducing §1 verbatim, taking
`(dec_objects, z_kp, z_scale, obj_on, z_depth, img_size, eps)` and returning the three
outputs. Mechanical edits only (`self.feature_map_size` → parameter), as was done for
`stn_crop`/`stn_paste` in `lpwm_stn/_baseline.py`.

**Forward fixtures** — SHA-256 over raw bytes of all three outputs, per case.

**Gradient fixtures** — for a fixed seeded cotangent on each of the three outputs, grads
w.r.t. all five differentiable inputs. Note the cotangent must cover `alpha_masks` too, even
though the model never backprops through it, or the kernel's handling of that output goes
untested.

**Cases** — beyond the standard shapes: `obj_on` at 0 and 1 (fully off / fully on particles);
`z_depth` spread wide enough to make the softmax-like weighting saturate; overlapping
particles at identical `z_kp`; particles pushed off-canvas so the paste contributes nothing;
near-zero alpha where the `+ eps` denominator dominates; `K=1` (reduction degenerate).

**Tolerances** — reuse the established `--tol 2e-5 / --grad-tol 1e-5` regime. The composite's
`sum(dim=1)` over `K` will reassociate under any fused reduction, so expect gradient
differences of the same character as the existing crop/paste kernels. Establish the tolerance
from the *reference's own* run-to-run spread before writing the kernel, not afterward from
whatever the kernel produces.

## 5b. Measured position after harness validation (2026-09-07, RTX 4090)

Validation run 2 on bair bs=1 confirmed where the remaining STN cost sits. With
crop and paste on Triton, the per-scope device kernel time is:

| scope | device kernel time | state |
|---|---:|---|
| masks (`create_masks_fast`) | **4.871 ms** | **unoptimized** — Triton backend does not implement it, so the registry resolves to `lpwm_stn.reference` |
| paste (Triton) | 0.432 ms | accelerated |
| crop (Triton) | 0.024 ms | accelerated |

Most of the masks cost is `affine_grid` grid construction, not sampling:
`F.affine_grid` dispatches `aten::bmm` (cuBLAS SGEMM on CUDA), `aten::linspace`,
`aten::fill_` and `aten::copy_` before `grid_sample` runs.

This does **not** change the decision to do composite fusion first. Fusion is
justified by memory — recovering bair's intended batch size — and `create_masks_fast`
runs under `no_grad`, so eliminating its 4.9 ms cannot recover the
`[B*T, K, C, H, W]` intermediates that force bs=1. But it does mean masks is the
largest remaining *timed* STN operation and the natural next target after fusion.

Validated end-to-end numbers from the same run (independent of profiler scope
attribution): inference `D_clean` 87.581 ms reference vs 81.687 ms Triton;
training 280.384 ms vs 258.442 ms; training peak allocated 18493.1 MB vs
17119.4 MB. Ten-iteration harness-validation figures — they do not replace the
pinned full baseline.

## 6. Order of work

1. Land the reviewed harness (`benchmarks/stn/`) as its own commit.
2. Write the reference wrapper + fixtures; validate the wrapper reproduces the live model path bit-exactly.
3. Only then create `research/stn-composite-fusion` from `389608c`.
4. Decide the `alpha_masks` question (§4) before, not during, kernel implementation.
