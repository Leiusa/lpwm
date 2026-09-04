# LPWM STN optimization — stage 1: isolation and reference tests

Goal of the overall effort: replace the spatial-transformer stage with a custom
fused kernel to speed up training and inference.

Goal of **stage 1** (this document): make that safe. Get every spatial-transformer
operation behind one module with one swappable backend, and pin the current
numerics — forward *and* backward — so a kernel can be judged against something
exact instead of against a vibe. **Stage 1 changes no algorithm and no precision.**

---

## What the STN stage is

Two directions, both `F.affine_grid` + `F.grid_sample` under a 2×3 similarity
transform built from a particle's position and scale:

| direction | where | call |
|---|---|---|
| **crop** — image → one glimpse per particle | `ParticleAttributeEncoder.forward`, `ParticleFeaturesEncoder.forward` | `stn_crop`, `inverse=False`, `padding_mode='border'` |
| **paste** — decoded glimpse per particle → canvas | `DLPDecoder.translate_patches` | `stn_paste`, `inverse=True`, `padding_mode='zeros'` |
| **masks** — square particle masks | `DLPEncoder.get_bg_mask_from_particle_glimpses` | `create_masks_fast`, `create_masks_with_scale`, both under `no_grad` |

The reference harness combines config values as
`batch_size × timestep_horizon × n_kp`. These are deterministic fixture sizes,
not a substitute for capturing the tensors in a representative model run. The
table shows both the retained count (`n_kp_enc`) and the larger proposal count
(`n_kp_prior`) used by the attribute crop:

| config | bs | T | retained / prior | image | glimpse | retained / prior problems |
|---|---|---|---|---|---|---|
| `bair` | 5 | 16 | 90 / 256 | 128² | 16² | 7,200 / 20,480 |
| `bair64` | 8 | 16 | 80 / 256 | 64² | 8² | 10,240 / 32,768 |
| `obj3d128` | 6 | 20 | 12 / 64 | 128² | 32² | 1,440 / 7,680 |
| `balls` | 5 | 20 | 12 / 64 | 64² | 16² | 1,200 / 6,400 |

## What stage 1 built

```
lpwm_stn/
  __init__.py      backend registry + the dispatching front-ends the model imports
  reference.py     the isolated ops (primitives + the two fused ops)
  workloads.py     problem sizes read out of configs/*.json, deterministic inputs
  _baseline.py     FROZEN verbatim copy of the pre-optimization code — the oracle
tests/stn/
  stn_cases.py     the shared case list (op × workload × options)
  golden.py        fixture format: SHA-256 over raw tensor bytes
  gen_golden.py    records the frozen baseline's answers
  test_stn.py      checks an implementation against them
  bench_stn.py     time and memory, same shapes
  fixtures/stn_golden_cpu.pt
```

`utils/util_func.py` and `modules/modules.py` now import their STN ops from
`lpwm_stn`; the inlined crop/paste blocks at the three call sites are gone.

### The two fused ops

`stn_crop` and `stn_paste` take the **un-expanded** image / patch tensors and do
the `repeat` internally. That is deliberate: it puts the expansion inside the op
boundary, so a kernel can fuse it away without touching model code — while the
reference still performs it, so stage 1 stays bit-identical.

## The contract a kernel must meet

`python tests/stn/test_stn.py --scope all --backend <name>`

1. **Forward, bit-exact** — every case, against the frozen baseline.
2. **Backward, bit-exact** — `dL/dimage`, `dL/dkp`, `dL/dz_scale`, `dL/dtheta`
   for a fixed upstream cotangent.
3. **Determinism** — two runs agree byte for byte. Catches a backward that
   accumulates `dL/dimage` with atomics in nondeterministic order.
4. **Isolation** — no raw `affine_grid`/`grid_sample` outside `lpwm_stn`, the
   model still imports the ops, and `DLPDecoder.translate_patches` still matches.

Comparison is **SHA-256 over raw tensor bytes**, not `allclose`. That is what lets
full training shapes be pinned in a 1.5 MB fixture. Small tensors are also stored
whole so a failure can be diffed element-wise.

A kernel that legitimately reassociates arithmetic will not be bit-exact. Run it
with an explicit `--tol` and record why — do not relax the default.
`--gradcheck` gives an oracle-free float64 check for the caller-typed
`affine_grid_sample` primitive. The fused paths deliberately retain their
baseline float32-theta behavior and are covered by the pinned gradient tests.

### CUDA backward caveat

The first RTX 4090 run showed that PyTorch 2.6's CUDA `grid_sample` backward is
not bit-deterministic for the sampled-input gradient: repeated executions of
the frozen baseline differed by roughly `1e-7` to `4e-6` on the correctness
cases. Forward remained bit-exact, coordinate/scale gradients stayed within the
same numerical envelope, and float64 `gradcheck` passed. This is the documented
atomic-accumulation behaviour of the CUDA sampler, not an isolation change.

For baseline acceptance on CUDA, keep the forward comparison bit-exact, apply
an explicit tolerance only to gradients, and skip only the gradient rerun
assertion:

```bash
python tests/stn/test_stn.py --device cuda --scope all --grad-tol 1e-5 --skip-gradient-determinism
```

Forward determinism remains checked by that command. Keep gradient determinism
enabled when evaluating a custom backend if deterministic backward is one of
that backend's requirements. The test runner also stores failure messages
rather than exception objects so failed CUDA cases cannot retain traceback
tensors and exhaust GPU memory during a sweep.

## Registering a backend

```python
import lpwm_stn
lpwm_stn.register_backend("triton", my_module)   # needs only the ops it accelerates
lpwm_stn.set_backend("triton")
```
Ops a backend does not define fall back to `reference`.

---

## Findings carried into stage 2

**1. The `repeat` is the leading optimization candidate from code inspection.**
`stn_crop` materializes `[bs·T·n_kp, ch, H, W]`. With the config-derived BAIR
fixture, that is about **1.4 GB** for 90 retained particles and **4.0 GB** for
256 proposal particles. Every particle in a frame samples the same source
image, so a fused kernel may be able to index it directly. A GPU profile must
confirm the live shapes, allocation, and runtime impact before calling this a
speedup; changed accumulation order must still pass the numerical contract.

**1b. The paste direction allocates more than the crop, and is the more
promising fusion.** `stn_paste` returns `[bs·T, n_kp, ch, H, W]` — about
**1.9 GB** on the BAIR fixture (80 × 90 × 4 × 128²) — and
`get_objects_alpha_rgb_with_depth` immediately reduces it over the particle axis
into one canvas. The per-particle stack is never read for anything else, so
fusing paste with the depth-weighted alpha composite would remove it entirely.
Two caveats before acting: this fusion crosses the current op boundary (paste
and the composite are separate functions today), so it needs its own golden case
pinning the *composited* canvas rather than the stack; and the CPU timings below
should not be read as a GPU ranking. The same reduce-after-materialize shape
applies to the mask builders.

**2. `spatial_transform` hardcodes the affine theta to float32.**
`torch.zeros(2, 3, device=image.device)` carries no `dtype`. Consequently,
float64 `gradcheck` cannot run on these fused ops because `grid_sample` rejects
a float32 grid with a float64 input. A kernel must **reproduce** the existing
float32 theta behavior in this stage. The actual CUDA autocast dtypes for
`affine_grid` and `grid_sample` remain to be measured before the later mixed-
precision study.

**3. `theta` is built by six masked scatter-writes into a zeroed tensor**, then
`affine_grid` expands it to a full sampling grid. For a similarity transform the
grid is separable in x and y and can be computed on the fly in the sampling loop.

**4. The masks path is `no_grad` and `nearest`-mode**, so it needs forward only —
a cheaper, separate kernel from the bilinear crop/paste path.

**5. `translate_patches` accepts a `translation` argument it never uses.** The
signature is preserved for call compatibility; do not build kernel behaviour on it.

## Recorded CPU baseline

`python tests/stn/bench_stn.py --scope bench --iters 3`, reference backend,
torch 2.9.1, median of 3. **CPU only — this is a starting line and a regression
check, not a GPU ranking and not a target.** Re-record on the target GPU before
kernel work begins.

| case | fwd ms | fwd+bwd ms | note |
|---|---:|---:|---|
| `bair/crop.scale` | 71.5 | 195.4 | repeat allocates 1416 MB (×90) |
| `bair/paste.scale` | 852.4 | 1943.9 | output stack ≈1.9 GB, then reduced away |
| `bair/masks.fast` | 681.5 | — | forward only (`no_grad`) |
| `bair64/crop.scale` | 20.4 | 64.9 | repeat allocates 503 MB (×80) |
| `bair64/paste.scale` | 447.3 | 852.6 | |
| `obj3d128/crop.scale` | 30.8 | 67.4 | repeat allocates 283 MB (×12) |
| `obj3d128/paste.scale` | 179.6 | 385.9 | |
| `balls/crop.scale` | 6.4 | 14.0 | repeat allocates 59 MB (×12) |
| `balls/paste.scale` | 50.5 | 98.6 | |

On CPU the ordering is consistent across configs: paste costs roughly 10× crop,
and the mask builders cost about as much as paste despite being `no_grad` and
`nearest`-mode. Whether that ordering survives on GPU is exactly what the first
stage-2 measurement should establish.

## Recorded RTX 4090 baseline

RunPod Secure Cloud, NVIDIA GeForce RTX 4090 (SM 8.9), Python 3.10.16,
torch 2.6.0+cu126, float32. Each time is the median of 10 iterations after 3
warmups. Memory is PyTorch peak allocated memory with the case inputs live, so
it represents the capacity needed to execute the isolated call rather than only
the incremental allocation inside the op.

| case | forward ms | forward+backward ms | forward peak MB | training peak MB |
|---|---:|---:|---:|---:|
| `bair/crop.scale` | 2.033 | 7.015 | 1,479.0 | 2,940.8 |
| `bair/paste.scale` | 23.768 | 103.639 | 2,878.0 | 6,210.7 |
| `bair/masks.fast` | 22.872 | — | 2,856.0 | — |
| `bair/masks.scale` | 22.234 | — | 2,384.1 | — |
| `bair64/crop.scale` | 0.820 | 2.709 | 540.2 | 1,056.4 |
| `bair64/paste.scale` | 8.512 | 36.739 | 1,034.6 | 2,219.8 |
| `obj3d128/crop.scale` | 0.646 | 2.516 | 353.3 | 665.9 |
| `obj3d128/paste.scale` | 4.801 | 20.861 | 606.9 | 1,291.2 |
| `balls/crop.scale` | 0.130 | 0.550 | 87.4 | 152.5 |
| `balls/paste.scale` | 1.011 | 4.347 | 140.0 | 282.5 |

The GPU ranking is now measured rather than inferred: paste backward is the
largest training cost, while paste and both mask builders dominate inference
time. Paste also has the largest peak allocation. `stn_crop` remains a useful
first custom kernel because it has the narrowest boundary and eliminating its
materialized repeat directly removes 1,416 MB in the BAIR retained-particle
case. After that proof of correctness and backend wiring, paste is the larger
performance target.

## Running it

```bash
python tests/stn/test_stn.py                       # small shapes, fast
python tests/stn/test_stn.py --scope all           # + full training shapes
python tests/stn/test_stn.py --gradcheck
python tests/stn/bench_stn.py --device cuda --scope bench
python tests/stn/gen_golden.py --device cuda --scope all  # pin on a GPU box
python tests/stn/test_stn.py --device cuda --scope all --grad-tol 1e-5 --skip-gradient-determinism
```

Fixtures are device-specific: CUDA's sampler does not produce CPU's bytes.
Generate the CUDA fixture from `_baseline` on the target GPU **before** starting
kernel work, so the kernel is compared against that machine's baseline.

The immutable Stage 1 tag makes no acceleration claim; the GPU measurements
above were recorded later on the kernel-development branch against that tag.
The implementation and validation reports are in
[`stn_triton_crop.md`](stn_triton_crop.md) and
[`stn_triton_paste.md`](stn_triton_paste.md).
