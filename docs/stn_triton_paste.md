# LPWM STN optimization — Triton paste forward and backward

This stage adds `stn_paste` to the same backend registry as the custom crop.
The public API, output layout, transform, dtype, and model call site stay
unchanged: float32 still uses `align_corners=False`, bilinear sampling, zeros
padding, and the inverse scale/translation equations from the reference.

## Kernel boundary

The forward kernel replaces theta construction, `affine_grid`, and
`grid_sample`. It evaluates the inverse affine coordinate for each canvas
element and reads the four valid patch neighbors directly. It still returns
the original `[batch, n_kp, channels, image, image]` tensor; removing that
large stack requires the later paste + alpha/depth compositing fusion.

Backward uses two paths:

1. an output-parallel kernel atomically accumulates `dL/dpatches`;
2. a tiled kernel computes partial coordinate derivatives, followed by a
   deterministic per-particle reduction for `dL/dkp` and `dL/dscale`.

The normalized-scale path and the unnormalized sigmoid chain are both
implemented. `translation` remains accepted and ignored, exactly as in the
reference. Unsupported device, dtype, shape, or channel-count cases fall back
to the reference backend.

## Correctness contract

The combined crop + paste backend passed the following on the RTX 4090 Stage 1
environment:

```bash
python tests/stn/test_stn.py --device cuda --scope all \
  --backend triton --tol 2e-5 --grad-tol 1e-5 \
  --crop-grad-tol 1e-2 --paste-grad-tol 1e-1 \
  --skip-gradient-determinism --live-forward --live-gradients --gradcheck
```

This checks 60 fixed CUDA golden cases through both reference and dispatch,
then performs full-tensor live comparisons for 32 accelerated crop/paste
cases. Paste contributes 12 scale/no-scale cases across the two correctness
and four real retained-particle workloads. Extra tests independently exercise
normalized scale, the ignored translation argument, and each conditional
gradient output.

The largest observed paste forward absolute error was `1.174e-5`. Across the
real workloads the largest backward absolute errors were `4.768e-5` for
`patches_batch`, `7.080e-2` for `kp_batch`, and `2.527e-2` for `scale`.
Reference coordinate gradients reached approximately `1.8e4`; the explicit
`1e-1` paste bound records float32 inverse-coordinate and reduction
reassociation. The algorithm and float32 precision are unchanged.

CUDA input-gradient accumulation is atomic and therefore not byte-
deterministic in both the PyTorch reference sampler and this kernel. Forward
determinism remains enabled; only the gradient rerun hash is skipped.

## RTX 4090 inference result

Python 3.10.16, torch 2.6.0+cu126, Triton 3.2.0, float32; median of 50
iterations after 10 warmups. Peak memory includes live inputs and output.

| case | reference forward ms | Triton forward ms | speedup | reference peak MB | Triton peak MB |
|---|---:|---:|---:|---:|---:|
| `bair/paste.scale` | 23.756 | 2.185 | 10.9× | 2,869.5 | 1,925.6 |
| `bair64/paste.scale` | 8.495 | 0.808 | 10.5× | 1,026.1 | 690.3 |
| `obj3d128/paste.scale` | 4.800 | 0.480 | 10.0× | 598.4 | 409.6 |
| `balls/paste.scale` | 1.009 | 0.127 | 7.9× | 131.5 | 92.1 |

## RTX 4090 training result

The same benchmark with gradients enabled produced:

| case | reference fwd+bwd ms | Triton fwd+bwd ms | speedup | reference peak MB | Triton peak MB |
|---|---:|---:|---:|---:|---:|
| `bair/paste.scale` | 105.359 | 9.323 | 11.3× | 6,210.7 | 3,858.6 |
| `bair64/paste.scale` | 36.408 | 3.522 | 10.3× | 2,219.8 | 1,383.2 |
| `obj3d128/paste.scale` | 20.883 | 1.864 | 11.2× | 1,291.2 | 820.7 |
| `balls/paste.scale` | 4.373 | 0.438 | 10.0× | 282.5 | 184.5 |

For BAIR, the isolated paste gains about 10.9× in inference and 11.3× for
forward+backward. Peak allocated memory falls by about 944 MB for forward and
2.35 GB for training. The remaining 1.9 GB forward footprint is principally
the required per-particle canvas stack and is the target of compositing fusion.

```bash
python tests/stn/bench_stn.py --device cuda --scope bench \
  --op stn_paste --iters 50 --warmup 10 --compare reference triton
```
