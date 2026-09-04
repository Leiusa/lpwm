# LPWM STN optimization — Triton crop forward and backward

This stage adds the first custom kernel behind the Stage 1 backend registry.
It changes neither the STN transform nor its dtype: float32 inputs still use
the same `align_corners=False`, bilinear, `padding_mode="border"` mapping.

## Scope

`lpwm_stn.triton_backend.stn_crop` fuses these reference operations:

1. expand one image per particle with `x.unsqueeze(1).repeat(...)`;
2. build the per-particle affine grid;
3. bilinearly sample the repeated image.

The kernel maps each output element back to the original image's batch index,
computes the affine coordinate on demand, and reads the four bilinear
neighbors directly. It therefore materializes neither the repeated input nor
the full sampling grid. Unsupported devices, dtypes, shapes, and padding modes
fall back to the reference implementation.

Backward is also fused and no longer recomputes the reference graph:

1. an output-parallel kernel atomically accumulates bilinear weights into the
   shared input image gradient;
2. a particle-parallel kernel reduces the coordinate derivatives into `kp`
   and `z_scale`, including the original sigmoid chain rule.

The border clamp derivative is zero at and beyond either image edge, matching
PyTorch's CUDA grid sampler. The input-gradient atomics retain CUDA's existing
nondeterministic accumulation characteristic; they do not change the operation
or dtype.

## Correctness contract

On the RTX 4090 Stage 1 environment:

```bash
python tests/stn/test_stn.py --device cuda --scope all \
  --backend triton --tol 2e-5 --grad-tol 1e-5 \
  --crop-grad-tol 1e-2 --paste-grad-tol 1e-1 \
  --skip-gradient-determinism --live-forward --live-gradients --gradcheck
```

The full-output live comparison covers every element of 20 crop cases: small
correctness shapes, four retained-particle workloads, and all four larger
`n_kp_prior` attribute-encoder workloads. The largest observed forward absolute
error was `1.258e-5` for `obj3d128_prior/crop.scale`; no-scale cases were within
one float32 ULP (`1.192e-7`). Forward execution was deterministic. The nonzero
tolerance records fp32 coordinate-arithmetic reassociation, not a dtype or
algorithm change.

The full-gradient live comparison covers `x`, `kp`, and `z_scale` for the same
20 cases, plus exact-border and selective-gradient tests. Across the real
retained and proposal workloads, the largest observed absolute errors were
`4.273e-4` for `x`, `9.186e-3` for `kp`, and `1.099e-3` for `z_scale`. The
largest reference `kp` gradient magnitude was about `6.0e3`; the explicit
`1e-2` acceptance bound records float32 coordinate and parallel-reduction
reassociation. No input, intermediate, or accumulator was demoted from
float32.

## RTX 4090 forward result

Python 3.10.16, torch 2.6.0+cu126, Triton 3.2.0, float32; median of 100
iterations after 10 warmups. Peak memory includes the live case inputs.

| case | reference ms | Triton ms | speedup | reference peak MB | Triton peak MB |
|---|---:|---:|---:|---:|---:|
| `bair/crop.scale` | 2.033 | 0.065 | 31.3× | 1,479.0 | 57.1 |
| `bair64/crop.scale` | 0.818 | 0.044 | 18.6× | 540.2 | 31.4 |
| `obj3d128/crop.scale` | 0.645 | 0.045 | 14.3× | 353.3 | 58.4 |
| `balls/crop.scale` | 0.128 | 0.033 | 3.9× | 87.4 | 25.9 |

The BAIR forward peak falls by about 1.42 GB (96%).

## RTX 4090 training result

The following includes forward and all requested input gradients. It uses the
same 100-iteration retained sweep as the forward table.

| case | reference fwd+bwd ms | Triton fwd+bwd ms | speedup | reference peak MB | Triton peak MB |
|---|---:|---:|---:|---:|---:|
| `bair/crop.scale` | 6.861 | 0.395 | 17.4× | 2,940.8 | 97.0 |
| `bair64/crop.scale` | 2.701 | 0.179 | 15.1× | 1,056.4 | 46.8 |
| `obj3d128/crop.scale` | 2.509 | 0.222 | 11.3× | 665.9 | 99.7 |
| `balls/crop.scale` | 0.579 | 0.117 | 4.9× | 152.5 | 34.5 |

For BAIR, custom forward+backward reduces peak allocated memory by about 2.84
GB (96.7%).

The attribute encoder's actual proposal path is larger than the retained path
above. It is tracked separately with `--scope prior` so it does not enlarge the
golden fixture. These rows are medians of 50 iterations after 10 warmups:

| case | particles | reference ms | Triton ms | speedup | reference peak MB | Triton peak MB |
|---|---:|---:|---:|---:|---:|---:|
| `bair_prior/crop.scale` | 256 | 5.696 | 0.128 | 44.5× | 4,157.7 | 97.2 |
| `bair64_prior/crop.scale` | 256 | 2.537 | 0.070 | 36.2× | 1,677.5 | 49.3 |
| `obj3d128_prior/crop.scale` | 64 | 3.444 | 0.171 | 20.1× | 1,708.2 | 135.2 |
| `balls_prior/crop.scale` | 64 | 0.699 | 0.047 | 14.9× | 369.6 | 41.8 |

The corresponding proposal-path training measurements are:

| case | reference fwd+bwd ms | Triton fwd+bwd ms | speedup | reference peak MB | Triton peak MB |
|---|---:|---:|---:|---:|---:|
| `bair_prior/crop.scale` | 19.306 | 0.881 | 21.9× | 8,297.1 | 176.2 |
| `bair64_prior/crop.scale` | 8.309 | 0.403 | 20.6× | 3,329.2 | 81.3 |
| `obj3d128_prior/crop.scale` | 12.769 | 0.978 | 13.1× | 3,375.3 | 253.3 |
| `balls_prior/crop.scale` | 2.761 | 0.248 | 11.1× | 716.8 | 66.4 |

```bash
python tests/stn/bench_stn.py --device cuda --scope prior \
  --iters 50 --warmup 10 --compare reference triton
```
