# LPWM STN optimization — Triton crop forward

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

The current autograd wrapper intentionally recomputes
`lpwm_stn.reference.stn_crop` during backward. Training gradients therefore
remain on the frozen path while the custom backward is developed. This version
claims forward acceleration only.

## Correctness contract

On the RTX 4090 Stage 1 environment:

```bash
python tests/stn/test_stn.py --device cuda --scope all \
  --backend triton --tol 1.5e-5 --grad-tol 1e-5 \
  --skip-gradient-determinism --live-forward --gradcheck
```

The full-output live comparison covers every element of 20 crop cases: small
correctness shapes, four retained-particle workloads, and all four larger
`n_kp_prior` attribute-encoder workloads. The largest observed forward absolute
error was `1.258e-5` for `obj3d128_prior/crop.scale`; no-scale cases were within
one float32 ULP (`1.192e-7`). Forward execution was deterministic. The nonzero
tolerance records fp32 coordinate-arithmetic reassociation, not a dtype or
algorithm change.

## RTX 4090 forward result

Python 3.10.16, torch 2.6.0+cu126, Triton 3.2.0, float32; median of 100
iterations after 10 warmups. Peak memory includes the live case inputs.

| case | reference ms | Triton ms | speedup | reference peak MB | Triton peak MB |
|---|---:|---:|---:|---:|---:|
| `bair/crop.scale` | 2.033 | 0.063 | 32.3× | 1,479.0 | 57.1 |
| `bair64/crop.scale` | 0.819 | 0.043 | 19.0× | 540.2 | 31.4 |
| `obj3d128/crop.scale` | 0.646 | 0.045 | 14.4× | 353.3 | 58.4 |
| `balls/crop.scale` | 0.128 | 0.033 | 3.9× | 87.4 | 25.9 |

The BAIR forward peak falls by about 1.42 GB (96%). Forward+backward does not
yet improve because backward deliberately pays for a reference recomputation;
the next stage must replace that path and validate gradients for `x`, `kp`, and
`z_scale`.

The attribute encoder's actual proposal path is larger than the retained path
above. It is tracked separately with `--scope prior` so it does not enlarge the
golden fixture. These rows are medians of 50 iterations after 10 warmups:

| case | particles | reference ms | Triton ms | speedup | reference peak MB | Triton peak MB |
|---|---:|---:|---:|---:|---:|---:|
| `bair_prior/crop.scale` | 256 | 5.758 | 0.133 | 43.3× | 4,157.7 | 97.2 |
| `bair64_prior/crop.scale` | 256 | 2.537 | 0.069 | 36.8× | 1,677.5 | 49.3 |
| `obj3d128_prior/crop.scale` | 64 | 3.441 | 0.170 | 20.2× | 1,708.2 | 135.2 |
| `balls_prior/crop.scale` | 64 | 0.697 | 0.047 | 14.8× | 369.6 | 41.8 |

```bash
python tests/stn/bench_stn.py --device cuda --scope prior \
  --iters 50 --warmup 10 --compare reference triton
```
