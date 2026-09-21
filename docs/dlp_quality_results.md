# DLP quality validation: 3 seeds x 3 paths x 8 epochs on BAIR (static DLP, original vgg loss)

Protocol (fixed before any run): `dlp_quality_protocol.md`, sha256 `399f32cd6758a74a33986c32d14510a66c769f5f9e61ba114d079928858a67b7`.
This stage looks for differences between the three paths over a longer, multi-seed training. **It does not prove quality
equivalence, no threshold was applied to any result, and no epoch, checkpoint or seed was selected.**

**Conclusion, limited to what was tested:** within the tested configuration, 3 seeds and 8 epochs, no clear and
consistent degradation of validation quality was observed. Quality equivalence is not claimed. The fused path stays
off by default.

## Setup (as pre-declared)

* Code: commit `b2cfa83b05d47edb254748fdd3b3c2223519103d` (clean snapshot). Paths: reference, triton (Triton crop/paste),
  fused (Triton crop + fused composite).
* Data: the fixed-prefix BAIR subset (400 train / 40 internal-validation episodes, manifest aggregate sha256
  `64917d7eeb7da3826e9d1524b6c3f2d70ec0ea0fb5f79d4722ba7a8fc086f431`), 128x128, 90 particles, original vgg loss and every
  other setting of `static_bair128_vgg.json` (sha256 `f40efb09794db93e132a3f16f4c54d7fab68c68d1cc6135a06026b9e08a8b64d`); B=16.
* Seeds 0, 1, 2, one A6000 job per seed (3959715, 3959716, 3959717, all on grogu-1-25). Within a seed the three paths share
  initial weights, data order and sampling noise (weight and first-batch hashes are identical); run orders were balanced
  across seeds. 8 epochs per path per seed through `train_dlp.py`.
* Reported checkpoint: the final one (after epoch 8). Evaluation: every final checkpoint on the reference path over all
  1,200 validation images (MSE over all pixels, PSNR = -10 log10 MSE); the validation loss is the one `train_dlp.py` logs
  after epoch 8. Per-epoch curves are in section E of the output below.
* Integrity checks (all passed): finite values everywhere, 8 epochs per run, 1,200 validation images, identical
  initial-weight and first-batch hashes across paths within a seed, path check correct (fused executions in step 0:
  reference 0, triton 0, fused 1). The first-step loss differs between paths by 1.2e-6, 4.9e-7 and 1.3e-6 (relative,
  seeds 0/1/2); it is reported, not a criterion.

## Reading of the results

Pre-declared comparisons at the final checkpoint, per seed (difference = first minus second; negative validation loss and
MSE and positive PSNR mean the first path is better):

| Seed | Comparison | Val loss | Full-val MSE | Full-val PSNR |
|---|---|---:|---:|---:|
| 0 | triton - reference | -0.198 (-1.66%) | -3.75% | +0.166 dB |
| 1 | triton - reference | -0.009 (-0.08%) | -0.51% | +0.022 dB |
| 2 | triton - reference | -0.099 (-0.84%) | +1.44% | -0.062 dB |
| 0 | fused - triton | -0.186 (-1.58%) | -4.97% | +0.222 dB |
| 1 | fused - triton | -0.164 (-1.40%) | -0.48% | +0.021 dB |
| 2 | fused - triton | +0.084 (+0.72%) | +3.17% | -0.135 dB |

Over the 3 seeds (mean, sample std, min to max): triton - reference: validation loss -0.86% (0.79%, -1.66% to -0.08%), PSNR
+0.042 dB (0.115, -0.062 to +0.166); fused - triton: validation loss -0.75% (1.28%, -1.58% to +0.72%), PSNR +0.036 dB
(0.179, -0.135 to +0.222).

* No path is worse in the mean and no consistent degradation appeared. The signs of the MSE and PSNR differences change
  from seed to seed; the validation-loss difference is negative in all three seeds for triton - reference and in two of
  three for fused - triton.
* The scale to read these against is each path's own spread over seeds (section C): PSNR standard deviation 0.10 to
  0.20 dB and validation-loss standard deviation 0.05 to 0.13. The paired differences are of the same size. With three
  seeds the study cannot resolve differences below that scale, so it neither establishes equivalence nor an improvement.
* Training-loss offset (kept separate, section D): the +1.0% to +2.2% offset of the earlier 3-epoch seed-0 run was not
  reproduced as a consistent offset. Triton relative to the reference: +0.73% at epoch 1, shrinking to +0.12% at
  epoch 8 (final epoch per seed -0.27%, +0.56%, +0.08%); fused: +0.23% at epoch 1 to -0.53% at epoch 8 (per seed
  -0.71%, -0.05%, -0.83%). The sign depends on seed and epoch. It is recorded and not explained.
* The seed-0 reconstruction grid (target, reference, triton, fused) shows no colour shift or corruption on any path.
* Cost context is consistent with the batch scan: step time about 328 to 333 / 281 to 285 / 268 to 272 ms and peak
  allocated memory 10.76 / 9.47 / 8.34 GB for reference / triton / fused.

## Limitations that remain in force

* **Default off, strict input support.** `fused_composite` is opt-in. It accepts fp32, contiguous CUDA tensors,
  `dec_objects [B,K,4,p,p]` with square patches and `translation is None`; anything else (AMP/fp16, non-contiguous,
  CPU, ...) raises `ValueError` instead of falling back. Double backward is refused.
* **Two known Triton crop failures are unchanged.** `tests/stn/live_check.py` (same GPU, original thresholds): 30 of 32
  cases pass; `bair_prior/crop.scale` (`grad[kp]`, `grad[z_scale]`) and `obj3d128_prior/crop.scale` (`grad[kp]`) fail.
  They were attributed, without proof, to bilinear-kink sensitivity and were not resolved by this work.
* The `dec_objects` gradient uses atomics and is not bit-reproducible run to run; the forward composite kernel keeps an
  unaddressed flush-to-zero difference in `tl.math.div_rn`.
* Static DLP only; the LPWM dynamics model was not evaluated. The validation split is an internal split of a fixed
  prefix subset (400 train / 40 validation episodes), not the official validation set.

## Not established

Long-term quality (8 epochs, 400 training episodes, 3 seeds); behaviour with a different validation split, other
batch sizes, resolutions or particle counts; dynamics (LPWM) models.

## Provenance (results live outside the repository)

Results: `/grogu/user/junhong3/lpwm-work/experiments/fused-integration/quality-seed{0,1,2}-<job>/` and `.../quality/`.

| File | sha256 |
|---|---|
| `benchmarks/stn/quality_run.py` | `0b947e0f187dab7acf9406d1f4efaaa6fff9bc4919b84676562b00994827c053` |
| `benchmarks/stn/summarize_quality.py` | `a28898c21ae085dc4fb9b859d83820248377c78320d456dced75ec1b9305230a` |
| job script `run_quality.sh` | `b4b379dd7aac2272be7e8db010d555f4ea4f4ad13015086e5b6b8277e3710e4d` |
| `quality_summary.txt` (the output below) | `16619cb41825c1d320377e8dd35ed4a520409d11511a2dc912354dba63c7d8b3` |
| `quality_summary.json` | `c0e9a8ffe457a09626c991a9b8d31f906796464bd95bfa4a8ef6a6731b42a661` |

Per-seed files (`train_report.json`, `env.txt`, `reconstructions.png`):

```
seed 0 (job 3959715)
  bee9ac5b6d09279a2f9e2dc491fd9e51488d3f042cabdaf2b110b2b7770571e3  train_report.json
  0752b4f9eca12a4584caa2b9d74a813e37a747f1426f546b2f63c35a64751b50  env.txt
  e0c196f1382e92eec07000d620751dacdc24d2e547ceb97e0d1296da0aa81fc6  reconstructions.png
seed 1 (job 3959716)
  8d173cbc2610a63f362e87f7ac6c84a263b37c0858926f8fb0475ce8bfc7ef16  train_report.json
  0b7ba6f9188ee1d82a711a61bf8ba73fcb78873e0ed054dfe532ee4fc41bdb7b  env.txt
  2665dc4609d59bc4b1a12813996404878bd6e0f90f3637282f1d1b16d8086209  reconstructions.png
seed 2 (job 3959717)
  3e32e7133339b8a652f2d37397074d58def4751eb8b88d200b35723caffdc189  train_report.json
  38b6b1df83a12822235ab0de4f1d775328f3d7b207d308634c4be5423abe7cfc  env.txt
  cc51ab865d3ce8115b873511f4501bdcfe445b27856638951b633fa940285fb2  reconstructions.png
```

Each `env.txt` lists the snapshot file hashes, the data manifest hash and the weight hashes. Reproduce a seed inside an A6000 allocation:
`SEED=<s> ORDER=<a+b+c> sbatch quality/run_quality.sh`, then
`python benchmarks/stn/summarize_quality.py out.json <seed dirs>` (Python >= 3.8).

## Output of `summarize_quality.py`

```
integrity checks: ALL PASSED
first-step loss (reported, not a criterion): {0: {'values': {'reference': 74.41317749, 'triton': 74.413085938, 'fused': 74.413085938}, 'max_relative_spread': 1.2303197241236485e-06}, 1: {'values': {'reference': 77.291465759, 'triton': 77.291503906, 'fused': 77.291481018}, 'max_relative_spread': 4.935473745171218e-07}, 2: {'values': {'reference': 72.002555847, 'triton': 72.002578735, 'fused': 72.0026474}, 'max_relative_spread': 1.2715243080931676e-06}}
seeds: [0, 1, 2] | epochs per path: 8 | jobs: {0: '3959715', 1: '3959716', 2: '3959717'} | hosts: {0: 'grogu-1-25', 1: 'grogu-1-25', 2: 'grogu-1-25'} | run orders: {0: ['reference', 'triton', 'fused'], 1: ['triton', 'fused', 'reference'], 2: ['fused', 'reference', 'triton']}

=== A. Final checkpoint (epoch 8), per seed
seed  path       val loss     full-val MSE full-val PSNR train loss fixed-16 PSNR 
0     reference  11.944       5.8064e-03   22.361       9.880      21.885        
0     triton     11.746       5.5887e-03   22.527       9.853      22.237        
0     fused      11.560       5.3107e-03   22.748       9.809      22.428        
1     reference  11.700       5.3904e-03   22.684       9.882      22.226        
1     triton     11.691       5.3627e-03   22.706       9.937      22.557        
1     fused      11.527       5.3370e-03   22.727       9.878      22.122        
2     reference  11.740       5.5030e-03   22.594       9.945      22.442        
2     triton     11.641       5.5822e-03   22.532       9.953      22.348        
2     fused      11.725       5.7590e-03   22.397       9.863      22.023        

=== B. Per-seed differences (first name minus second) and their spread over seeds

  triton - reference
    seed 0: val loss -0.198 (-1.66%) | full-val MSE -2.177e-04 (-3.75%) | full-val PSNR +0.166 dB
    seed 1: val loss -0.009 (-0.08%) | full-val MSE -2.773e-05 (-0.51%) | full-val PSNR +0.022 dB
    seed 2: val loss -0.099 (-0.84%) | full-val MSE +7.921e-05 (+1.44%) | full-val PSNR -0.062 dB
    val_loss           mean -0.102 | sample std 0.0945 | min -0.198 | max -0.009
    val_loss_pct       mean -0.8593 | sample std 0.791 | min -1.658 | max -0.07692
    full_val_mse       mean -5.541e-05 | sample std 0.00015 | min -0.0002177 | max +7.921e-05
    full_val_mse_pct   mean -0.9415 | sample std 2.62 | min -3.75 | max +1.439
    full_val_psnr_db   mean +0.0421 | sample std 0.115 | min -0.06207 | max +0.166

  fused - triton
    seed 0: val loss -0.186 (-1.58%) | full-val MSE -2.780e-04 (-4.97%) | full-val PSNR +0.222 dB
    seed 1: val loss -0.164 (-1.40%) | full-val MSE -2.569e-05 (-0.48%) | full-val PSNR +0.021 dB
    seed 2: val loss +0.084 (+0.72%) | full-val MSE +1.768e-04 (+3.17%) | full-val PSNR -0.135 dB
    val_loss           mean -0.08867 | sample std 0.15 | min -0.186 | max +0.084
    val_loss_pct       mean -0.7549 | sample std 1.28 | min -1.584 | max +0.7216
    full_val_mse       mean -4.228e-05 | sample std 0.000228 | min -0.000278 | max +0.0001768
    full_val_mse_pct   mean -0.7617 | sample std 4.08 | min -4.974 | max +3.168
    full_val_psnr_db   mean +0.03566 | sample std 0.179 | min -0.1354 | max +0.2216

  fused - reference   (additional, not a pre-declared comparison)
    seed 0: val loss -0.384 (-3.22%) | full-val MSE -4.957e-04 (-8.54%) | full-val PSNR +0.388 dB
    seed 1: val loss -0.173 (-1.48%) | full-val MSE -5.341e-05 (-0.99%) | full-val PSNR +0.043 dB
    seed 2: val loss -0.015 (-0.13%) | full-val MSE +2.560e-04 (+4.65%) | full-val PSNR -0.198 dB
    val_loss           mean -0.1907 | sample std 0.185 | min -0.384 | max -0.015
    val_loss_pct       mean -1.607 | sample std 1.55 | min -3.215 | max -0.1278
    full_val_mse       mean -9.769e-05 | sample std 0.000378 | min -0.0004957 | max +0.000256
    full_val_mse_pct   mean -1.625 | sample std 6.62 | min -8.537 | max +4.653
    full_val_psnr_db   mean +0.07776 | sample std 0.294 | min -0.1975 | max +0.3875

=== C. Context: each path's own spread over seeds at the final checkpoint (the scale the paired differences should be read against)
  reference  val loss       mean 11.79 | sample std 0.131 | range 11.7 .. 11.94
  reference  full-val MSE   mean 0.005567 | sample std 0.000215 | range 0.00539 .. 0.005806
  reference  full-val PSNR  mean 22.55 | sample std 0.167 | range 22.36 .. 22.68
  triton     val loss       mean 11.69 | sample std 0.0525 | range 11.64 .. 11.75
  triton     full-val MSE   mean 0.005511 | sample std 0.000129 | range 0.005363 .. 0.005589
  triton     full-val PSNR  mean 22.59 | sample std 0.102 | range 22.53 .. 22.71
  fused      val loss       mean 11.6 | sample std 0.106 | range 11.53 .. 11.72
  fused      full-val MSE   mean 0.005469 | sample std 0.000252 | range 0.005311 .. 0.005759
  fused      full-val PSNR  mean 22.62 | sample std 0.197 | range 22.4 .. 22.75

=== D. Training-loss offset relative to reference, per epoch (percent), kept separate from the quality comparison
  triton  e1 +0.73  e2 +0.29  e3 +0.20  e4 +0.42  e5 +0.17  e6 +0.08  e7 +0.15  e8 +0.12   (mean over seeds)
          per-seed final epoch: seed 0 -0.27%, seed 1 +0.56%, seed 2 +0.08%
  fused   e1 +0.23  e2 +0.48  e3 +0.15  e4 -0.13  e5 -0.38  e6 -0.66  e7 -0.68  e8 -0.53   (mean over seeds)
          per-seed final epoch: seed 0 -0.71%, seed 1 -0.05%, seed 2 -0.83%

=== E. Validation loss per epoch (curves; every epoch is shown, none selected)
  seed 0 reference   16.346  14.070  13.026  12.413  12.358  11.963  11.956  11.944
  seed 0 triton      16.643  14.046  13.002  12.454  12.164  11.981  11.808  11.746
  seed 0 fused       16.494  14.030  13.005  12.459  12.221  11.931  11.835  11.560
  seed 1 reference   16.269  13.977  13.005  12.585  12.138  11.998  11.815  11.700
  seed 1 triton      16.300  14.092  13.041  12.441  12.201  12.007  11.769  11.691
  seed 1 fused       16.269  14.004  12.990  12.378  12.019  11.858  11.629  11.527
  seed 2 reference   16.206  14.100  12.924  12.586  12.116  12.092  11.881  11.740
  seed 2 triton      16.118  13.967  12.869  12.434  12.155  11.901  11.767  11.641
  seed 2 fused       15.952  13.947  12.876  12.503  12.157  11.800  11.601  11.725

=== F. Cost context per path (median over epochs within each job; jobs may sit on different nodes)
  reference  step ms [328.2, 332.5, 328.8] | peak alloc MB [10761, 10760, 10762] | epoch wall s [247.0, 250.2, 247.7]
  triton     step ms [281.3, 285.4, 282.3] | peak alloc MB [9472, 9472, 9473] | epoch wall s [211.9, 214.8, 212.7]
  fused      step ms [268.0, 272.0, 269.8] | peak alloc MB [8338, 8338, 8340] | epoch wall s [202.0, 204.9, 203.4]

wrote /grogu/user/junhong3/lpwm-work/experiments/fused-integration/quality/quality_summary.json
```
