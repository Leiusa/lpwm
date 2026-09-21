# DLP batch-size scan (B = 16 / 32 / 64) on BAIR, static DLP, original vgg loss

Question: do the reference, the Triton crop/paste path and the Triton + fused-composite path keep their relative
benefit at larger batch sizes, and does the optimised path let a larger batch fit? Only the batch size changes.

**Tested up to B = 64 only. This is not a search for the largest batch and no maximum is claimed.**

## Setup

* Code: commit `b2cfa83b05d47edb254748fdd3b3c2223519103d` (clean git snapshot, same version for all paths).
* Data and settings: the fixed-prefix BAIR subset (400 train / 40 internal-validation episodes; manifest aggregate
  sha256 `64917d7eeb7da3826e9d1524b6c3f2d70ec0ea0fb5f79d4722ba7a8fc086f431`), static DLP, 128x128, 90 particles, the
  original vgg loss and every other setting of `static_bair128_vgg.json` (see `dlp_fused_composite_integration.md`).
* Flow: `benchmarks/stn/batch_scan.py` reproduces the real `train_dlp.py` step (BAIRImage DataLoader with seeded
  shuffling, 4 workers, pinned memory, drop_last; original loss; same Adam settings; model_output alive until the step
  ends), post-warmup steady state (`warmup=False`), `return_alpha_masks=False` for the timed steps. One extra step with
  `return_alpha_masks=True` is measured because `train_dlp.py` requests the masks on the last batch of each plotting epoch.
* Measurement: 5 warm-up steps excluded (compilation, allocator, DataLoader start-up), then 30 timed steps for every
  configuration. One process per (batch size, path). Pass 1 order reference, triton, fused; pass 2 reversed.
  All 18 processes ran inside one Slurm allocation (job 3959569, one NVIDIA RTX A6000, node grogu-1-25).
* Within a batch size the three paths shared initial weights, first batch and data order (hashes identical) and the
  path check (backend, number of fused executions) passed in every process.

## Results (mean of the two passes)

| B | Metric | reference | triton | fused |
|---|---|---:|---:|---:|
| 16 | wall per step, ms | 327.6 | 281.4 | 268.0 |
| | throughput, images/s | 48.8 | 56.9 | 59.7 |
| | peak allocated MB (with masks step) | 10,365 (10,453) | 9,077 (9,170) | 8,128 (8,222) |
| 32 | wall per step, ms | 736.9 | 643.4 | 616.7 |
| | throughput, images/s | 43.4 | 49.7 | 51.9 |
| | peak allocated MB (with masks step) | 20,477 (20,663) | 17,897 (18,086) | 16,010 (16,199) |
| 64 | wall per step, ms | 1464.0 | 1279.5 | 1225.8 |
| | throughput, images/s | 43.7 | 50.0 | 52.2 |
| | peak allocated MB (with masks step) | 40,734 (41,108) | 35,575 (35,953) | 31,799 (32,170) |

Relative change (wall time; throughput; peak allocated memory):

| B | triton vs reference | fused vs reference | fused vs triton |
|---|---|---|---|
| 16 | -14.1%; +16.4%; -12.4% | -18.2%; +22.2%; -21.6% | -4.7%; +5.0%; -10.5% |
| 32 | -12.7%; +14.5%; -12.6% | -16.3%; +19.5%; -21.8% | -4.2%; +4.3%; -10.5% |
| 64 | -12.6%; +14.4%; -12.7% | -16.3%; +19.4%; -21.9% | -4.2%; +4.4%; -10.6% |

* No configuration ran out of memory. The reference at B=64 peaked at 40.7 GB allocated / 46.3 GB reserved on a device
  with 50.9 GB.
* Memory per added image (B=16 to 64): reference ~632 MB, triton ~552 MB, fused ~493 MB. By linear extrapolation of
  these measured slopes the fused path would hold about 28% more images in the same memory; this is an extrapolation,
  not a measurement.
* Pass-to-pass spread of the wall time: at most 1.2% (B=16), 0.6% (B=32), about 0 (B=64). Wall time exceeds the
  CUDA-event step time by under 2 ms; DataLoader fetch time was 0.3 to 0.7 ms per step.
* **Throughput does not rise with batch size.** Time per image increases by 12% to 14% from B=16 to B=32 for every
  path and is flat from B=32 to B=64. The cause was not investigated (clock or power behaviour was not recorded during
  the runs).

## Limitations that remain in force

The known limitations of the fused path (opt-in and strict about input support, the two unresolved Triton crop
failures in `tests/stn/live_check.py`, non-reproducible atomic `dec_objects` gradients, the unaddressed `div_rn`
flush-to-zero difference) are listed in `dlp_quality_results.md` and `dlp_fused_composite_integration.md`; nothing
in this scan changes them.

## Not established

* Any batch size above 64, other resolutions or particle counts, other GPUs or nodes, more than one seed.
* Training quality or convergence: this is a 35-step performance measurement.

## Provenance (results live outside the repository)

Results: `/grogu/user/junhong3/lpwm-work/experiments/fused-integration/scan-3959569/`

| File | sha256 |
|---|---|
| `benchmarks/stn/batch_scan.py` (recorded inside every result JSON) | `5f5b12e5ec871c84a46e0fe4ff204d5e6f2b422ee0ef6e38073078906f97a855` |
| `benchmarks/stn/summarize_scan.py` | `173425b25401893e0d9cddd7d0df22b5c0a4bb77cfd2dbc0f0c57c88e0218415` |
| `scripts/run_scan.sh` (job script) | `65adcd33c8ab976ab3059c934bb37dc71146c69ca72e2f7c3ddf45a6ca96a734` |
| `static_bair128_vgg.json` (config) | `f40efb09794db93e132a3f16f4c54d7fab68c68d1cc6135a06026b9e08a8b64d` |
| `summary.txt` | `984a477cf2672392c8364f8f482f144183f5dc14af700f6326bd6b13a89b8a0a` |
| `env.txt` (node, GPU, versions, snapshot file hashes) | `d2d2c08ccdbea7c5835884a4a6e2eb09484e98ac4fe04eeecb750aa91f04d654` |

Per-process results (`scan-b<B>-p<pass>-<path>.json`):

```
617ad69413ddb29908cb61af58d3281f2d2671ac88b9f029f97325c8b5d5c0bd  scan-b16-p1-fused
c503daef31e433524cb404a87cb7b2731b687a87cca841d216961cb1316fc67b  scan-b16-p1-reference
18264ca53e39beec95dbc620677d7d82c799fc144d36cf6cb1e623acd5fd853e  scan-b16-p1-triton
c74806c780b27391627c8ce0070a74e3e58911abe4bd6a6e5bf4c105219791e5  scan-b16-p2-fused
06264b72fd86f885bcf0485fcbd7b24aaa4aad1c3e4d7109a421c3f98c185876  scan-b16-p2-reference
26eb5abb57a6baf9d5b503ead4adc2e332411bde173eeb86962ab943a229bde7  scan-b16-p2-triton
0e4c607505974797af4954eeec77c2088688a8ab3ebb5593a451f0199243e3d2  scan-b32-p1-fused
b1cfbfe19890ed81e864b7281dd9683007c468b89bd6ed4fba28eca0c59e33be  scan-b32-p1-reference
0a9714f4893711780789260dee17df98e4ff08246a742c6c9a1eb2170270a07d  scan-b32-p1-triton
fd1af9d5bdbe23f1a600c277c67cd68d1f2569d6bd13656cc195c9317d1fade0  scan-b32-p2-fused
61c03d2c4610b7fd3f5844fb4cc17920bed293f64861186dbdad52b6be81ac45  scan-b32-p2-reference
d3077ec5e89316b3ecd658d31635b12acfec0b74904cec61bfdc0d5ef3715eed  scan-b32-p2-triton
82538d46a3a41820bab74cff3a594aa4016af4cb007025c9db3e30ea1ab2925e  scan-b64-p1-fused
f362a290498a9d4359a57085475deb8107012567e17fcff272b67197a212b6c0  scan-b64-p1-reference
1f1a5ee97aa975c221c7f166e83878e6f042cb87af5d08f7585e788c8eff556a  scan-b64-p1-triton
b8aab86288d8ba9034d40aed85673479adba49b1c50005177fe2766f79c73fed  scan-b64-p2-fused
2843910a56e79492cc3bcef9be8fc797194124b255cbaca9e2bbce6955bad58f  scan-b64-p2-reference
6de6828d60da81831be202643978188c78257097a3f4b20f1ddf2066e850bfb7  scan-b64-p2-triton
```

Reproduce (inside a Slurm allocation on an A6000, snapshot checked out at the commit above):

```bash
TORCH_HOME=<dir>/torch python benchmarks/stn/batch_scan.py --repo-root . --config <static bair cfg> \
    --path reference|triton|fused --batch-size 16|32|64 --warmup-steps 5 --timed-steps 30 --out <json>   # cwd needs eval/lpips/vgg.pth
python benchmarks/stn/summarize_scan.py <dir with the JSONs and env.txt>
```
