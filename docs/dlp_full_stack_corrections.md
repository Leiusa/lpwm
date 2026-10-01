# Corrections to the full-stack report (revision 1 -> revision 2)

Written 2026-09-23 after a read-only audit of the server records. Nothing on the server was modified or overwritten;
the original `train_report.json` files (whose headers are wrong) are kept as they are. Machine-readable versions:
`metadata.json` (hosts, GPUs, code versions, checkpoint sha256; null = unknown), `quality_corrected.json`,
`gpu_ledger.json`, produced by `benchmarks/stn/summarize_full_stack_quality_corrected.py`.

## What was wrong

| Revision 1 | Correction |
|---|---|
| D checkpoint "evaluated through the same reference path" as A | That evaluation (3) used reference STN, no fusion, **channels-last on** (the "reference" eval config does not override `particle_dec_channels_last`). Main comparison is now (1) A checkpoint on the original path vs (2) D checkpoint on the full optimized path. (3) is an auxiliary diagnostic; (4) D checkpoint on the pure original path was never run. Values are the same to within 0.001%. |
| Deployment cross-check "no further difference" | It compared (2) with (3): supports Triton STN + fusion at inference, not channels-last at inference. |
| "No consistent direction" | Mean D-vs-A: full-val MSE +2.54% (std 3.22, range -1.17..+4.66), PSNR -0.108 dB (std 0.138, range -0.198..+0.051); signs differ by seed; fixed-16 PSNR negative in all seeds. Three paired seeds cannot confirm stable degradation or quality equivalence; sign changes are not attributed to noise. |
| Every job excluded `grogu-4-13`; RTX A6000 throughout | Quality-training script lacked `--exclude`; seeds 0 and 1 trained on grogu-4-13 = RTX 3080 Ti (12 GB). Seed 2 trained on an A6000 after a restart. Evaluations ran on an A6000. |
| Pooled training speed -18.9%, reserved-memory remarks | Withdrawn (mixed GPU models). Per GPU, A->D step time: 3080 Ti -14.6% / -14.8%; A6000 -26.8%. Speed and memory gains come only from the A6000 sweep (job 3967207). |
| ~4.0 of 4.0 GPU-hours | 4.36-4.66 GPU-hours (see ledger). Includes an aborted, previously uncounted seed-2 attempt. |
| z_depth crash hit 2 of 3 seeds | All three seeds' evaluation steps crashed; all three recovered from saved checkpoints. |
| Recovered-report metadata | Server headers hold seed = config path, epochs/order/host missing. Local copies had been hand-patched (host guessed). Hosts are now `training_host` / `eval_host` with unknowns marked. |

## Hosts, GPUs, code

| seed | training job | training_host / GPU | eval job | eval_host / GPU |
|---|---|---|---|---|
| 0 | 3967365 | grogu-4-13 / RTX 3080 Ti | 3967495 | grogu-1-25 (observed in squeue, not stored) / A6000 |
| 1 | 3967366 | grogu-4-13 / RTX 3080 Ti | 3967495 | grogu-1-25 (observed) / A6000 |
| 2 | 3967367 restart | grogu-1-25 / A6000 (first attempt: unknown, aborted) | 3967496 | unknown / A6000 |

Training and the performance sweep used `modules.py` `33c965d0` (recorded at job start); recovery evaluations used the fixed
`d65a7667` (deployed 2026-09-22 21:59:45 EDT; inferred from file times, hash not stored in the reports). The evaluation
processes never imported `train_dlp.py`, so `cudnn.deterministic` was most likely False there (training: True); unrecorded.

## GPU ledger (cap 4.0 GPU-h; times from file timestamps, sacct unavailable)

| job | what | GPU | minutes | outcome |
|---|---|---|---|---|
| 3967054 | perf sweep | 3080 Ti | 3.7 | discarded (wrong GPU) |
| 3967207 | perf sweep, 24 processes | A6000 | 13.4 | valid |
| 3967365 | seed 0 training | 3080 Ti | 62.1 | training valid |
| 3967366 | seed 1 training | 3080 Ti | 61.5 | training valid |
| 3967367 #1 | seed 2 first attempt | unknown | 22.4-38.3 | aborted, unused |
| 3967367 #2 | seed 2 restart | A6000 | 61.3 | training valid |
| 3967495 | recovery eval seeds 0,1 | A6000 | 1-2 (est.) | ok |
| 3967496 | recovery eval seed 2 | A6000 | 1-2 (est.) | ok |
| 3967532 | OBJ3D pilot | A6000 | 35.5 | cancelled at limit, no result |
| | **total** | | **262-280** | **4.36-4.66 GPU-h, over the cap by 0.36-0.66 h** |

By GPU: 3080 Ti about 2.1 h, A6000 about 1.9 h, unknown 0.4-0.6 h.

## Not affected
The A6000 performance sweep, training completeness and pairing checks, the z_depth analysis, and the OBJ3D findings.
