# Opt-in channels_last execution of the DLP particle decoder

`particle_dec_channels_last` (config key, `DLP` / `DLPDecoder` argument, `ObjectDecoderCNN(channels_last=...)`) runs the
particle decoder CNN with channels_last conv weights and activations. **Default off; only `particle_dec` is affected.**

## What it does

* Conv weights of `particle_dec.cnn` are converted with `Module.to(memory_format=channels_last)` after `init_weights()`
  (RNG stream and initial values unchanged); the CNN input is converted to channels_last; the CNN output is converted back
  to standard contiguous NCHW immediately, so the rest of `ObjectDecoderCNN.forward` and everything downstream is unchanged.
* `composite_fused` still receives standard contiguous fp32 `dec_objects`; its support check is untouched (a channels_last
  strided `dec_objects` is still rejected). Both conversions and their backward are inside every measurement below.
* Unsupported combination (context-conditioned `ObjectDecoderCNNFILM`) raises `ValueError`.
* `train_dlp.py` reads `particle_dec_channels_last`, logs it at startup and, in the first step, raises `RuntimeError` unless the
  channels_last path ran exactly as requested (`channels_last_calls`, conv weight layout). The per-epoch `[step-timing]` line
  also carries `peak_reserved_mb` and `reserved_now_mb`.
* TF32, `cudnn.deterministic` and `cudnn.benchmark` are not touched (production: benchmark False, deterministic True, cuDNN
  conv TF32 allowed by the torch default, matmul TF32 off).

## What it is not: a same-precision optimisation

With cuDNN conv TF32 allowed (the default), two of the ten `particle_dec` convolutions (8x8 input, 256->128 and 256->256
upsample) run an **fp32 FFT** algorithm in the forward pass under NCHW (about 40 ms per step); the other eight use TF32.
Under channels_last those two layers use a TF32 implicit-GEMM algorithm instead. So the switch **changes the actual computation
precision of two layers** in addition to the layout. Isolated-module error against an fp64 evaluation: output 1.7e-4
(NCHW) vs 2.0e-4 (channels_last); with cuDNN conv TF32 disabled the layout alone changes the output by 3.6e-7. The speed
gain also exists with TF32 disabled (algorithm/layout selection: whole step -6.2%), it does not depend on TF32, but in the
production setting it is bundled with that precision change.

## Wiring checks (job 3965722, A6000, snapshot b2cfa83 + the diff)

* `tests/composite/test_particle_dec_channels_last.py`: 6/6 pass (defaults, layout evidence, no new parameters and identical
  initial values, strict cross-loading, module contract, composite input contiguity and unchanged support check, ValueError).
  `test_decoder_fused_integration.py`: 6/6 pass.
* Default-off regression vs b2cfa83, two real BAIR batches, fixed RNG, no optimizer step: initial weights and batches
  identical, loss and reconstruction **bit-equal**, per-parameter gradient relative L2 median 7e-7 / 6e-7 (the known
  run-to-run floor of the atomic `dec_objects` gradient).
* Switch on vs off: loss relative difference 3.8e-6 / 3.7e-6, reconstruction relative L2 3.6e-5 / 3.5e-5, per-parameter
  gradient relative L2 median 0.73% / 0.79% (for scale: cuDNN TF32 on vs off in the unchanged production layout gives 3-7%).
  All finite. The single tensor carrying almost all gradient norm (encoder objectness projection) makes all-parameter gradient
  norms uninformative; use per-parameter quantiles.

## Short training validation (fusion on, real BAIR subset, 128x128, B=16, original vgg loss)

Seeds 0 and 1, 3 epochs each, switch off / on trained inside one allocation per seed (seed 0 off->on, seed 1 on->off) from
identical initial weights (hash), first batch, data order and sampling noise. Final checkpoints evaluated with the same
evaluation configuration (reference-path STN, NCHW particle decoder) on the whole 1,200-image validation set. Jobs
3965727 / 3965728, both on node grogu-2-5 (each with its own A6000 while running concurrently, so absolute times may include
shared-node effects; comparisons are paired within an allocation).

| | switch off | switch on | on vs off |
|---|---|---|---|
| training step (CUDA-event median, all epochs, both seeds) | 274.9 ms (272.8-277.6) | 246.3 ms (242.9-248.5) | -10.4% (per epoch -9.6% to -10.9%) |
| step throughput | 58.2 img/s | 65.0 img/s | +11.7% |
| epoch wall clock (incl. data loading and validation) | 207.9 s | 186.6 s | -10.3% |
| peak allocated | 8,238-8,337 MB | 8,241-8,343 MB | +3 to +9 MB, same epoch-to-epoch pattern |
| peak reserved | 11,205-12,113 MB | 8,286-10,482 MB | not stable, see below |

* Allocated memory is stable and equal within 9 MB. Reserved memory is allocator-history dependent in both settings: off
  varied 11.2-12.1 GB across epochs and seeds, on 8.3-8.4 GB (seed 0) and 10.1-10.5 GB (seed 1). It was never higher with the
  switch on in these runs, but a lower reserved value must not be read as a property of the switch (an earlier short harness
  process showed the opposite ordering).
* Final checkpoint (epoch 3), whole validation set, on minus off, per seed (seed 0 / seed 1): validation loss -0.035 / +0.005
  (-0.27% / +0.04%), MSE -0.5% / -2.1%, PSNR +0.020 / +0.093 dB, fixed 16-image batch PSNR +0.14 / +0.20 dB, train loss
  -0.7% / -0.3%. Absolute PSNR 22.26 / 22.29 dB (seed 0) and 22.03 / 22.12 dB (seed 1). Per-epoch validation losses differ
  by -0.24 .. +0.18 with both signs. Reconstruction grid (target / off / on, same 16 validation images): `reconstructions.png`
  in the seed directories; the two rows are visually indistinguishable at this training stage.
* Final weights of the two layouts differ by relative L2 0.105 / 0.101, the same scale as between the reference, Triton and
  fused paths in the earlier 3-epoch BAIR comparison (0.098-0.103). No same-path repeat control exists here.

## Conclusion and limits

In this configuration (two seeds, 3 epochs, one dataset subset) enabling the switch made the training step about 10% faster
with unchanged allocated memory, and no degradation of validation loss, MSE or PSNR was observed. This is not evidence of quality
equivalence, not a same-precision result (two layers change from fp32 FFT to TF32), and not a convergence claim (3 epochs). Reserved
memory behaviour is not established. Not covered: other resolutions, other decoders (FILM particle decoder is rejected), AMP,
longer training, other GPU models. The two known crop live_check failures and the other limits in `dlp_quality_results.md` are unchanged.

## Provenance (results live outside the repository)

Snapshot `experiments/fused-integration/tree-cl` = commit b2cfa83 + this diff (working tree, uncommitted at the time).
File hashes (sha256): `models.py` 04d1e37b..., `modules/modules.py` 33c965d0..., `train_dlp.py` dd2e84ac...,
`tests/composite/test_particle_dec_channels_last.py` db21a0e7..., `benchmarks/stn/layout_quality_run.py` 632d33a6...
(full values in each result directory's `scripts/tree-cl.sha256` and `env.txt`).

* Preflight: `clpre-3965722/` (wiring tests, regression, on/off comparison; `layout_integration_check.py` 0da52bd0...).
* Training: `cltrain-seed0-3965727/`, `cltrain-seed1-3965728/` (`train_report.json`, per-run logs, `reconstructions.png`).
* Earlier layout / TF32 study that motivated this: `layout-3964197/`, `layout-num2-3964220/`, `layout-num3-3964221/`.
* Summary: `python benchmarks/stn/summarize_layout_quality.py OUT.json <seed dirs>`.
