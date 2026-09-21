# Quality-validation protocol (fixed BEFORE any run of this stage)

Purpose: check whether the three paths behave alike over a longer, multi-seed BAIR training. It is NOT a proof of
quality equivalence and no threshold is applied to any result.

* Code under test: commit `b2cfa83b05d47edb254748fdd3b3c2223519103d` (clean snapshot `tree-b2cfa83`).
* Paths: `reference` (stn_backend=reference), `triton` (stn_backend=triton), `fused` (triton + fused_composite=true).
* Data: the existing fixed-prefix BAIR subset, 400 training episodes and 40 internal-validation episodes
  (manifest aggregate sha256 `64917d7eeb7da3826e9d1524b6c3f2d70ec0ea0fb5f79d4722ba7a8fc086f431`); not random, not the
  official validation set. 128x128, 90 particles, original vgg loss and all other settings of
  `static_bair128_vgg.json`; only `batch_size=16`. No hyper-parameter is changed.
* Seeds (fixed in advance): **0, 1, 2**. Within a seed the three paths share initial weights, data order and sampling
  noise (all set by the seed); across seeds they differ. One Slurm job (single A6000) per seed; the three paths of a
  seed are trained one after the other inside that job. Run orders (balanced across seeds): seed 0
  reference,triton,fused; seed 1 triton,fused,reference; seed 2 fused,reference,triton.
* Training: 8 epochs per path per seed through `train_dlp.py` (750 steps per epoch).
* Reported checkpoint: the FINAL checkpoint after epoch 8, for every path and seed. No epoch, checkpoint or seed is
  selected by outcome. Per-epoch curves (training loss/PSNR, validation loss) are kept from the training logs.
* Evaluation path: every final checkpoint is loaded into a model on the REFERENCE path and evaluated
  deterministically on ALL 1,200 validation images: MSE over all pixels and PSNR = -10 log10(MSE). The validation
  loss is the one `train_dlp.py` logs after epoch 8 (original vgg loss on the same validation set).
* Comparisons (each per seed, then mean, sample standard deviation and min-max over the 3 seeds):
  (1) triton minus reference, (2) fused minus triton, for validation loss, MSE and PSNR. fused minus reference is
  shown as an additional line. Each path's own spread over seeds is shown as the scale to read differences against.
* Training-loss offset relative to the reference is reported separately, per epoch and seed.
* Not used as a criterion: whether final weights are equal or close; a small mean difference is not stated as
  equivalence. Non-finite values, a lost path, or an unrecoverable run stop the stage and are reported.
* Budget: at most 3 single-GPU jobs of <= 2 h each. Expected per job ~96 min from measured epoch walls
  (reference 249 s, triton 214 s, fused 204 s per epoch, plus validation and evaluation overhead).
