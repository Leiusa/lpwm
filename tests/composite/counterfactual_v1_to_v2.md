# Backward counterfactual: does frozen policy v1 accept the already-shipped Triton paste?

No fused-backward data was used in constructing this result.

  R      = reference.stn_paste + PyTorch reference composite  (envelope was built from R)
  T_old  = already-shipped, previously accepted triton_backend.stn_paste + the SAME composite

## Variant A -- sampler backward only, identical dL/dsample fed to both
grad          fails  max ratio   max rel-L2
dec_objects       0      0.000    0.000e+00
z_kp              5      4.682    1.523e-06
z_scale           9      1.742    3.301e-06
obj_on            0      0.000    0.000e+00
z_depth           0      0.000    0.000e+00
first: bair_bs1/rgb_only idx=120 ref=-8.963097e+00 T_old=-8.962788e+00
       lo=-8.963131e+00 hi=-8.963077e+00 delta=9.963e-05 ratio=1.90

## Variant B -- complete old unfused chain, each running its own fwd+bwd
grad          fails  max ratio   max rel-L2
dec_objects       0      0.258    8.280e-07
z_kp             22     29.701    8.509e-06
z_scale          15      9.794    7.338e-06
obj_on            0      0.329    1.514e-07
z_depth           1      0.147    1.300e-05
first: bair_bs1/rgb_only idx=120 ref=-8.963097e+00 T_old=-8.962345e+00
       lo=-8.963131e+00 hi=-8.963077e+00 delta=9.963e-05 ratio=6.35

## Verdict
T_old FAILS policy v1, in the same z_kp/z_scale pattern and at the same magnitude
as the new fused backward (z_kp max ratio 29.701 for T_old vs 29.706 for the fused
kernel; same first failing case and index). dec_objects and obj_on pass for both.
z_depth's rel-L2 failure is also reproduced by T_old (1.300e-05 vs 1.35e-05).

Policy v1 is therefore incomplete: it rejects a pre-existing, independently
accepted implementation. NOT amended. Any policy v2 must be derived only from R
and T_old, never from the fused-kernel errors.

Note Variant A already fails (5 z_kp, 9 z_scale) with an IDENTICAL dL/dsample fed
to both sides, so the disagreement originates in the paste backward itself, not in
the composite feed.

## Fused-kernel status (recorded, not used above)
FP32 restored; both FP64 diagnostics reverted and kept only as
scratch/composite_backward.fp64_experiment.py. Sampler-only 3-way A/B/C: PASS.
Composite dL/dsample vs reference: PASS (alpha 1.43e-06, rgb 5.96e-08).
