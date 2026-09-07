#!/usr/bin/env python
"""
Standalone fused Triton composite FORWARD against the frozen policy.

The tolerance is read from ``fixtures/frozen_forward_policy.json``, which was
recorded BEFORE the kernel existed, from an FP64 oracle on the composite half
plus legal particle-permutation reorderings. The test refuses to run if that
file is missing rather than inventing a bound, and it re-asserts the frozen
values so a later edit to the file is a visible test failure, not a silent
loosening.

Both maximum absolute and maximum relative error are reported for every output,
including where relative error is large: on near-zero mask pixels the combined
bound's atol term is what carries the comparison, and that should be visible
rather than hidden behind a pass.

    python tests/composite/test_composite_triton_forward.py
"""

import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from lpwm_stn.composite_reference import composite_reference           # noqa: E402
from lpwm_stn.composite_triton import (                                # noqa: E402
    can_use_triton_composite, composite_forward_triton)
from composite_cases import CASES, make_inputs                         # noqa: E402

POLICY_PATH = os.path.join(_HERE, "fixtures", "frozen_forward_policy.json")
OUT_NAMES = ("alpha_masks", "bg_mask", "dec_objects_trans")

#: the values frozen on 2026-09-07T03:08:04Z, restated here so that editing the
#: policy file to make a kernel pass fails this test instead of going unnoticed
EXPECTED_ATOL = 2e-6
EXPECTED_RTOL = 1e-5


def load_policy():
    if not os.path.exists(POLICY_PATH):
        raise FileNotFoundError(
            f"frozen forward policy missing at {POLICY_PATH}. It is recorded before the "
            f"kernel exists and must not be regenerated to fit one.")
    p = json.load(open(POLICY_PATH))
    assert p["FROZEN_forward_atol"] == EXPECTED_ATOL, (
        f"frozen atol changed: {p['FROZEN_forward_atol']} != {EXPECTED_ATOL}")
    assert p["FROZEN_forward_rtol"] == EXPECTED_RTOL, (
        f"frozen rtol changed: {p['FROZEN_forward_rtol']} != {EXPECTED_RTOL}")
    assert p.get("frozen_before_kernel_exists") is True
    return p


def compare(label, ref, got, atol, rtol):
    """Returns (ok, max_abs, max_rel). Bound: |diff| <= atol + rtol*|ref|."""
    if ref is None or got is None:
        return (ref is None) == (got is None), 0.0, 0.0
    if ref.shape != got.shape:
        raise AssertionError(f"{label}: shape {tuple(got.shape)} != {tuple(ref.shape)}")
    if ref.dtype != got.dtype:
        raise AssertionError(f"{label}: dtype {got.dtype} != {ref.dtype}")
    r = ref.float()
    d = (got.float() - r).abs()
    max_abs = float(d.max())
    max_rel = float((d / (r.abs() + 1e-30)).max())
    ok = bool((d <= atol + rtol * r.abs()).all())
    return ok, max_abs, max_rel


def main():
    if not torch.cuda.is_available():
        print("SKIPPED: the fused composite kernel is CUDA-only")
        return 0
    policy = load_policy()
    atol, rtol = policy["FROZEN_forward_atol"], policy["FROZEN_forward_rtol"]
    print(f"frozen policy {policy['frozen_at_utc']}: atol={atol:g} rtol={rtol:g}")
    print(f"  measured envelope: {policy['measured_worst_abs']:.3e} abs / "
          f"{policy['measured_worst_rel']:.3e} rel\n")
    print(f"{'case':<18} {'output':<20} {'max_abs':>11} {'max_rel':>11} {'ok':>5}")

    failures = []
    for name, kwargs in CASES:
        ins = make_inputs(device="cuda", **kwargs)
        if not can_use_triton_composite(ins["dec_objects"], ins["z_kp"], ins["z_scale"],
                                        ins["obj_on"], ins["z_depth"], ins["img_size"]):
            failures.append((name, "gate refused the supported case"))
            print(f"{name:<18} GATE REFUSED")
            continue
        ref = composite_reference(**ins)
        got = composite_forward_triton(
            ins["dec_objects"], ins["z_kp"], ins["z_scale"], ins["obj_on"], ins["z_depth"],
            ins["img_size"], return_alpha_masks=ins["return_alpha_masks"])
        for out_name, r, g in zip(OUT_NAMES, ref, got):
            ok, ma, mr = compare(f"{name}/{out_name}", r, g, atol, rtol)
            shown_a = "None" if r is None else f"{ma:.3e}"
            shown_r = "None" if r is None else f"{mr:.3e}"
            print(f"{name:<18} {out_name:<20} {shown_a:>11} {shown_r:>11} {str(ok):>5}")
            if not ok:
                failures.append((f"{name}/{out_name}", f"max_abs={ma:.3e} max_rel={mr:.3e}"))

    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} failure(s)")
    for f, why in failures:
        print(f"  {f}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
