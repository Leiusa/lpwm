#!/usr/bin/env python
"""
Tests for the composite oracle (paste + alpha/depth composite + reduction).

Three things are pinned, and they are different claims:

1. the oracle still reproduces its frozen fixtures, forward and backward;
2. the oracle still matches the LIVE decoder path in modules/modules.py -- an
   oracle that has drifted from the model is worthless no matter how
   self-consistent it is;
3. the oracle is backend-independent -- it calls the reference paste explicitly,
   so switching the registry to Triton must not change a single bit. If this
   fails, every fixture generated from it is suspect.

Comparison policy, established from the ORACLE'S OWN behaviour before any kernel
exists (the design doc requires the tolerance be set this way round, never from
whatever a kernel happens to produce):

* forward outputs -- bit-exact. The reference is deterministic here.
* grad[z_kp], grad[z_scale], grad[obj_on], grad[z_depth] -- bit-exact. Measured
  identical across repeated runs.
* grad[dec_objects] -- RELATIVE tolerance. It flows back through stn_paste into
  `grid_sampler_2d_backward`, which scatters with atomicAdd, so its accumulation
  order is non-deterministic on CUDA and the reference does not reproduce itself
  bit-for-bit. Measured over 5 repeats of all 12 cases:

      worst absolute spread : 2.48e-05   (bair_bs1, K=90)
      worst relative spread : 7.99e-07

  Absolute tolerance is the wrong instrument here: gradient magnitude scales with
  the number of overlapping particles, so a fixed absolute bound that fits balls
  is exceeded by bair. The relative spread is stable at ~1e-7..1e-6 across every
  case, so GRAD_RTOL is relative, set to 1e-5 -- more than 12x the worst measured
  spread of the reference against itself.

    python tests/composite/test_composite_reference.py --device cuda
"""

import argparse, os, sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lpwm_stn                                                  # noqa: E402
from lpwm_stn.composite_reference import composite_reference     # noqa: E402
from modules.modules import DLPDecoder                           # noqa: E402
from tests.stn import golden                                     # noqa: E402
from composite_cases import CASES, GRAD_WRT, make_inputs         # noqa: E402
from gen_composite_golden import OUT_NAMES, cotangent, fixture_path  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: gradients the reference reproduces exactly; anything else here is a real bug
DETERMINISTIC_GRADS = ("z_kp", "z_scale", "obj_on", "z_depth")

#: relative bound for grad[dec_objects]; see the module docstring for its derivation
GRAD_RTOL = 1e-5


def _fixture(device):
    path = fixture_path(device)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no composite fixture for '{device}': generate with "
                                f"tests/composite/gen_composite_golden.py --device {device}")
    return torch.load(path, weights_only=False)


class _Fake:
    def __init__(self, img):
        self.feature_map_size = img


def _compare_relative(label, expected, actual, rtol):
    """Relative comparison for the one gradient the reference cannot reproduce.

    Scaled by the expected tensor's own magnitude, so the bound means the same
    thing at K=12 and at K=90.
    """
    got = actual.detach()
    assert tuple(got.shape) == tuple(expected["shape"]), (
        f"{label}: shape {tuple(got.shape)} != {tuple(expected['shape'])}")
    assert str(got.dtype) == expected["dtype"], f"{label}: dtype {got.dtype} != {expected['dtype']}"
    scale = max(abs(expected["absmax"]), 1e-12)
    if "full" in expected:
        ref = expected["full"].to(got.device).float()
        worst = float((got.float() - ref).abs().max())
    else:
        # no stored tensor: fall back to the pinned reduction, scaled by element count
        got_sum = float(got.reshape(-1).double().sum())
        worst = abs(got_sum - expected["sum_f64"]) / max(expected["numel"] ** 0.5, 1.0)
    rel = worst / scale
    assert rel <= rtol, (
        f"{label}: relative deviation {rel:.3e} > rtol {rtol:.3e} "
        f"(max|diff|={worst:.3e}, scale={scale:.3e})")


def check_case_against_fixture(name, kwargs, expected, device):
    ins = make_inputs(device=device, requires_grad=True, **kwargs)
    outs = composite_reference(**ins)
    for out_name, t in zip(OUT_NAMES, outs):
        exp = expected["outputs"][out_name]
        if exp is None:
            assert t is None, f"{name}/{out_name}: expected None, got a tensor"
            continue
        golden.compare(f"{name}/{out_name}", exp, t)

    live = [(n, t) for n, t in zip(OUT_NAMES, outs) if t is not None]
    cots = [cotangent(t, f"{name}/{n}") for n, t in live]
    for (n, _), c in zip(live, cots):
        assert golden.describe(c)["sha256"] == expected["cotangent_sha256"][n], (
            f"{name}/{n}: cotangent differs from the one the fixture was recorded with")
    wrt = [k for k in GRAD_WRT if ins.get(k) is not None]
    grads = torch.autograd.grad([t for _, t in live], [ins[k] for k in wrt],
                                grad_outputs=cots, allow_unused=True)
    for k, g in zip(wrt, grads):
        exp = expected["grads"][k]
        if exp is None:
            assert g is None, f"{name}/grad[{k}]: expected None"
            continue
        if k in DETERMINISTIC_GRADS:
            golden.compare(f"{name}/grad[{k}]", exp, g)
        else:
            _compare_relative(f"{name}/grad[{k}]", exp, g, GRAD_RTOL)


def check_case_against_live_decoder(name, kwargs, device):
    ins = make_inputs(device=device, **kwargs)
    fake = _Fake(ins["img_size"])
    trans = DLPDecoder.translate_patches(fake, ins["z_kp"], ins["dec_objects"], ins["z_scale"])
    a_obj, rgb_obj = torch.split(trans, [1, trans.shape[2] - 1], dim=2)
    live = DLPDecoder.get_objects_alpha_rgb_with_depth(
        fake, a_obj, rgb_obj, obj_on=ins["obj_on"], z_depth=ins["z_depth"],
        return_alpha_masks=ins["return_alpha_masks"])
    ref = composite_reference(**ins)
    for out_name, lv, rv in zip(OUT_NAMES, live, ref):
        if lv is None or rv is None:
            assert lv is None and rv is None, f"{name}/{out_name}: None mismatch"
            continue
        assert torch.equal(lv, rv), (
            f"{name}/{out_name}: oracle has drifted from the live decoder, "
            f"max|diff|={float((lv - rv).abs().max()):.3e}")


def check_case_backend_independent(name, kwargs, device):
    ins = make_inputs(device=device, **kwargs)
    with lpwm_stn.use_backend("reference"):
        a = composite_reference(**ins)
    try:
        ctx = lpwm_stn.use_backend("triton")
    except KeyError:
        return "skipped: triton backend not registered"
    with ctx:
        b = composite_reference(**ins)
    for out_name, x, y in zip(OUT_NAMES, a, b):
        if x is None or y is None:
            assert x is None and y is None, f"{name}/{out_name}: None mismatch across backends"
            continue
        assert torch.equal(x, y), (
            f"{name}/{out_name}: oracle changed when the registry switched to Triton -- "
            f"it is not calling the reference paste explicitly")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args(argv)
    payload = _fixture(args.device)
    print(f"fixture: {os.path.relpath(fixture_path(args.device), _ROOT)} "
          f"(torch {payload['meta']['torch']}, {payload['meta'].get('gpu')})")
    print(f"running torch {torch.__version__} on {args.device}, {len(CASES)} cases\n")

    failures, skipped = [], []
    for label, fn in (("fixture", check_case_against_fixture),
                      ("live-decoder", check_case_against_live_decoder),
                      ("backend-independence", check_case_backend_independent)):
        print(f"[{label}]")
        for name, kwargs in CASES:
            try:
                if label == "fixture":
                    fn(name, kwargs, payload["cases"][name], args.device)
                else:
                    note = fn(name, kwargs, args.device)
                    if note:
                        skipped.append((f"{label} {name}", note))
            except Exception as exc:                              # noqa: BLE001
                failures.append((f"{label} {name}", exc))
                print(f"  FAIL {name}: {exc}")
        print(f"  {len(CASES) - sum(1 for f, _ in failures if f.startswith(label))}"
              f"/{len(CASES)} ok")
    tail = f", {len(skipped)} skipped" if skipped else ""
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} failure(s){tail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
