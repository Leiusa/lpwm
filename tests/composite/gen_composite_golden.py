#!/usr/bin/env python
"""
Freeze the PyTorch oracle's answers for the composite chain.

Runs :func:`lpwm_stn.composite_reference.composite_reference` -- PyTorch only,
explicitly on the reference paste -- over the shared case list and records
forward outputs and input gradients.

Re-run this ONLY to add cases. Regenerating it because a fused kernel disagrees
would destroy the only thing that makes the comparison meaningful.

    python tests/composite/gen_composite_golden.py --device cuda
"""

import argparse, os, sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lpwm_stn.composite_reference import composite_reference   # noqa: E402
from tests.stn import golden                                    # noqa: E402
from composite_cases import CASES, GRAD_WRT, make_inputs        # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
OUT_NAMES = ("alpha_masks", "bg_mask", "dec_objects_trans")


def fixture_path(device):
    return os.path.join(FIXTURE_DIR, f"composite_golden_{device}.pt")


def cotangent(t, tag):
    """Fixed upstream gradient, seeded off a stable hash of the output's name."""
    import zlib
    g = torch.Generator(device=t.device)
    g.manual_seed(0x5757 + (zlib.crc32(tag.encode()) & 0xFFFF))
    return torch.rand(t.shape, generator=g, device=t.device, dtype=t.dtype)


def record(name, kwargs, device):
    ins = make_inputs(device=device, requires_grad=True, **kwargs)
    outs = composite_reference(**ins)
    rec = {"kwargs": kwargs, "outputs": {}, "grads": {}}
    for out_name, t in zip(OUT_NAMES, outs):
        rec["outputs"][out_name] = None if t is None else golden.describe(t)

    live = [(n, t) for n, t in zip(OUT_NAMES, outs) if t is not None]
    cots = [cotangent(t, f"{name}/{n}") for n, t in live]
    rec["cotangent_sha256"] = {n: golden.describe(c)["sha256"] for (n, _), c in zip(live, cots)}
    wrt_names = [k for k in GRAD_WRT if ins.get(k) is not None]
    grads = torch.autograd.grad([t for _, t in live], [ins[k] for k in wrt_names],
                                grad_outputs=cots, allow_unused=True)
    for k, g in zip(wrt_names, grads):
        rec["grads"][k] = None if g is None else golden.describe(g)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    path = fixture_path(a.device)
    if os.path.exists(path) and not a.force:
        raise SystemExit(f"{path} exists; --force only to deliberately re-pin the oracle")

    payload = {"version": 1,
               "meta": {"source": "lpwm_stn.composite_reference (PyTorch oracle, reference paste)",
                        "torch": torch.__version__, "device": a.device,
                        "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
                        "git_commit": golden.git_commit()},
               "cases": {}}
    for name, kwargs in CASES:
        payload["cases"][name] = record(name, kwargs, a.device)
        print(f"  recorded {name}")
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    torch.save(payload, path)
    print(f"\nwrote {len(payload['cases'])} cases -> {path} ({os.path.getsize(path)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
