#!/usr/bin/env python
"""
Record the frozen baseline's answers for every STN case.

Runs :mod:`lpwm_stn._baseline` -- the untouched pre-optimization code -- over
the shared case list and writes ``tests/stn/fixtures/stn_golden_<device>.pt``.
Re-run this ONLY to add cases; regenerating it to make a failing test pass
would defeat its purpose.

    python tests/stn/gen_golden.py                # cpu, small correctness shapes
    python tests/stn/gen_golden.py --device cuda  # once a GPU box is available
    python tests/stn/gen_golden.py --scope all    # include full config shapes
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lpwm_stn import _baseline
from tests.stn import golden
from tests.stn.stn_cases import BENCH_CASES, CORRECTNESS_CASES

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture_path(device, scope="all"):
    suffix = "" if scope == "all" else f"_{scope}"
    return os.path.join(FIXTURE_DIR, f"stn_golden_{device}{suffix}.pt")


def record_case(case, impl, device, dtype):
    """Run one case on ``impl`` and reduce forward + gradients to pinned records."""
    ins = case.inputs(device=device, dtype=dtype)
    out = case.run(impl, ins)
    rec = {"op": case.op, "workload": case.workload.name, "grad_wrt": case.grad_wrt,
           "out": golden.describe(out)}
    if case.needs_grad:
        cot = case.cotangent(out)
        wrt = [ins[k] for k in case.grad_wrt]
        grads = torch.autograd.grad(out, wrt, grad_outputs=cot, retain_graph=False, allow_unused=False)
        rec["cotangent_sha256"] = golden.describe(cot)["sha256"]
        rec["grads"] = {k: golden.describe(g) for k, g in zip(case.grad_wrt, grads)}
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scope", default="correctness", choices=("all", "correctness", "bench"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing fixture file")
    args = ap.parse_args(argv)

    cases = {"all": CORRECTNESS_CASES + BENCH_CASES,
             "correctness": CORRECTNESS_CASES,
             "bench": BENCH_CASES}[args.scope]

    path = args.out or fixture_path(args.device, args.scope)
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"{path} already exists; pass --force only if you are deliberately re-pinning the baseline")

    torch.manual_seed(0)
    payload = {
        "version": golden.FIXTURE_VERSION,
        "meta": {
            "source": "lpwm_stn._baseline (frozen copy of utils/util_func.py + modules/modules.py @ 4cf53c4)",
            "torch": torch.__version__,
            "device": args.device,
            "dtype": "torch.float32",
            "git_commit": golden.git_commit(),
            "scope": args.scope,
        },
        "cases": {},
    }
    for case in cases:
        payload["cases"][case.name] = record_case(case, _baseline, args.device, torch.float32)
        print(f"  recorded {case.name}")

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    torch.save(payload, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\nwrote {len(payload['cases'])} cases -> {path} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
