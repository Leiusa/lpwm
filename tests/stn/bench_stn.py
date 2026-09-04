#!/usr/bin/env python
"""
Baseline timings and memory for the STN stage.

Stage 1 does not make anything faster; this exists to record where the time and
the memory go *before* any kernel work, so stage 2 has a number to beat and a
regression check to run. The config presets currently model retained-particle
calls; confirm the attribute crop's ``n_kp_prior`` shape in a representative
GPU profile before treating results as the full-model baseline.

Reported per case:
    fwd      forward only, ms
    fwd+bwd  forward plus backward through the case's gradient contract, ms
    fwd MB   peak allocated bytes with inputs live during forward
    train MB peak allocated bytes with inputs live during fwd+bwd
    inflate  for the crop cases, how much larger the ``repeat``-ed input is
             than the image it came from -- the allocation a fused kernel removes

    python tests/stn/bench_stn.py                     # cpu smoke test, small shapes
    python tests/stn/bench_stn.py --scope bench       # full config shapes
    python tests/stn/bench_stn.py --scope prior       # attribute-encoder n_kp_prior crop shapes
    python tests/stn/bench_stn.py --device cuda --backend triton
    python tests/stn/bench_stn.py --compare reference triton
"""

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lpwm_stn
from tests.stn.stn_cases import BENCH_CASES, CORRECTNESS_CASES, PRIOR_BENCH_CASES


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def _time(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples)


def bench_case(case, device, iters, warmup, backward=True):
    ins = case.inputs(device=device, requires_grad=case.needs_grad and backward)

    def forward_only():
        with torch.no_grad():
            case.run(lpwm_stn, ins)

    row = {"case": case.name, "op": case.op}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    row["fwd_ms"], row["fwd_ms_min"] = _time(forward_only, device, iters, warmup)
    row["fwd_peak_bytes"] = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0

    if case.needs_grad and backward:
        wrt = [ins[k] for k in case.grad_wrt]
        probe = case.run(lpwm_stn, ins)
        cot = case.cotangent(probe)
        del probe

        def fwd_bwd():
            out = case.run(lpwm_stn, ins)
            torch.autograd.grad(out, wrt, grad_outputs=cot)

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        row["fwd_bwd_ms"], row["fwd_bwd_ms_min"] = _time(fwd_bwd, device, iters, warmup)
        row["fwd_bwd_peak_bytes"] = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    else:
        row["fwd_bwd_ms"] = row["fwd_bwd_ms_min"] = float("nan")
        row["fwd_bwd_peak_bytes"] = 0
    # Compatibility for callers written against the original single peak field.
    row["peak_bytes"] = row["fwd_bwd_peak_bytes"] or row["fwd_peak_bytes"]

    if case.op == "stn_crop":
        wl = case.workload
        image_elems = wl.flat_batch * wl.ch * wl.image_size * wl.image_size
        row["inflate"] = wl.n_kp  # the repeat duplicates the image once per particle
        row["repeat_bytes"] = image_elems * wl.n_kp * 4
    return row


def run(device="cpu", scope="correctness", iters=10, warmup=3, backend=None, backward=True,
        op=None):
    cases = {"bench": BENCH_CASES, "prior": PRIOR_BENCH_CASES, "correctness": CORRECTNESS_CASES,
             "all": CORRECTNESS_CASES + BENCH_CASES}[scope]
    if op is not None:
        cases = [case for case in cases if case.op == op]
    ctx = lpwm_stn.use_backend(backend) if backend else _null_context()
    with ctx:
        name = lpwm_stn.get_backend_name()
        print(f"# torch {torch.__version__}  device={device}  backend={name}  "
              f"iters={iters} (median of), warmup={warmup}")
        print(f"{'case':<26} {'fwd ms':>9} {'fwd+bwd ms':>11} {'fwd MB':>9} {'train MB':>9}  notes")
        print("-" * 89)
        rows = []
        for case in cases:
            row = bench_case(case, device, iters, warmup, backward=backward)
            rows.append(row)
            note = ""
            if "repeat_bytes" in row:
                action = "avoids reference repeat" if (
                    name != "reference" and getattr(lpwm_stn.get_backend(), "stn_crop", None)
                ) else "repeat alloc"
                note = f"{action} {row['repeat_bytes'] / 1e6:.0f} MB (x{row['inflate']})"
            fwd_peak = row["fwd_peak_bytes"] / 1e6
            train_peak = row["fwd_bwd_peak_bytes"] / 1e6
            print(f"{row['case']:<26} {row['fwd_ms']:>9.3f} {row['fwd_bwd_ms']:>11.3f} "
                  f"{fwd_peak:>9.1f} {train_peak:>9.1f}  {note}")
    return rows


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scope", default="correctness", choices=("bench", "prior", "correctness", "all"))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--op", default=None,
                    choices=("affine_grid_sample", "spatial_transform", "stn_crop", "stn_paste",
                             "create_masks_fast", "create_masks_with_scale"),
                    help="benchmark only one operation")
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="run the sweep once per named backend and print each in turn")
    args = ap.parse_args(argv)

    for backend in (args.compare or [args.backend]):
        run(device=args.device, scope=args.scope, iters=args.iters, warmup=args.warmup,
            backend=backend, backward=not args.no_backward, op=args.op)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
