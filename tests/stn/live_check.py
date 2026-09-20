#!/usr/bin/env python
"""
Same-machine live check: Triton stn_crop / stn_paste against the PyTorch reference.

Independent of the frozen golden fixtures: they are neither read nor written, so
this is valid on any GPU. For every case the inputs are built once, cloned into
two independent leaf sets, and both paths receive the same upstream cotangent.

Acceptance thresholds are the ones already used for this backend and are not
adjustable here (docs/stn_triton_crop.md:39-41, docs/stn_triton_paste.md:34-35):
forward max|diff| <= 2e-5, crop input-gradient max|diff| <= 1e-2, paste
input-gradient max|diff| <= 1e-1. Relative L2 is reported as a diagnostic only.
When the reference norm is zero it is 0.0 if the difference is also exactly zero
and None (undefined) otherwise; it never affects pass/fail.

The Triton path is asserted through the custom autograd path, not kernel names:
the output's grad_fn must belong to _StnCrop / _StnPaste (forward), and that node
must execute in backward and produce a gradient for every requested input.

    python live_check.py --repo-root /path/to/lpwm --out result.json
    python live_check.py --repo-root /path/to/lpwm --list      # enumerate only, no GPU
"""

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback

import torch

FORWARD_TOL = 2e-5
GRAD_TOL = {"stn_crop": 1e-2, "stn_paste": 1e-1}
CHUNK = 1 << 24

# op -> (Triton autograd Function name, {grad_wrt key: position among the Function's tensor
# inputs}). A node hook's grad_inputs holds tensor inputs only, and the optional scale is last.
TRITON_NODE = {
    "stn_crop": ("_StnCrop", {"x": 0, "kp": 1, "z_scale": 2}),
    "stn_paste": ("_StnPaste", {"kp_batch": 0, "patches_batch": 1, "scale": 2}),
}

lpwm_stn = reference = triton_backend = None


def bootstrap(repo_root):
    global lpwm_stn, reference, triton_backend
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import lpwm_stn as _pkg
    from lpwm_stn import reference as _ref, triton_backend as _tri
    lpwm_stn, reference, triton_backend = _pkg, _ref, _tri
    from tests.stn.stn_cases import BENCH_CASES, CORRECTNESS_CASES, PRIOR_BENCH_CASES
    return repo_root, CORRECTNESS_CASES + BENCH_CASES + PRIOR_BENCH_CASES


def enumerate_cases(all_cases):
    unique = {}
    for case in all_cases:
        unique.setdefault(case.name, case)
    selected, excluded = [], []
    for case in unique.values():
        if case.op in TRITON_NODE and case.needs_grad:
            selected.append(case)
        else:
            reason = "no Triton implementation for this op" if case.op not in TRITON_NODE else "no gradient contract"
            excluded.append((case.name, case.op, reason))
    return selected, excluded


def clone_inputs(ins):
    out = {}
    for key, value in ins.items():
        out[key] = (value.detach().clone().requires_grad_(value.requires_grad)
                    if torch.is_tensor(value) else value)
    return out


def is_node_of(node, cls):
    if node is None:
        return False
    forward_cls = getattr(type(node), "_forward_cls", None)
    return forward_cls is cls if forward_cls is not None else type(node).__name__ == cls.__name__ + "Backward"


def compare(actual, expected, tol):
    rec = {"shape": list(actual.shape), "dtype": str(actual.dtype), "tol": tol}
    problems = []
    if actual.shape != expected.shape:
        problems.append(f"shape {tuple(actual.shape)} != reference {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        problems.append(f"dtype {actual.dtype} != reference {expected.dtype}")
    if problems:
        rec.update(max_abs=None, rel_l2=None, ref_absmax=None, finite=None, passed=False)
        return rec, problems

    a, b = actual.detach().reshape(-1), expected.detach().reshape(-1)
    finite_a = finite_b = True
    max_abs = ref_absmax = sq_diff = sq_ref = 0.0
    for i in range(0, a.numel(), CHUNK):
        ac, bc = a[i:i + CHUNK], b[i:i + CHUNK]
        finite_a = finite_a and bool(torch.isfinite(ac).all())
        finite_b = finite_b and bool(torch.isfinite(bc).all())
        if not (finite_a and finite_b):
            break
        diff = ac.double() - bc.double()
        max_abs = max(max_abs, float(diff.abs().max()))
        ref_absmax = max(ref_absmax, float(bc.abs().max()))
        sq_diff += float((diff * diff).sum())
        sq_ref += float((bc.double() * bc.double()).sum())
    if not finite_a:
        problems.append("Triton tensor contains NaN/Inf")
    if not finite_b:
        problems.append("reference tensor contains NaN/Inf")
    if problems:
        rec.update(max_abs=None, rel_l2=None, ref_absmax=None, finite=False, passed=False)
        return rec, problems

    if sq_ref > 0.0:
        rel_l2 = math.sqrt(sq_diff) / math.sqrt(sq_ref)
    else:
        rel_l2 = 0.0 if sq_diff == 0.0 else None
    rec.update(max_abs=max_abs, rel_l2=rel_l2, ref_absmax=ref_absmax, finite=True, passed=max_abs <= tol)
    if not rec["passed"]:
        problems.append(f"max|diff|={max_abs:.3e} > tol={tol:.1e}")
    return rec, problems


def check_case(case, device):
    node_name, slots = TRITON_NODE[case.op]
    expected_cls = getattr(triton_backend, node_name)
    rec = {"case": case.name, "op": case.op, "grad_wrt": list(case.grad_wrt), "checks": {}, "problems": []}
    problems = rec["problems"]

    ins = case.inputs(device=device)

    ref_ins = clone_inputs(ins)
    out_ref = case.run(reference, ref_ins)
    if is_node_of(out_ref.grad_fn, expected_cls):
        problems.append("reference arm unexpectedly produced a Triton autograd node")
    cotangent = case.cotangent(out_ref)
    ref_grads = torch.autograd.grad(out_ref, [ref_ins[k] for k in case.grad_wrt],
                                    grad_outputs=cotangent, allow_unused=False)
    out_ref = out_ref.detach()
    del ref_ins

    tri_ins = clone_inputs(ins)
    with lpwm_stn.use_backend("triton"):
        out_tri = case.run(lpwm_stn, tri_ins)
    node = out_tri.grad_fn
    rec["triton_forward_node"] = type(node).__name__ if node is not None else None
    if not is_node_of(node, expected_cls):
        problems.append(f"forward did not use {node_name} (grad_fn={rec['triton_forward_node']}); "
                        "the dispatcher fell back to the reference")
    fired = []
    if node is not None:
        node.register_hook(lambda grad_inputs, grad_outputs: fired.append([g is not None for g in grad_inputs]))
    tri_grads = torch.autograd.grad(out_tri, [tri_ins[k] for k in case.grad_wrt],
                                    grad_outputs=cotangent, allow_unused=False)
    if is_node_of(node, expected_cls):
        if len(fired) != 1:
            problems.append(f"{node_name}.backward executed {len(fired)} times, expected 1")
        else:
            mask = fired[0]
            missing = [k for k in case.grad_wrt if slots[k] >= len(mask) or not mask[slots[k]]]
            if missing:
                problems.append(f"{node_name}.backward produced no gradient for {missing}")
    rec["triton_backward_executed"] = len(fired) == 1

    checks = [("forward", out_tri.detach(), out_ref, FORWARD_TOL)]
    checks += [(f"grad[{k}]", t, r, GRAD_TOL[case.op]) for k, t, r in zip(case.grad_wrt, tri_grads, ref_grads)]
    for label, actual, expected, tol in checks:
        check_rec, check_problems = compare(actual, expected, tol)
        rec["checks"][label] = check_rec
        problems.extend(f"{label}: {p}" for p in check_problems)

    rec["passed"] = not problems
    return rec


def provenance(repo_root, script_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", repo_root, *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    out = {"repo_root": repo_root, "script": os.path.abspath(script_path),
           "script_sha256": hashlib.sha256(open(script_path, "rb").read()).hexdigest()}
    try:
        out.update(branch=git("rev-parse", "--abbrev-ref", "HEAD"), commit=git("rev-parse", "HEAD"),
                   dirty_paths=git("status", "--porcelain").splitlines(),
                   remotes=git("remote", "-v").splitlines())
    except Exception as exc:
        out["git_error"] = str(exc)
    return out


def environment(device):
    env = {"torch": torch.__version__, "device": device, "cuda_runtime": torch.version.cuda,
           "triton": getattr(getattr(triton_backend, "triton", None), "__version__", None)}
    if device.startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["capability"] = list(torch.cuda.get_device_capability(0))
    return env


def fmt(value):
    return "n/a" if value is None else f"{value:.3e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the full JSON result here")
    ap.add_argument("--list", action="store_true", help="enumerate the cases and exit")
    ap.add_argument("--only", default=None, help="run only cases whose name contains this substring")
    args = ap.parse_args(argv)

    repo_root, all_cases = bootstrap(args.repo_root)
    selected, excluded = enumerate_cases(all_cases)
    if args.only:
        selected = [c for c in selected if args.only in c.name]

    print(f"cases to run: {len(selected)}  (excluded from this check: {len(excluded)})")
    by_op = {}
    for case in selected:
        by_op[case.op] = by_op.get(case.op, 0) + 1
    print("  by op: " + ", ".join(f"{op}={n}" for op, n in sorted(by_op.items())))
    if args.list:
        for case in selected:
            print(f"  {case.name:32s} {case.op:10s} grad_wrt={list(case.grad_wrt)}")
        print("excluded: " + ", ".join(sorted({f"{op} ({reason})" for _, op, reason in excluded})))
        return 0
    if not selected:
        print("no cases selected")
        return 2
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; refusing to run (the Triton path needs a GPU)")
        return 2

    env = environment(args.device)
    print(f"independent same-machine check; frozen fixtures are neither read nor written\n{json.dumps(env)}\n")
    results = []
    for case in selected:
        try:
            rec = check_case(case, args.device)
        except Exception as exc:  # noqa: BLE001 - recorded, not handled; no traceback tensors retained
            rec = {"case": case.name, "op": case.op, "grad_wrt": list(case.grad_wrt), "checks": {},
                   "problems": [f"exception: {type(exc).__name__}: {exc}"], "passed": False,
                   "traceback": traceback.format_exc()}
            del exc
        results.append(rec)
        detail = " ".join(f"{k}:abs={fmt(v['max_abs'])},relL2={fmt(v['rel_l2'])}" for k, v in rec["checks"].items())
        print(f"{'PASS' if rec['passed'] else 'FAIL'} {case.name:30s} {detail}")
        for problem in rec["problems"]:
            print(f"     - {problem}")
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    failed = [r for r in results if not r["passed"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} cases passed; {len(failed)} failed")
    report = {"schema_version": 1, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "kind": "independent same-machine Triton-vs-reference live check (no fixtures)",
              "thresholds": {"forward_max_abs": FORWARD_TOL, "grad_max_abs": GRAD_TOL,
                             "relative_l2": "diagnostic only; 0.0 when both exact, null when reference norm is 0"},
              "env": env, "provenance": provenance(repo_root, __file__),
              "command": sys.argv, "n_cases": len(results), "n_failed": len(failed),
              "excluded": [{"case": n, "op": o, "reason": r} for n, o, r in excluded],
              "cases": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
