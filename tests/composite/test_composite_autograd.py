#!/usr/bin/env python
"""
Checkpoint 1: the autograd wrapper around the standalone fused composite.

Two validation lines, kept apart:

  LINE A (asserted)   Does the wrapper introduce any error? ``composite_fused`` is compared with the
                      standalone kernels it wraps, on the same inputs and the same upstream gradient.
                      Forward outputs and the four non-atomic gradients must match bit for bit (same
                      kernels, same data). The atomic ``dec_objects`` gradient is not asked to be
                      bit-reproducible: it must agree within ATOMIC_REL_L2_BOUND, fixed here in
                      advance, about 30x the largest run-to-run difference previously observed for
                      this gradient (3.2e-8). It is a plumbing bound, not an accuracy acceptance.

  LINE B (reported)   How far are the fused chain and the old Triton chain (Triton paste + the PyTorch
                      composite) from the original PyTorch Reference? Raw metrics only. No frozen rule
                      applies to these inputs (element-wise envelopes exist only for the 47 fixture
                      combinations) and no threshold is derived from any error seen here.

    python tests/composite/test_composite_autograd.py [--out result.json]

GPU tests are reported as SKIP, never as passed, when CUDA or Triton is unavailable.
"""

import argparse
import json
import os
import sys
import types
import zlib
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from composite_cases import CASES, make_inputs  # noqa: E402
from lpwm_stn.composite_autograd import composite_fused, fused_composite_supported  # noqa: E402

GRADS = ("dec_objects", "z_kp", "z_scale", "obj_on", "z_depth")
OUTS = ("alpha_masks", "bg_mask", "dec_objects_trans")
ATOMIC_REL_L2_BOUND = 1e-6
DETERMINISTIC_GRADS = ("z_kp", "z_scale", "obj_on", "z_depth")


class Skip(Exception):
    pass


def _need_gpu():
    if not torch.cuda.is_available():
        raise Skip("CUDA is not available")
    try:
        import triton  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise Skip(f"Triton is not importable: {exc}")


def cot(shape, device, tag):
    g = torch.Generator(device=device)
    g.manual_seed(0x9317 + (zlib.crc32(tag.encode()) & 0xFFFF))
    return torch.rand(shape, generator=g, device=device)


def cotangents(ins, alpha, which, device="cuda"):
    b, k = ins["dec_objects"].shape[:2]
    h = ins["img_size"]
    full = {"alpha_masks": (b, k, 1, h, h), "bg_mask": (b, 1, h, h), "dec_objects_trans": (b, 3, h, h)}
    names = [n for n in which if n in full and (n != "alpha_masks" or alpha)]
    return {n: cot(full[n], device, f"ag/{n}") for n in names}


def standalone(ins, alpha, cots):
    from lpwm_stn.composite_backward import composite_backward_triton
    from lpwm_stn.composite_triton import composite_forward_triton
    args = (ins["dec_objects"], ins["z_kp"], ins["z_scale"], ins["obj_on"], ins["z_depth"], ins["img_size"])
    masks, bg, rgb = composite_forward_triton(*args, return_alpha_masks=alpha)
    grads = composite_backward_triton(*args, cots.get("alpha_masks"), cots.get("bg_mask"), cots.get("dec_objects_trans"))
    torch.cuda.synchronize()
    return dict(zip(OUTS, (masks, bg, rgb))), dict(zip(GRADS, grads))


def wrapped(ins, alpha, cots, wrt=GRADS):
    leaves = {k: (ins[k].detach().clone().requires_grad_(k in wrt) if ins[k] is not None else None) for k in GRADS}
    outs = dict(zip(OUTS, composite_fused(leaves["dec_objects"], leaves["z_kp"], leaves["z_scale"], leaves["obj_on"],
                                         leaves["z_depth"], ins["img_size"], return_alpha_masks=alpha)))
    sel = [(n, outs[n]) for n in OUTS if n in cots and outs[n] is not None]
    want = [k for k in GRADS if k in wrt and leaves[k] is not None]
    grads = {}
    if want and sel:
        got = torch.autograd.grad([t for _, t in sel], [leaves[k] for k in want],
                                  grad_outputs=[cots[n] for n, _ in sel], allow_unused=True)
        grads = dict(zip(want, got))
    torch.cuda.synchronize()
    return outs, grads, leaves


def rel_l2(a, b):
    nb = float(torch.linalg.vector_norm(b.double()))
    nd = float(torch.linalg.vector_norm((a.double() - b.double())))
    return (nd / nb) if nb > 0 else (0.0 if nd == 0 else float("inf"))


def _check_line_a(s_out, s_grad, w_out, w_grad, wrt, tag, floor, record):
    for n in OUTS:
        if s_out[n] is None:
            assert w_out[n] is None, f"{tag}: wrapper returned {n} where the kernel returned None"
        else:
            assert torch.equal(s_out[n], w_out[n]), f"{tag}: forward output {n} differs from the standalone kernel"
    for k in GRADS:
        if k not in wrt or s_grad[k] is None:
            assert w_grad.get(k) is None, f"{tag}: gradient for {k} returned although it was not requested / has none"
            continue
        assert w_grad.get(k) is not None, f"{tag}: gradient for {k} missing"
        assert w_grad[k].shape == s_grad[k].shape and w_grad[k].dtype == s_grad[k].dtype, f"{tag}: {k} shape/dtype"
        assert bool(torch.isfinite(w_grad[k]).all()), f"{tag}: non-finite gradient for {k}"
        if k in DETERMINISTIC_GRADS:
            assert torch.equal(w_grad[k], s_grad[k]), f"{tag}: deterministic gradient {k} differs from standalone"
        else:
            r = rel_l2(w_grad[k], s_grad[k])
            record.append({"tag": tag, "wrapper_vs_standalone_rel_l2_dec_objects": r, "standalone_run_to_run_rel_l2": floor})
            assert r <= ATOMIC_REL_L2_BOUND, f"{tag}: atomic gradient rel-L2 {r:.2e} > {ATOMIC_REL_L2_BOUND:.0e}"


# ------------------------------------------------------------------------------------------------
# gate (CPU-runnable)
# ------------------------------------------------------------------------------------------------
def test_gate_rejects_cpu_inputs_with_a_reason():
    ins = make_inputs(bs=1, n_kp=4, p=8, img=32, device="cpu")
    ok, reason = fused_composite_supported(ins["dec_objects"], ins["z_kp"], ins["z_scale"], ins["obj_on"],
                                           ins["z_depth"], ins["img_size"])
    assert not ok and "not on CUDA" in reason, reason
    try:
        composite_fused(ins["dec_objects"], ins["z_kp"], ins["z_scale"], ins["obj_on"], ins["z_depth"], ins["img_size"])
    except ValueError as exc:
        assert "not on CUDA" in str(exc)
    else:
        raise AssertionError("composite_fused must raise on an unsupported input, not fall back silently")


def test_gate_reports_non_tensor_input():
    ok, reason = fused_composite_supported(None, None, None, None, None, 32)
    assert not ok and "not a tensor" in reason, reason


def test_gate_rejects_each_unsupported_layout_on_cuda():
    _need_gpu()
    base = make_inputs(bs=2, n_kp=6, p=8, img=32, device="cuda")

    def call(**over):
        a = dict(dec_objects=base["dec_objects"], z_kp=base["z_kp"], z_scale=base["z_scale"], obj_on=base["obj_on"],
                 z_depth=base["z_depth"], img_size=base["img_size"])
        a.update(over)
        return fused_composite_supported(**a)

    assert call() == (True, "")
    cases = {
        "dtype": dict(dec_objects=base["dec_objects"].double()),
        "non-contiguous": dict(z_kp=base["z_kp"].transpose(0, 1).contiguous().transpose(0, 1)),
        "channels": dict(dec_objects=torch.rand(2, 6, 3, 8, 8, device="cuda")),
        "non-square": dict(dec_objects=torch.rand(2, 6, 4, 8, 6, device="cuda")),
        "z_kp shape": dict(z_kp=base["z_kp"][:, :5].contiguous()),
        "z_scale shape": dict(z_scale=base["z_scale"][:, :, :1].contiguous()),
        "obj_on shape": dict(obj_on=base["obj_on"][:, :5].contiguous()),
        "z_depth shape": dict(z_depth=base["z_depth"].squeeze(-1).contiguous()),
        "img_size": dict(img_size=0),
        "device": dict(z_depth=base["z_depth"].cpu()),
    }
    for name, over in cases.items():
        ok, reason = call(**over)
        assert not ok and reason, f"{name}: expected a rejection with a reason, got {(ok, reason)}"
    assert call(z_scale=None) == (True, ""), "z_scale=None is supported"


# ------------------------------------------------------------------------------------------------
# LINE A: wrapper vs the standalone kernels
# ------------------------------------------------------------------------------------------------
def test_line_a_all_fixture_cases_both_alpha_settings_and_cotangent_scenarios(record=None):
    _need_gpu()
    record = [] if record is None else record
    scen = {"all": OUTS, "bg_only": ("bg_mask",), "rgb_only": ("dec_objects_trans",), "masks_only": ("alpha_masks",)}
    n = 0
    for cname, kw in CASES:
        for alpha in (True, False):
            ins = make_inputs(device="cuda", **{**kw, "return_alpha_masks": alpha})
            ok, reason = fused_composite_supported(ins["dec_objects"], ins["z_kp"], ins["z_scale"], ins["obj_on"],
                                                   ins["z_depth"], ins["img_size"])
            assert ok, f"{cname}: fixture input unexpectedly unsupported: {reason}"
            for sname, which in scen.items():
                cots = cotangents(ins, alpha, which)
                if not cots:
                    continue
                s_out, s_grad = standalone(ins, alpha, cots)
                _, s_grad2 = standalone(ins, alpha, cots)
                floor = rel_l2(s_grad2["dec_objects"], s_grad["dec_objects"])
                w_out, w_grad, _ = wrapped(ins, alpha, cots)
                _check_line_a(s_out, s_grad, w_out, w_grad, GRADS, f"{cname}/alpha={alpha}/{sname}", floor, record)
                n += 1
    assert n == len(CASES) * 7, n          # 4 cotangent scenarios with masks, 3 without
    return record


def test_unused_outputs_reach_the_backward_kernel_as_none():
    """Zero gradients must not be materialized for unused outputs: their terms are compiled out."""
    _need_gpu()
    import lpwm_stn.composite_backward as cb
    ins = make_inputs(device="cuda", bs=2, n_kp=8, p=8, img=32, return_alpha_masks=True)
    real = cb.composite_backward_triton
    seen = []

    def spy(*a, **k):
        seen.append((a[6] is None, a[7] is None, a[8] is None))
        return real(*a, **k)

    for which, expect in ((("bg_mask",), (True, False, True)), (("dec_objects_trans",), (True, True, False)),
                          (("alpha_masks",), (False, True, True)), (OUTS, (False, False, False))):
        seen.clear()
        with mock.patch.object(cb, "composite_backward_triton", side_effect=spy):
            wrapped(ins, True, cotangents(ins, True, which))
        assert seen == [expect], f"{which}: (grad_masks is None, grad_bg is None, grad_rgb is None) = {seen}"


def test_alpha_masks_false_returns_none_and_differentiable_reduced_outputs():
    _need_gpu()
    ins = make_inputs(device="cuda", bs=1, n_kp=4, p=8, img=32, return_alpha_masks=False)
    leaves = {k: (ins[k].clone().requires_grad_(True) if ins[k] is not None else None) for k in GRADS}
    outs = composite_fused(leaves["dec_objects"], leaves["z_kp"], leaves["z_scale"], leaves["obj_on"],
                           leaves["z_depth"], ins["img_size"], return_alpha_masks=False)
    assert outs[0] is None and outs[1].requires_grad and outs[2].requires_grad
    assert torch.autograd.grad(outs[2].sum(), [leaves["z_kp"]])[0] is not None


def test_requires_grad_combinations_and_no_grad_inputs():
    _need_gpu()
    for cname in ("bair_bs1", "small_noscale", "balls_bs2"):
        kw = dict(CASES)[cname]
        for alpha in (True, False):
            ins = make_inputs(device="cuda", **{**kw, "return_alpha_masks": alpha})
            cots = cotangents(ins, alpha, OUTS)
            s_out, s_grad = standalone(ins, alpha, cots)
            combos = [tuple(GRADS)] + [(g,) for g in GRADS] + [("z_kp", "z_depth"), ("dec_objects", "obj_on"),
                                                              ("z_scale", "z_depth", "obj_on")]
            for wrt in combos:
                w_out, w_grad, _ = wrapped(ins, alpha, cots, wrt=wrt)
                _check_line_a(s_out, s_grad, w_out, w_grad, wrt, f"{cname}/alpha={alpha}/wrt={wrt}", 0.0, [])
            leaves = {k: (ins[k].clone() if ins[k] is not None else None) for k in GRADS}
            outs = composite_fused(leaves["dec_objects"], leaves["z_kp"], leaves["z_scale"], leaves["obj_on"],
                                   leaves["z_depth"], ins["img_size"], return_alpha_masks=alpha)
            assert not any(o.requires_grad for o in outs if o is not None), "no input requires grad: outputs must not"


def test_z_scale_none_and_scale_normalized():
    _need_gpu()
    from lpwm_stn.composite_backward import composite_backward_triton
    from lpwm_stn.composite_triton import composite_forward_triton
    ins = make_inputs(device="cuda", bs=2, n_kp=8, p=8, img=32, with_scale=False)
    cots = cotangents(ins, True, OUTS)
    s_out, s_grad = standalone(ins, True, cots)
    assert s_grad["z_scale"] is None
    w_out, w_grad, _ = wrapped(ins, True, cots)
    assert w_grad.get("z_scale") is None
    _check_line_a(s_out, s_grad, w_out, w_grad, GRADS, "z_scale=None", 0.0, [])

    ins = make_inputs(device="cuda", bs=2, n_kp=8, p=8, img=32)
    norm = torch.sigmoid(ins["z_scale"]).contiguous()
    args = (ins["dec_objects"], ins["z_kp"], norm, ins["obj_on"], ins["z_depth"], ins["img_size"])
    s = composite_forward_triton(*args, return_alpha_masks=False, scale_normalized=True)
    leaves = [t.clone().requires_grad_(True) for t in args[:5]]
    outs = composite_fused(*leaves, ins["img_size"], return_alpha_masks=False, scale_normalized=True)
    assert torch.equal(outs[1], s[1]) and torch.equal(outs[2], s[2])
    c = cotangents(ins, False, OUTS)
    g = torch.autograd.grad([outs[1], outs[2]], leaves, grad_outputs=[c["bg_mask"], c["dec_objects_trans"]])
    sg = composite_backward_triton(*args, None, c["bg_mask"], c["dec_objects_trans"], scale_normalized=True)
    for name, a, b in zip(GRADS, g, sg):
        if name in DETERMINISTIC_GRADS:
            assert torch.equal(a, b), f"scale_normalized: {name}"


def test_saves_only_the_inputs_and_never_a_per_particle_plane():
    _need_gpu()
    ins = make_inputs(device="cuda", bs=4, n_kp=90, p=8, img=128, return_alpha_masks=False)
    leaves = {k: (ins[k].clone().requires_grad_(True) if ins[k] is not None else None) for k in GRADS}
    input_ptrs = {t.untyped_storage().data_ptr() for t in leaves.values() if t is not None}
    saved = []

    def pack(t):
        saved.append((t.untyped_storage().data_ptr(), t.numel() * t.element_size()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        outs = composite_fused(leaves["dec_objects"], leaves["z_kp"], leaves["z_scale"], leaves["obj_on"],
                               leaves["z_depth"], ins["img_size"], return_alpha_masks=False)
    assert saved and {p for p, _ in saved} <= input_ptrs, f"saved tensors other than the inputs: {saved}"
    plane_bytes = 4 * 90 * 1 * 128 * 128 * 4
    torch.cuda.synchronize()
    c = cotangents(ins, False, OUTS)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outs = composite_fused(leaves["dec_objects"], leaves["z_kp"], leaves["z_scale"], leaves["obj_on"],
                           leaves["z_depth"], ins["img_size"], return_alpha_masks=False)
    torch.autograd.grad([outs[1], outs[2]], [leaves[k] for k in GRADS if leaves[k] is not None],
                        grad_outputs=[c["bg_mask"], c["dec_objects_trans"]])
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    assert peak < plane_bytes, f"peak {peak/1e6:.1f} MB reaches one per-particle plane ({plane_bytes/1e6:.1f} MB)"


def test_double_backward_is_refused_not_silently_dropped():
    _need_gpu()
    ins = make_inputs(device="cuda", bs=1, n_kp=4, p=8, img=32, return_alpha_masks=False)
    leaves = {k: (ins[k].clone().requires_grad_(True) if ins[k] is not None else None) for k in GRADS}
    outs = composite_fused(leaves["dec_objects"], leaves["z_kp"], leaves["z_scale"], leaves["obj_on"],
                           leaves["z_depth"], ins["img_size"], return_alpha_masks=False)
    try:
        torch.autograd.grad(outs[2].sum(), [leaves["z_kp"]], create_graph=True)
    except RuntimeError as exc:
        assert "double backward" in str(exc), exc
    else:
        raise AssertionError("create_graph=True must be refused")
    g = torch.autograd.grad(outs[2].sum(), [leaves["z_kp"]])[0]     # the ordinary first-order path still works
    assert torch.isfinite(g).all()


# ------------------------------------------------------------------------------------------------
# LINE B: total difference vs the original Reference (reported, never asserted)
# ------------------------------------------------------------------------------------------------
def line_b_report():
    _need_gpu()
    import lpwm_stn.composite_reference as cr
    from lpwm_stn import triton_backend
    shim = types.SimpleNamespace(stn_paste=triton_backend.stn_paste)

    def chain(ins, alpha, cots, kind):
        leaves = {k: (ins[k].detach().clone().requires_grad_(True) if ins[k] is not None else None) for k in GRADS}
        call = dict(dec_objects=leaves["dec_objects"], z_kp=leaves["z_kp"], z_scale=leaves["z_scale"],
                    obj_on=leaves["obj_on"], z_depth=leaves["z_depth"], img_size=ins["img_size"],
                    return_alpha_masks=alpha)
        if kind == "fused":
            outs = composite_fused(**call)
        elif kind == "old":
            with mock.patch.object(cr, "reference", shim):
                outs = cr.composite_reference(**call)
        else:
            outs = cr.composite_reference(**call)
        outs = dict(zip(OUTS, outs))
        sel = [(n, outs[n]) for n in OUTS if n in cots and outs[n] is not None]
        want = [k for k in GRADS if leaves[k] is not None]
        gs = torch.autograd.grad([t for _, t in sel], [leaves[k] for k in want], grad_outputs=[cots[n] for n, _ in sel])
        return {n: t.detach() for n, t in outs.items() if t is not None}, dict(zip(want, gs))

    rows = []
    for cname, kw in CASES:
        for alpha in (True, False):
            ins = make_inputs(device="cuda", **{**kw, "return_alpha_masks": alpha})
            # the old chain is only "old" if the Triton paste really runs: a silent fallback would make
            # old-vs-reference exactly zero and the comparison meaningless
            assert triton_backend._can_use_triton_paste(ins["z_kp"], ins["dec_objects"], ins["z_scale"],
                                                        ins["img_size"]), f"{cname}: Triton paste not used by the old chain"
            cots = cotangents(ins, alpha, OUTS)
            res = {k: chain(ins, alpha, cots, k) for k in ("ref", "old", "fused")}
            for pair, (a, b) in {"old_vs_ref": ("old", "ref"), "fused_vs_ref": ("fused", "ref"),
                                 "fused_vs_old": ("fused", "old")}.items():
                for n in res["ref"][0]:
                    rows.append({"case": cname, "alpha": alpha, "pair": pair, "tensor": n,
                                 "max_abs": float((res[a][0][n].double() - res[b][0][n].double()).abs().max()),
                                 "rel_l2": rel_l2(res[a][0][n], res[b][0][n])})
                for k in res["ref"][1]:
                    rows.append({"case": cname, "alpha": alpha, "pair": pair, "tensor": f"grad[{k}]",
                                 "max_abs": float((res[a][1][k].double() - res[b][1][k].double()).abs().max()),
                                 "rel_l2": rel_l2(res[a][1][k], res[b][1][k])})
    return rows


def summarize_line_b(rows):
    worst = {}
    for r in rows:
        key = (r["pair"], r["tensor"])
        w = worst.setdefault(key, {"max_abs": 0.0, "rel_l2": 0.0, "case": None})
        if r["rel_l2"] > w["rel_l2"]:
            w.update(rel_l2=r["rel_l2"], case=f"{r['case']}/alpha={r['alpha']}")
        w["max_abs"] = max(w["max_abs"], r["max_abs"])
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    record = []
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures, skipped = [], []
    for name, fn in tests:
        try:
            fn(record) if name == "test_line_a_all_fixture_cases_both_alpha_settings_and_cotangent_scenarios" else fn()
        except Skip as exc:
            skipped.append(name)
            print(f"  SKIP {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, str(exc)))
            print(f"  FAIL {name}: {exc}")
        else:
            print(f"  ok   {name}")
    report = {"kind": "autograd wrapper checkpoint: line A asserted, line B reported only",
              "atomic_rel_l2_bound_fixed_in_advance": ATOMIC_REL_L2_BOUND, "failures": failures, "skipped": skipped}
    if record:
        vals = [r["wrapper_vs_standalone_rel_l2_dec_objects"] for r in record]
        floors = [r["standalone_run_to_run_rel_l2"] for r in record]
        report["line_a_atomic_dec_objects"] = {"n": len(vals), "worst_wrapper_vs_standalone": max(vals),
                                               "worst_standalone_run_to_run": max(floors)}
        print(f"\nLINE A atomic dec_objects gradient: {len(vals)} comparisons, worst wrapper-vs-standalone rel-L2 "
              f"{max(vals):.2e}; worst standalone run-to-run {max(floors):.2e}; fixed bound {ATOMIC_REL_L2_BOUND:.0e}")
    if torch.cuda.is_available() and not skipped:
        rows = line_b_report()
        worst = summarize_line_b(rows)
        report["line_b_rows"] = rows
        report["line_b_worst"] = {f"{p}|{t}": v for (p, t), v in worst.items()}
        print("\nLINE B (reported only; worst over 12 fixture cases x alpha True/False; max_abs | rel_l2 | at)")
        for pair in ("old_vs_ref", "fused_vs_ref", "fused_vs_old"):
            print(f"  {pair}")
            for (p, t), v in worst.items():
                if p == pair:
                    print(f"     {t:19s} {v['max_abs']:.2e} | {v['rel_l2']:.2e} | {v['case']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote", args.out)
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} failed, {len(skipped)} skipped, {len(tests)} total")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
