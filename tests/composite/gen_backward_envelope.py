"""
Pre-kernel per-element reference-reordering envelope for grad[z_kp] / grad[z_scale].

Built entirely from the PyTorch reference under a fixed, recorded set of particle
orders. No Triton backward is involved, so the acceptance contract cannot be
fitted to a kernel.

Method
------
For each case and cotangent scenario the reference VJP is evaluated under several
particle orders, every gradient un-permuted back to canonical order, and the
per-element extremes recorded:

    lower_i = min over development orders
    upper_i = max over development orders
    delta_i = 1e-5 + 1e-5 * max(|lower_i|, |upper_i|)

acceptance:  lower_i - delta_i <= candidate_i <= upper_i + delta_i
             AND relative_L2(candidate, canonical) <= 1e-5

Where the reference is stable the envelope collapses to a point and the rule is a
strict 1e-5 halo -- including at zero. Only the specific elements for which the
reference itself demonstrates reordering variability get any extra room, and only
as much as it actually demonstrated.

The method is validated on HELD-OUT orders never used to build the envelope: if
those do not fall inside it, the envelope is not descriptive and must not be
frozen.
"""
import json, os, sys
sys.path.insert(0, "/workspace/lpwm"); sys.path.insert(0, "/workspace/lpwm/tests/composite")
import torch
from lpwm_stn.composite_reference import composite_reference
from composite_cases import CASES, make_inputs

OUT_NAMES = ("alpha_masks", "bg_mask", "dec_objects_trans")
GRADS = ("dec_objects", "z_kp", "z_scale", "obj_on", "z_depth")
PARTICLE_INDEXED = GRADS
SCENARIOS = ("rgb_only", "bg_only", "masks_only", "mixed")
ENVELOPED = ("z_kp", "z_scale")

ATOL = {"dec_objects": 2e-5, "z_kp": 1e-5, "z_scale": 1e-5, "obj_on": 1e-5, "z_depth": 1e-5}
RTOL = 1e-5
L2_TOL = 1e-5
TINY = 1e-30

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FIXTURE_PT = os.path.join(_FIX, "backward_envelope.pt")
POLICY_JSON = os.path.join(_FIX, "frozen_backward_policy.json")

#: development orders -- these BUILD the envelope. Recorded exactly.
DEV_SPEC = [("identity", None), ("reverse", None),
            ("cyclic_1", 1), ("cyclic_half", "half"), ("cyclic_third", "third"),
            ("rand_101", 101), ("rand_202", 202), ("rand_303", 303), ("rand_404", 404)]
#: held-out orders -- never used to build it; they must fall inside.
HELDOUT_SPEC = [("rand_911", 911), ("rand_922", 922), ("rand_933", 933),
                ("rand_944", 944), ("cyclic_2", 2)]


def make_order(name, spec, K, device):
    if name == "identity":
        return torch.arange(K, device=device)
    if name == "reverse":
        return torch.arange(K - 1, -1, -1, device=device)
    if name.startswith("cyclic"):
        shift = {"half": K // 2, "third": K // 3}.get(spec, spec if isinstance(spec, int) else 1)
        shift = max(int(shift) % K, 0) if K > 1 else 0
        return torch.roll(torch.arange(K, device=device), shifts=shift)
    g = torch.Generator(device=device); g.manual_seed(int(spec))
    return torch.randperm(K, generator=g, device=device)


def cot(t, tag):
    import zlib
    g = torch.Generator(device=t.device); g.manual_seed(0x9317 + (zlib.crc32(tag.encode()) & 0xFFFF))
    return torch.rand(t.shape, generator=g, device=t.device, dtype=t.dtype)


def base_cots(ins, scenario, tag):
    outs = dict(zip(OUT_NAMES, composite_reference(**ins)))
    want = {"rgb_only": ["dec_objects_trans"], "bg_only": ["bg_mask"],
            "masks_only": ["alpha_masks"], "mixed": list(OUT_NAMES)}[scenario]
    return [(n, cot(outs[n], f"{tag}/{n}")) for n in want if outs.get(n) is not None]


def run_grads(ins, cots_base, order=None):
    """Run the reference under `order`, returning gradients in CANONICAL order."""
    ins = dict(ins); inv = None
    if order is not None:
        inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=order.device)
        for k in PARTICLE_INDEXED:
            if ins.get(k) is not None:
                ins[k] = ins[k][:, order].contiguous()
    leaves = {k: (ins[k].detach().clone().requires_grad_(True) if ins.get(k) is not None else None)
              for k in GRADS}
    call = dict(ins); call.update({k: v for k, v in leaves.items() if v is not None})
    outs = dict(zip(OUT_NAMES, composite_reference(**call)))
    cmap = dict(cots_base); sel, cots = [], []
    for n, _ in cots_base:
        if outs.get(n) is None: continue
        c = cmap[n]
        if order is not None and c.dim() == 5:      # alpha_masks cotangent is particle-indexed
            c = c[:, order].contiguous()
        sel.append((n, outs[n])); cots.append(c)
    wrt = [k for k in GRADS if leaves.get(k) is not None]
    gs = torch.autograd.grad([t for _, t in sel], [leaves[k] for k in wrt],
                             grad_outputs=cots, allow_unused=True)
    return {k: (None if g is None else (g[:, inv].contiguous() if inv is not None else g))
            for k, g in zip(wrt, gs)}


def envelope_report(lower, upper, cand):
    """distance outside [lower-delta, upper+delta], normalised by delta."""
    delta = ATOL["z_kp"] + RTOL * torch.maximum(lower.abs(), upper.abs())
    lo, hi = lower - delta, upper + delta
    dist = torch.clamp(lo - cand, min=0) + torch.clamp(cand - hi, min=0)
    ratio = dist / delta
    idx = int(ratio.argmax())
    fr = ratio.reshape(-1)
    return {"ok": bool((dist == 0).all()), "n_out": int((dist > 0).sum()),
            "numel": int(dist.numel()), "idx": idx,
            "worst_ratio": float(fr[idx]),
            "cand": float(cand.reshape(-1)[idx]), "lower": float(lower.reshape(-1)[idx]),
            "upper": float(upper.reshape(-1)[idx]), "delta": float(delta.reshape(-1)[idx])}


def combined_report(ref, cand, name):
    """argmax(diff/bound) selection -- diff, ref, bound, idx all from ONE element."""
    atol = ATOL[name]
    d = (cand - ref).abs(); bound = atol + RTOL * ref.abs()
    ratio = d / bound; idx = int(ratio.argmax())
    rel = float(torch.linalg.vector_norm(cand - ref) /
                max(float(torch.linalg.vector_norm(ref)), TINY))
    return {"ok": bool((d <= bound).all()) and rel <= L2_TOL, "n_viol": int((d > bound).sum()),
            "idx": idx, "diff": float(d.reshape(-1)[idx]), "abs_ref": float(ref.reshape(-1)[idx].abs()),
            "bound": float(bound.reshape(-1)[idx]), "ratio": float(ratio.reshape(-1)[idx]),
            "rel_l2": rel}


def selftest_reporting():
    ref = torch.tensor([[1000.0, 0.0]], device="cuda")
    cand = torch.tensor([[1000.02, 2e-5]], device="cuda")
    r = combined_report(ref, cand, "z_kp")
    assert r["idx"] != int((cand - ref).abs().argmax()), "followed argmax(diff), not argmax(diff/bound)"
    assert abs(r["diff"] - 2e-5) < 1e-12 and r["abs_ref"] == 0.0
    assert abs(r["ratio"] - 2.0) < 1e-6
    return True


if __name__ == "__main__":
    torch.manual_seed(0)
    print("reporting regression guard:", "PASS" if selftest_reporting() else "FAIL", "\n")
    env_store, stats = {}, []
    worst_heldout = 0.0; heldout_fail = []
    worst_other = {g: 0.0 for g in ("dec_objects", "obj_on", "z_depth")}
    worst_l2 = {g: 0.0 for g in GRADS}

    for cname, kw in CASES:
        K = kw["n_kp"]
        ins = make_inputs(device="cuda", **kw)
        dev = TINY
        for sc in SCENARIOS:
            cb = base_cots(ins, sc, cname)
            if not cb: continue
            canonical = run_grads(ins, cb)
            # ---- build envelope from DEVELOPMENT orders ----
            stack = {g: [] for g in ENVELOPED}
            for name, spec in DEV_SPEC:
                o = make_order(name, spec, K, ins["z_kp"].device)
                g = run_grads(ins, cb, order=o)
                for e in ENVELOPED:
                    if g.get(e) is not None: stack[e].append(g[e])
            for e in ENVELOPED:
                if not stack[e]: continue
                S = torch.stack(stack[e])
                lo, hi = S.min(0).values, S.max(0).values
                env_store[(cname, sc, e)] = (lo, hi)
                width = (hi - lo)
                stats.append({"case": cname, "scenario": sc, "grad": e,
                              "max_width": float(width.max()),
                              "n_nonzero": int((width > 0).sum()), "numel": int(width.numel())})
            # ---- validate on HELD-OUT orders ----
            for name, spec in HELDOUT_SPEC:
                o = make_order(name, spec, K, ins["z_kp"].device)
                g = run_grads(ins, cb, order=o)
                for e in ENVELOPED:
                    if g.get(e) is None or (cname, sc, e) not in env_store: continue
                    lo, hi = env_store[(cname, sc, e)]
                    r = envelope_report(lo, hi, g[e])
                    worst_heldout = max(worst_heldout, r["worst_ratio"])
                    if not r["ok"]: heldout_fail.append((cname, sc, name, e, r))
                    l2 = float(torch.linalg.vector_norm(g[e] - canonical[e]) /
                               max(float(torch.linalg.vector_norm(canonical[e])), TINY))
                    worst_l2[e] = max(worst_l2[e], l2)
                for o_g in ("dec_objects", "obj_on", "z_depth"):
                    if g.get(o_g) is None: continue
                    r = combined_report(canonical[o_g], g[o_g], o_g)
                    worst_other[o_g] = max(worst_other[o_g], r["ratio"])
                    worst_l2[o_g] = max(worst_l2[o_g], r["rel_l2"])
        del ins; torch.cuda.empty_cache()

    print(f"development orders: {[n for n,_ in DEV_SPEC]}")
    print(f"held-out orders   : {[n for n,_ in HELDOUT_SPEC]}\n")
    for e in ENVELOPED:
        rows = [s for s in stats if s["grad"] == e]
        mw = max(r["max_width"] for r in rows); nz = sum(r["n_nonzero"] for r in rows)
        tot = sum(r["numel"] for r in rows)
        print(f"{e:<9} max envelope width={mw:.3e}  elements with nonzero width={nz}/{tot} "
              f"({100.0*nz/tot:.1f}%)")
    print(f"\nworst held-out distance-to-envelope ratio: {worst_heldout:.3e}  "
          f"({'INSIDE' if worst_heldout == 0 else 'OUTSIDE'})")
    print(f"held-out failures: {len(heldout_fail)}")
    for c, sc, o, e, r in heldout_fail[:10]:
        print(f"   {c}/{sc}/{o} grad[{e}] cand={r['cand']:.6e} lower={r['lower']:.6e} "
              f"upper={r['upper']:.6e} delta={r['delta']:.3e} ratio={r['worst_ratio']:.3f}")
    print(f"\nunchanged-policy gradients, worst diff/bound over held-out orders:")
    for g, v in worst_other.items(): print(f"   {g:<13} {v:.3f}")
    print(f"\nworst relative L2 per gradient:")
    for g in GRADS: print(f"   {g:<13} {worst_l2[g]:.3e}  {'OK' if worst_l2[g] <= L2_TOL else 'EXCEEDS'}")
    print(f"\n{'ENVELOPE VALIDATED' if not heldout_fail else 'ENVELOPE NOT VALIDATED'}")
    if not heldout_fail:
        # ---- compact dense fixture: ONLY the lower/upper bounds ----
        # every bound is materialized independently so no small view can retain a
        # large backing storage through serialization
        env_out = {}
        for (cname, sc, g), (lo, hi) in env_store.items():
            env_out[f"{cname}|{sc}|{g}"] = {
                "lower": lo.detach().to(device="cpu", dtype=torch.float32).contiguous().clone(),
                "upper": hi.detach().to(device="cpu", dtype=torch.float32).contiguous().clone(),
            }
        torch.save({"version": 1, "env": env_out}, FIXTURE_PT)

        meta = {
            "version": 1,
            "frozen_at_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         __import__("time").gmtime()),
            "frozen_before_backward_kernel_exists": True,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "envelope_fixture": os.path.basename(FIXTURE_PT),
            "development_orders": [[n, s_] for n, s_ in DEV_SPEC],
            "heldout_orders": [[n, s_] for n, s_ in HELDOUT_SPEC],
            "enveloped_gradients": list(ENVELOPED),
            "atol": ATOL, "rtol": RTOL, "l2_tol": L2_TOL,
            "formulas": {
                "z_kp/z_scale": ("delta_i = atol + rtol*max(|lower_i|,|upper_i|); "
                                 "lower_i - delta_i <= cand_i <= upper_i + delta_i"),
                "dec_objects": "|cand-ref| <= 2e-5 + 1e-5*|ref|  (grid_sampler atomicAdd exception)",
                "obj_on/z_depth": "|cand-ref| <= 1e-5 + 1e-5*|ref|",
                "all": "relative_L2(cand, canonical_reference) <= 1e-5",
                "reporting": "worst element selected by argmax(diff/bound); diff, ref, bound "
                             "and index all read from that same element",
            },
            "provenance": ("built from the PyTorch reference only, under the recorded development "
                           "orders; validated on held-out orders never used to build it. No Triton "
                           "backward result was involved."),
            "validation": {"heldout_failures": 0, "worst_heldout_ratio": worst_heldout,
                           "worst_other_diff_over_bound": worst_other,
                           "worst_relative_l2": worst_l2},
            "stats": stats,
        }
        json.dump(meta, open(POLICY_JSON, "w"), indent=2, default=float)
        print(f"\nwrote {FIXTURE_PT} ({os.path.getsize(FIXTURE_PT)} bytes)")
        print(f"wrote {POLICY_JSON} ({os.path.getsize(POLICY_JSON)} bytes)")
