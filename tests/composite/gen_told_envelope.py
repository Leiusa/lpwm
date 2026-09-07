#!/usr/bin/env python
"""
Policy v2 envelope: widen the v1 reference envelope with cross-implementation spread.

v1 was built from particle reorderings of ONE paste implementation, so it never
saw the spread between two implementations the project already accepts. The saved
counterfactual showed that gap rejects the previously accepted Triton paste itself
(grad[z_kp] max ratio 29.701, first failure bair_bs1/rgb_only idx=120), including
in a variant where both sides were fed a byte-identical dL/dsample -- which places
the disagreement in the paste VJP, not the composite feed.

This generator collects the missing half: the same recorded development and
held-out particle orders, run through the ALREADY-ACCEPTED Triton paste.

    lower_v2 = min(lower_v1_reference, min over T_old development orders)
    upper_v2 = max(upper_v1_reference, max over T_old development orders)
    delta_i  = 1e-5 + 1e-5 * max(|lower_v2_i|, |upper_v2_i|)

z_depth gains an envelope in v2 that it did not have in v1, so its reference
development bounds are generated here too -- that is missing data, not a rerun of
the canonical 47-combination counterfactual, which is reused as saved.

This file must never import the fused backward; test_policy_v2.py enforces that.
"""

import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from lpwm_stn import reference, triton_backend                       # noqa: E402
from lpwm_stn.composite_reference import EPS                          # noqa: E402
from composite_cases import CASES, make_inputs                        # noqa: E402
from gen_backward_envelope import (                                   # noqa: E402
    ATOL, DEV_SPEC, HELDOUT_SPEC, L2_TOL, RTOL, SCENARIOS, TINY, cot, make_order)

OUT_NAMES = ("alpha_masks", "bg_mask", "dec_objects_trans")
GRADS = ("dec_objects", "z_kp", "z_scale", "obj_on", "z_depth")
PARTICLE_INDEXED = GRADS
#: v2 envelopes these; v1 covered only the first two
ENVELOPED_V2 = ("z_kp", "z_scale", "z_depth")
#: v1 stored reference bounds for these -- reused verbatim, never regenerated
FROM_V1 = ("z_kp", "z_scale")
#: v1 had no envelope for this, so its reference bounds are generated here
REFERENCE_REGENERATED = ("z_depth",)
#: the accepted Triton paste this contract is widened to admit; asserted at runtime
TOLD_BLOB_SHA = "bd18b02c5e01c72a0715c12e1d5924cec21bafe4"
TOLD_SOURCE = "lpwm_stn/triton_backend.py"
#: predefined ladder for the z_depth relative-L2 threshold
L2_LADDER = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4)

_FIX = os.path.join(_HERE, "fixtures")
V1_PT = os.path.join(_FIX, "backward_envelope.pt")
V2_PT = os.path.join(_FIX, "backward_envelope_v2.pt")
V2_JSON = os.path.join(_FIX, "frozen_backward_policy_v2.json")


def composite_from_trans(trans, on, dep, want, return_alpha_masks=True):
    """The PyTorch reference composite, identical on both sides.

    Respects return_alpha_masks exactly as composite_reference does: when it is
    False the per-particle stack is not produced at all, so a masks_only scenario
    has no output and is skipped -- which is what v1 did, and why v1 has no
    envelope entry for small_nomasks/masks_only.
    """
    a_obj, rgb_obj = torch.split(trans, [1, trans.shape[2] - 1], dim=2)
    a = on[:, :, None, None, None] * a_obj
    rgba = a * rgb_obj
    imp = a * torch.sigmoid(-dep[:, :, :, None, None])
    impn = imp / (torch.sum(imp, dim=1, keepdim=True) + EPS)
    outs = {"dec_objects_trans": (rgba * impn).sum(1),
            "bg_mask": 1.0 - (impn * a).sum(1)}
    if return_alpha_masks:
        outs["alpha_masks"] = impn * a
    return [(n, outs[n]) for n in want if outs.get(n) is not None]


def base_cots(ins, scenario, tag, paste_fn):
    trans = paste_fn(ins["z_kp"], ins["dec_objects"], ins["img_size"],
                     scale=ins["z_scale"], translation=None, scale_normalized=False)
    want = {"rgb_only": ["dec_objects_trans"], "bg_only": ["bg_mask"],
            "masks_only": ["alpha_masks"], "mixed": list(OUT_NAMES)}[scenario]
    outs = dict(composite_from_trans(trans, ins["obj_on"], ins["z_depth"], want,
                                     return_alpha_masks=ins["return_alpha_masks"]))
    return [(n, cot(outs[n], f"{tag}/{n}")) for n in want if outs.get(n) is not None]


def chain_grads(paste_fn, ins, cots, order=None):
    """Full forward+backward through `paste_fn` + the reference composite."""
    ins = dict(ins)
    inv = None
    if order is not None:
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        for k in PARTICLE_INDEXED:
            if ins.get(k) is not None:
                ins[k] = ins[k][:, order].contiguous()
    p = ins["dec_objects"].detach().clone().requires_grad_(True)
    k = ins["z_kp"].detach().clone().requires_grad_(True)
    s = None if ins["z_scale"] is None else ins["z_scale"].detach().clone().requires_grad_(True)
    on = ins["obj_on"].detach().clone().requires_grad_(True)
    dep = ins["z_depth"].detach().clone().requires_grad_(True)
    trans = paste_fn(k, p, ins["img_size"], scale=s, translation=None, scale_normalized=False)
    cmap = dict(cots)
    sel = composite_from_trans(trans, on, dep, [n for n, _ in cots],
                               return_alpha_masks=ins["return_alpha_masks"])
    cots_use = []
    for n, _ in sel:
        c = cmap[n]
        cots_use.append(c[:, order].contiguous() if (order is not None and c.dim() == 5) else c)
    wrt = [p, k] + ([s] if s is not None else []) + [on, dep]
    gs = torch.autograd.grad([t for _, t in sel], wrt, grad_outputs=cots_use, allow_unused=True)
    names = ["dec_objects", "z_kp"] + (["z_scale"] if s is not None else []) + ["obj_on", "z_depth"]
    out = {}
    for nm, g in zip(names, gs):
        out[nm] = None if g is None else (g[:, inv].contiguous() if inv is not None else g)
    out.setdefault("z_scale", None)
    return out


def assert_told_provenance(root):
    """Pin the exact T_old source this contract was widened to admit."""
    import subprocess
    sha = subprocess.check_output(["git", "-C", root, "hash-object", TOLD_SOURCE],
                                  text=True).strip()
    if sha != TOLD_BLOB_SHA:
        raise SystemExit(f"T_old source changed: {TOLD_SOURCE} is {sha}, expected {TOLD_BLOB_SHA}. "
                         f"The v2 envelope is only valid for the pinned implementation.")
    return sha


def bounds_over_orders(paste_fn, ins, cots, orders, K, device, only=None):
    """min/max per element over a set of particle orders, in canonical order."""
    acc = {}
    for name, spec in orders:
        o = make_order(name, spec, K, device)
        g = chain_grads(paste_fn, ins, cots, order=o)
        for e in (only or ENVELOPED_V2):
            if g.get(e) is None:
                continue
            if e not in acc:
                acc[e] = [g[e].clone(), g[e].clone()]
            else:
                acc[e][0] = torch.minimum(acc[e][0], g[e])
                acc[e][1] = torch.maximum(acc[e][1], g[e])
    return acc


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    v1 = torch.load(V1_PT, weights_only=False)
    told_sha = assert_told_provenance(_ROOT)
    env_v2, told_heldout, ref_heldout = {}, [], []
    l2_dev, l2_heldout = {g: 0.0 for g in GRADS}, {g: 0.0 for g in GRADS}
    widened = {e: 0 for e in ENVELOPED_V2}
    total = {e: 0 for e in ENVELOPED_V2}
    other_fail = []

    for cname, kw in CASES:
        K = kw["n_kp"]
        ins = make_inputs(device="cuda", **kw)
        dev_t = ins["z_kp"].device
        for sc in SCENARIOS:
            cots = base_cots(ins, sc, cname, reference.stn_paste)
            if not cots:
                continue
            r_canon = chain_grads(reference.stn_paste, ins, cots)

            # reference regenerated ONLY for z_depth; z_kp/z_scale come from v1
            r_dev = bounds_over_orders(reference.stn_paste, ins, cots, DEV_SPEC, K, dev_t,
                                       only=REFERENCE_REGENERATED)
            t_dev = bounds_over_orders(triton_backend.stn_paste, ins, cots, DEV_SPEC, K, dev_t)

            for e in ENVELOPED_V2:
                if e not in t_dev:
                    continue
                key = f"{cname}|{sc}|{e}"
                v1key = key if key in v1["env"] else None
                if e in FROM_V1:
                    assert v1key is not None, f"v1 envelope missing {key}"
                    lo1 = v1["env"][v1key]["lower"].to("cuda")   # reused, not recomputed
                    hi1 = v1["env"][v1key]["upper"].to("cuda")
                else:
                    assert e in REFERENCE_REGENERATED
                    lo1, hi1 = r_dev[e][0], r_dev[e][1]
                lo2 = torch.minimum(lo1, t_dev[e][0])
                hi2 = torch.maximum(hi1, t_dev[e][1])
                widened[e] += int(((lo2 < lo1) | (hi2 > hi1)).sum())
                total[e] += int(lo2.numel())
                env_v2[key] = (lo2, hi2)

            # reference held-out check for the newly generated z_depth bounds
            for oname, spec in HELDOUT_SPEC:
                o = make_order(oname, spec, K, dev_t)
                g = chain_grads(reference.stn_paste, ins, cots, order=o)
                for e in REFERENCE_REGENERATED:
                    key = f"{cname}|{sc}|{e}"
                    if g.get(e) is None or key not in env_v2:
                        continue
                    lo, hi = env_v2[key]
                    delta = ATOL["z_kp"] + RTOL * torch.maximum(lo.abs(), hi.abs())
                    dist = torch.clamp((lo - delta) - g[e], min=0) + torch.clamp(g[e] - (hi + delta), min=0)
                    if bool((dist > 0).any()):
                        ref_heldout.append((cname, sc, oname, e, float((dist / delta).max())))

            # T_old under HELD-OUT orders: validation, never used to build
            for oname, spec in HELDOUT_SPEC:
                o = make_order(oname, spec, K, dev_t)
                g = chain_grads(triton_backend.stn_paste, ins, cots, order=o)
                for e in ENVELOPED_V2:
                    key = f"{cname}|{sc}|{e}"
                    if g.get(e) is None or key not in env_v2:
                        continue
                    lo, hi = env_v2[key]
                    delta = ATOL["z_kp"] + RTOL * torch.maximum(lo.abs(), hi.abs())
                    dist = torch.clamp((lo - delta) - g[e], min=0) + torch.clamp(g[e] - (hi + delta), min=0)
                    if bool((dist > 0).any()):
                        told_heldout.append((cname, sc, oname, e, float((dist / delta).max())))
                for gname in GRADS:
                    if g.get(gname) is None or r_canon.get(gname) is None:
                        continue
                    l2 = float(torch.linalg.vector_norm(g[gname] - r_canon[gname]) /
                               max(float(torch.linalg.vector_norm(r_canon[gname])), TINY))
                    l2_heldout[gname] = max(l2_heldout[gname], l2)
                # elementwise policies kept for dec_objects / obj_on
                for gname in ("dec_objects", "obj_on"):
                    if g.get(gname) is None:
                        continue
                    d = (g[gname] - r_canon[gname]).abs()
                    bound = ATOL[gname] + RTOL * r_canon[gname].abs()
                    if bool((d > bound).any()):
                        other_fail.append((cname, sc, oname, gname, float((d / bound).max())))

            # T_old development-order L2 spread (used for the z_depth ladder)
            for oname, spec in DEV_SPEC:
                o = make_order(oname, spec, K, dev_t)
                g = chain_grads(triton_backend.stn_paste, ins, cots, order=o)
                for gname in GRADS:
                    if g.get(gname) is None or r_canon.get(gname) is None:
                        continue
                    l2 = float(torch.linalg.vector_norm(g[gname] - r_canon[gname]) /
                               max(float(torch.linalg.vector_norm(r_canon[gname])), TINY))
                    l2_dev[gname] = max(l2_dev[gname], l2)
        del ins
        torch.cuda.empty_cache()

    # z_depth L2 threshold from the ladder, chosen on DEVELOPMENT data only
    z_thresh = next((v for v in L2_LADDER if l2_dev["z_depth"] <= v), None)
    if z_thresh is None:
        raise SystemExit(f"z_depth dev L2 {l2_dev['z_depth']:.3e} exceeds the ladder")

    print(f"{'grad':<13} {'T_old dev L2':>14} {'T_old heldout L2':>18}")
    for g in GRADS:
        print(f"{g:<13} {l2_dev[g]:>14.3e} {l2_heldout[g]:>18.3e}")
    print(f"\nz_depth L2 threshold from ladder (dev-only): {z_thresh:.0e}")
    print(f"z_depth held-out max L2: {l2_heldout['z_depth']:.3e}  "
          f"{'PASS' if l2_heldout['z_depth'] <= z_thresh else 'FAIL'}")
    print(f"\nenvelope widening vs v1:")
    for e in ENVELOPED_V2:
        print(f"  {e:<9} {widened[e]}/{total[e]} elements widened")
    print(f"\nT_old held-out envelope violations: {len(told_heldout)}")
    for r in told_heldout[:10]:
        print(f"   {r[0]}/{r[1]}/{r[2]} grad[{r[3]}] ratio={r[4]:.3f}")
    print(f"reference held-out violations (z_depth): {len(ref_heldout)}")
    for r in ref_heldout[:10]:
        print(f"   {r[0]}/{r[1]}/{r[2]} grad[{r[3]}] ratio={r[4]:.3f}")
    print(f"dec_objects/obj_on held-out violations: {len(other_fail)}")
    for r in other_fail[:10]:
        print(f"   {r[0]}/{r[1]}/{r[2]} grad[{r[3]}] ratio={r[4]:.3f}")

    if told_heldout or other_fail or ref_heldout:
        raise SystemExit("v2 NOT validated -- reporting rather than adjusting")

    out = {}
    for key, (lo, hi) in env_v2.items():
        out[key] = {"lower": lo.detach().to(device="cpu", dtype=torch.float32).contiguous().clone(),
                    "upper": hi.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()}
    torch.save({"version": 2, "env": out}, V2_PT)
    meta = {
        "version": 2,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supersedes": "frozen_backward_policy.json (v1)",
        "why_v1_was_incomplete": (
            "v1 was built from particle reorderings of a single paste implementation and "
            "therefore omitted cross-implementation paste-VJP spread. The saved counterfactual "
            "showed it rejects the already-accepted Triton paste itself (grad[z_kp] max ratio "
            "29.701), including when both sides are fed a byte-identical dL/dsample, which "
            "places the disagreement in the paste VJP rather than the composite feed."),
        "derived_from": ["reference.stn_paste + PyTorch reference composite",
                         "triton_backend.stn_paste (already accepted) + the same composite"],
        "told_source": TOLD_SOURCE, "told_blob_sha": told_sha,
        "told_commit": "389608c",
        "reference_bounds_reused_from_v1": list(FROM_V1),
        "reference_bounds_regenerated": list(REFERENCE_REGENERATED),
        "paste_calls": "explicit reference.stn_paste / triton_backend.stn_paste; "
                       "the generic lpwm_stn.stn_paste dispatcher is never used",
        "never_derived_from": "the fused composite backward candidate",
        "enveloped_gradients": list(ENVELOPED_V2),
        "development_orders": [[n, s] for n, s in DEV_SPEC],
        "heldout_orders": [[n, s] for n, s in HELDOUT_SPEC],
        "atol": ATOL, "rtol": RTOL,
        "l2_tol": {g: (z_thresh if g == "z_depth" else L2_TOL) for g in GRADS},
        "l2_ladder": list(L2_LADDER),
        "z_depth_l2_selected_from": "T_old development data only",
        "measured": {"told_dev_l2": l2_dev, "told_heldout_l2": l2_heldout,
                     "widened_elements": widened, "total_elements": total},
        "validation": {"told_heldout_envelope_violations": 0,
                       "reference_heldout_violations": 0,
                       "dec_objects_obj_on_heldout_violations": 0},
        "unchanged_from_v1": ["dec_objects elementwise", "obj_on elementwise"],
    }
    json.dump(meta, open(V2_JSON, "w"), indent=2, default=float)
    print(f"\nwrote {V2_PT} ({os.path.getsize(V2_PT)} bytes)")
    print(f"wrote {V2_JSON} ({os.path.getsize(V2_JSON)} bytes)")


if __name__ == "__main__":
    main()
