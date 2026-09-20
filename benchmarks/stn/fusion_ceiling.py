#!/usr/bin/env python
"""
Bounded comparison of the STANDALONE fused composite against the incumbent chain, at one
operating point, on tensors captured from the real static-DLP forward. No model integration.

Three arms, all from the small RGBA patches to the reduced outputs and the five input gradients
(dec_objects, z_kp, z_scale, obj_on, z_depth), with alpha_masks=False and one shared upstream
gradient:

  ref    original PyTorch reference: reference.stn_paste + composite   (composite_reference)
  old    incumbent: Triton stn_paste + the model's own PyTorch composite (autograd)
  fused  candidate: composite_forward_triton + composite_backward_triton (no autograd, no canvas)

Correctness reports, per output and per gradient: fused vs ref (total difference), old vs ref
(the incumbent's difference) and fused vs old (what fusion adds), plus each arm against itself
(run-to-run floor). No pass/fail is declared: the frozen composite policies were built from
fixture cases (element-wise envelopes exist only for those) and no threshold is derived here.

Performance and memory are measured separately from any diagnostic instrumentation. The
measurement boundary is identical for every arm: the five input tensors and the upstream
gradients are alive before the clock starts; forward outputs and the five gradients are alive
when the peak is read. Peaks are read over the whole chain, never summed from isolated blocks.
"""

import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_e2e as be  # noqa: E402
from attribute_memory import Ledger  # noqa: E402

MB = 1e6
GRADS = ("dec_objects", "z_kp", "z_scale", "obj_on", "z_depth")


def cotangents(shapes, kind, device):
    """Same generator style as the frozen composite gate: seeded uniform [0,1); 'signed' shifts it."""
    import zlib
    out = {}
    for name, shape in shapes.items():
        g = torch.Generator(device=device)
        g.manual_seed(0x9317 + (zlib.crc32(f"real/{name}".encode()) & 0xFFFF))
        c = torch.rand(shape, generator=g, device=device)
        out[name] = c - 0.5 if kind == "signed" else c
    return out


def compare(a, b, frozen=None):
    """a: the arm being judged, b: what it is compared against. Diagnostic metrics only."""
    d = a.double() - b.double()
    ad = d.abs()
    nb = float(torch.linalg.vector_norm(b.double()))
    nd = float(torch.linalg.vector_norm(d))
    i = int(ad.argmax())
    sq = (d * d).reshape(-1)
    total_sq = float(sq.sum())
    rec = {"shape": list(a.shape), "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
           "max_abs": float(ad.max()), "mean_abs": float(ad.mean()), "ref_absmax": float(b.abs().max()),
           "rel_l2": (nd / nb) if nb > 0 else (0.0 if nd == 0 else None),
           "argmax_index": [int(v) for v in torch.unravel_index(torch.tensor(i), d.shape)],
           "value_a_at_argmax": float(a.reshape(-1)[i]), "value_b_at_argmax": float(b.reshape(-1)[i]),
           "top1_share_of_squared_error": (float(sq[i]) / total_sq) if total_sq > 0 else None}
    if frozen is not None:                      # context only: how many elements exceed the frozen scalars
        atol, rtol = frozen
        rec["n_elements_over_frozen_atol_rtol_context_only"] = int((ad > atol + rtol * b.double().abs()).sum())
    return rec


class RefArm:
    name = "ref"

    def __init__(self, inputs, img_size, dec):
        self.img, self.x = img_size, {k: v.detach().clone().requires_grad_(True) for k, v in inputs.items()}

    def forward(self):
        from lpwm_stn.composite_reference import composite_reference
        x = self.x
        _, bg, rgb = composite_reference(x["dec_objects"], x["z_kp"], x["z_scale"], x["obj_on"], x["z_depth"],
                                         self.img, return_alpha_masks=False)
        return {"bg_mask": bg, "dec_objects_trans": rgb}

    def backward(self, outs, cot):
        g = torch.autograd.grad([outs["bg_mask"], outs["dec_objects_trans"]], [self.x[k] for k in GRADS],
                                grad_outputs=[cot["bg_mask"], cot["dec_objects_trans"]])
        return dict(zip(GRADS, g))


class OldArm(RefArm):
    name = "old"

    def __init__(self, inputs, img_size, dec):
        super().__init__(inputs, img_size, dec)
        self.dec = dec
        self.paste_node = None

    def forward(self):
        x, dec = self.x, self.dec
        stack = dec.translate_patches(x["z_kp"], x["dec_objects"], x["z_scale"])
        self.paste_node = type(stack.grad_fn).__name__
        a_obj, rgb_obj = torch.split(stack, [1, stack.shape[2] - 1], dim=2)
        _, bg, rgb = dec.get_objects_alpha_rgb_with_depth(a_obj, rgb_obj, obj_on=x["obj_on"], z_depth=x["z_depth"],
                                                          return_alpha_masks=False)
        return {"bg_mask": bg, "dec_objects_trans": rgb}


class FusedArm:
    name = "fused"

    def __init__(self, inputs, img_size, dec):
        self.img, self.x = img_size, {k: v.detach().clone() for k, v in inputs.items()}

    def forward(self):
        from lpwm_stn.composite_triton import composite_forward_triton
        x = self.x
        _, bg, rgb = composite_forward_triton(x["dec_objects"], x["z_kp"], x["z_scale"], x["obj_on"], x["z_depth"],
                                              self.img, return_alpha_masks=False)
        return {"bg_mask": bg, "dec_objects_trans": rgb}

    def backward(self, outs, cot):
        from lpwm_stn.composite_backward import composite_backward_triton
        x = self.x
        g = composite_backward_triton(x["dec_objects"], x["z_kp"], x["z_scale"], x["obj_on"], x["z_depth"], self.img,
                                      None, cot["bg_mask"], cot["dec_objects_trans"])
        return dict(zip(GRADS, g))


def run_once(arm, cot):
    outs = arm.forward()
    grads = arm.backward(outs, cot)
    torch.cuda.synchronize()
    return outs, grads


def time_arm(arm, cot, iters, warmup):
    fwd, bwd = [], []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        outs = arm.forward()
        e1.record()
        grads = arm.backward(outs, cot)
        e2.record()
        torch.cuda.synchronize()
        if i >= warmup:
            fwd.append(e0.elapsed_time(e1))
            bwd.append(e1.elapsed_time(e2))
        del outs, grads
    tot = [f + b for f, b in zip(fwd, bwd)]
    q = lambda v: {"median": statistics.median(v), "min": min(v), "max": max(v)}  # noqa: E731
    return {"forward_ms": q(fwd), "backward_ms": q(bwd), "total_ms": q(tot), "iters": iters, "warmup": warmup,
            "spread_pct": 100 * (max(tot) - min(tot)) / statistics.median(tot)}


def memory_arm(arm, cot):
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outs = arm.forward()
    torch.cuda.synchronize()
    fwd_peak, retained = torch.cuda.max_memory_allocated() - base, torch.cuda.memory_allocated() - base
    grads = arm.backward(outs, cot)
    torch.cuda.synchronize()
    chain_peak, end = torch.cuda.max_memory_allocated() - base, torch.cuda.memory_allocated() - base
    del outs, grads
    return {"forward_peak_mb": fwd_peak / MB, "retained_after_forward_mb": retained / MB,
            "whole_chain_peak_mb": chain_peak / MB, "alive_at_end_outputs_plus_grads_mb": end / MB}


def input_stats(inp, outs):
    def s(t):
        return {"min": float(t.min()), "max": float(t.max()), "mean": float(t.mean())}
    return {"obj_on": s(inp["obj_on"]), "obj_on_gt_half_fraction": float((inp["obj_on"] > 0.5).float().mean()),
            "sigmoid_scale": s(torch.sigmoid(inp["z_scale"])), "z_kp": s(inp["z_kp"]), "z_depth": s(inp["z_depth"]),
            "patch_alpha": s(inp["dec_objects"][:, :, 0]), "patch_rgb": s(inp["dec_objects"][:, :, 1:]),
            "bg_mask_mean_of_reference_output": float(outs["bg_mask"].mean()),
            "shapes": {k: list(v.shape) for k, v in inp.items()},
            "strides": {k: list(v.stride()) for k, v in inp.items()},
            "contiguous": {k: bool(v.is_contiguous()) for k, v in inp.items()},
            "dtypes": {k: str(v.dtype) for k, v in inp.items()}}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    repo_root = be._bootstrap(args.repo_root)
    config = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)

    import lpwm_stn
    from build_model import build, sequence_length
    from lpwm_stn.composite_triton import can_use_triton_composite

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, cfg, _ = build(config)
    if model.is_dynamics_model:
        raise SystemExit("static DLP path only")
    g = torch.Generator(device="cuda")
    g.manual_seed(args.seed)
    x = torch.rand(args.batch_size, sequence_length(cfg), cfg["ch"], cfg["image_size"], cfg["image_size"],
                   generator=g, device="cuda")
    betas = {"beta_kl": cfg.get("beta_kl", 0.1), "beta_dyn": cfg.get("beta_dyn", 0.1), "beta_rec": cfg.get("beta_rec", 1.0)}
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 2e-4))
    weights_sha, input_sha = be.tensors_sha256(model.parameters()), be.tensors_sha256([x])
    lpwm_stn.set_backend("triton")

    for _ in range(args.warmup_steps):          # same capture point as attribute_memory.py
        model.train()
        opt.zero_grad(set_to_none=True)
        be.loss_of(model(x, deterministic=False, with_loss=True, return_alpha_masks=False, **betas)).backward()
        opt.step()

    dec = model.decoder_module
    cap = {}
    o_paste, o_comp = dec.translate_patches, dec.get_objects_alpha_rgb_with_depth

    def cap_paste(*a, **k):
        cap["z_kp"], cap["dec_objects"], cap["z_scale"] = (t.detach() for t in a[:3])
        return o_paste(*a, **k)

    def cap_comp(a_obj, rgb_obj, *a, **k):
        cap["obj_on"], cap["z_depth"] = k["obj_on"].detach(), k["z_depth"].detach()
        return o_comp(a_obj, rgb_obj, *a, **k)

    dec.translate_patches, dec.get_objects_alpha_rgb_with_depth = cap_paste, cap_comp
    try:
        model.train()
        torch.manual_seed(args.seed)
        with torch.no_grad():
            model(x, deterministic=False, with_loss=False, return_alpha_masks=False)
    finally:
        del dec.translate_patches, dec.get_objects_alpha_rgb_with_depth
    inputs = {k: cap[k] for k in GRADS}
    img = int(dec.feature_map_size)
    captured_contiguous = {k: bool(v.is_contiguous()) for k, v in inputs.items()}

    arms = {"ref": RefArm(inputs, img, dec), "old": OldArm(inputs, img, dec), "fused": FusedArm(inputs, img, dec)}
    gate = can_use_triton_composite(arms["fused"].x["dec_objects"], arms["fused"].x["z_kp"], arms["fused"].x["z_scale"],
                                    arms["fused"].x["obj_on"], arms["fused"].x["z_depth"], img)
    if not gate:
        raise SystemExit("captured inputs do not satisfy can_use_triton_composite")

    probe = arms["ref"].forward()
    shapes = {"bg_mask": tuple(probe["bg_mask"].shape), "dec_objects_trans": tuple(probe["dec_objects_trans"].shape)}
    stats_in = input_stats(inputs, probe)
    del probe

    # ---------------- correctness (no instrumentation) ----------------
    pol = json.load(open(os.path.join(repo_root, "tests/composite/fixtures/frozen_backward_policy_v2.json")))
    frozen = {k: (pol["atol"][k], pol["rtol"]) for k in GRADS}
    correctness = {}
    for kind in ("uniform", "signed"):
        cot = cotangents(shapes, kind, "cuda")
        res = {}
        for name, arm in arms.items():
            res[name] = [run_once(arm, cot) for _ in range(2)]        # two runs each: run-to-run floor
        pairs = {"old_vs_ref": ("old", "ref"), "fused_vs_ref": ("fused", "ref"), "fused_vs_old": ("fused", "old")}
        rec = {"cotangent": kind, "old_forward_paste_grad_fn": arms["old"].paste_node}
        for pname, (a, b) in pairs.items():
            outs_a, grads_a = res[a][0]
            outs_b, grads_b = res[b][0]
            rec[pname] = {"outputs": {k: compare(outs_a[k], outs_b[k]) for k in outs_a},
                          "grads": {k: compare(grads_a[k], grads_b[k], frozen[k]) for k in GRADS}}
        rec["self_run_to_run"] = {n: {"outputs": {k: compare(res[n][0][0][k], res[n][1][0][k]) for k in res[n][0][0]},
                                      "grads": {k: compare(res[n][0][1][k], res[n][1][1][k]) for k in GRADS}}
                                  for n in arms}
        correctness[kind] = rec
        del res
    cot = cotangents(shapes, "uniform", "cuda")

    # ---------------- clean timing and memory (no hooks) ----------------
    timing = {n: [] for n in arms}
    for order in (("ref", "old", "fused"), ("fused", "old", "ref")):
        for n in order:
            timing[n].append(time_arm(arms[n], cot, args.iters, args.warmup))
    memory = {n: [] for n in arms}
    for order in (("ref", "old", "fused"), ("fused", "old", "ref"), ("old", "fused", "ref")):
        for n in order:
            memory[n].append(memory_arm(arms[n], cot))

    # ---------------- diagnostics, kept separate from the numbers above ----------------
    ledger = Ledger({t.untyped_storage().data_ptr() for t in arms["old"].x.values()} |
                    {t.untyped_storage().data_ptr() for t in cot.values()})
    with ledger.hooks():
        outs = arms["old"].forward()
    diag_saved = ledger.summary(top=8)
    del outs

    report = {"schema_version": 1, "kind": "bounded standalone-fusion comparison on captured static-DLP tensors (no model integration)",
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "batch_size": args.batch_size, "seed": args.seed,
              "return_alpha_masks": False, "img_size": img,
              "capture": {"point": f"static DLP, triton backend, after {args.warmup_steps} Adam steps on torch.rand images; "
                                   "no_grad train-mode forward; NOT a trained model and NOT real data",
                          "initial_weights_sha256": weights_sha, "input_sha256": input_sha,
                          "captured_tensors_sha256": {k: be.tensors_sha256([v]) for k, v in inputs.items()},
                          "captured_contiguous": captured_contiguous, "can_use_triton_composite_on_leaves": gate},
              "input_statistics": stats_in,
              "acceptance_rules": {"applied": "none: no pass/fail is declared",
                                   "frozen_policy_scalars_shown_for_context": {"atol": pol["atol"], "rtol": pol["rtol"], "l2_tol": pol["l2_tol"]},
                                   "why_not_applied": "z_kp/z_scale/z_depth use per-element envelopes that exist only for the 47 fixture case/scenario "
                                                      "combinations; the scalar values were frozen for the fixture regime"},
              "correctness": correctness, "timing": timing, "memory": memory,
              "diagnostic_saved_tensors_old_chain_forward": diag_saved,
              "config": config, "config_sha256": be.file_sha256(config), "config_content": cfg,
              "script_sha256": be.file_sha256(os.path.abspath(__file__)),
              "repo_module_sha256": {n: be.file_sha256(os.path.join(repo_root, "lpwm_stn", n))
                                     for n in ("composite_triton.py", "composite_backward.py", "composite_reference.py",
                                               "triton_backend.py", "reference.py")},
              "env": be.env_record(), "repo": be.repo_record(repo_root)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
