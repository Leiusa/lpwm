#!/usr/bin/env python
"""
Are the custom Triton kernels (STN crop, STN paste, fused paste + composite) correct at many particles, heavy overlap
and edge cases, and does the whole single-image DLP model with them compute what the original computes?
Measurement only; nothing in the repository changes. Needs one CUDA GPU.

Ground truth = the ORIGINAL implementation evaluated in float64. For every output and every gradient we measure
  err_ref32 = relL2(original fp32  - original fp64)    (how far the original fp32 code is from the exact answer)
  err_new   = relL2(optimized fp32 - original fp64)    (how far the optimized fp32 code is)
relL2(a - b) = ||a - b|| / ||b||. Matmul TF32 and cuDNN TF32 are OFF throughout, so kernel errors are not hidden.

Pass rule, fixed before running: a tensor passes if it is finite and err_new <= 10 * err_ref32 + 1e-7.
(A wrong formula, a wrong index or a race gives errors of 1e-3..1, orders of magnitude above fp32 rounding.)
Verdict per part: PASS if every tensor passes, else FAIL with the offending tensors listed.

Part 1 -- ops, random cotangents, gradients w.r.t. every float input, per scenario (K particles, default 256):
  random     centers uniform in [-1, 1], glimpse size around the default
  overlap    all centers within +-0.05 (every glimpse covers the same pixels)
  border     centers in +-[0.9, 1.0], enlarged glimpses (cut by the image border)
  tiny       glimpses far below one pixel (scale logit -5)
  huge       glimpses covering almost the whole image (scale logit +3)
  ties       composite: all depths equal, obj_on exactly 0 or 1
  all_off    composite: obj_on = 0 everywhere (normalisation dominated by eps)
Part 2 -- whole model, the config's single-image DLP with n_kp_enc (default "all" = no filtering), one fixed batch,
  model.eval() + deterministic=True (posterior means, no sampling noise, no dropout) with_loss=True, so float64 and
  float32 compute the same function: original (reference STN, unfused) vs optimized (Triton STN + fused composite),
  identical weights; loss, outputs and every parameter gradient.

    cd ~/dlp-lpwm-optimization && python lpwm/benchmarks/stn/dlp_kernel_stress_check.py \
        --config static_bair128_vgg_lambda.json --out results/kernel_stress
"""
import argparse
import json
import math
import os
import sys
import tempfile

RULE_FACTOR, RULE_FLOOR = 10.0, 1e-7


def rel(a, b):
    import torch
    a, b = a.double(), b.double()
    nb = torch.linalg.vector_norm(b).item()
    nd = torch.linalg.vector_norm(a - b).item()
    return nd / nb if nb > 0 else nd


def judge(name, new32, ref32, ref64, rows):
    import torch
    finite = bool(torch.isfinite(new32).all()) and bool(torch.isfinite(ref32).all())
    e_new, e_ref = rel(new32, ref64), rel(ref32, ref64)
    ok = finite and e_new <= RULE_FACTOR * e_ref + RULE_FLOOR
    rows.append({"tensor": name, "err_new": e_new, "err_ref32": e_ref, "ratio": e_new / e_ref if e_ref > 0 else None,
                 "max_abs_new_vs_ref64": (new32.double() - ref64).abs().max().item(), "finite": finite, "pass": ok})
    return ok


def part1(args, torch):
    from lpwm_stn import reference, triton_backend
    from lpwm_stn.composite_reference import composite_reference
    from lpwm_stn.composite_autograd import composite_fused
    dev = "cuda"
    B, K, S, P = args.batch_size, args.particles, args.img_size, args.patch_size
    g = torch.Generator(device="cpu").manual_seed(0)
    logit = lambda p: math.log(p / (1 - p))  # noqa: E731
    base = logit(P / S)

    def scen(name):
        u = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
        n = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
        kp = u(B, K, 2) * 2 - 1
        sc = base + 0.5 * n(B, K, 2)
        obj = u(B, K)
        dep = n(B, K, 1)
        if name == "overlap":
            kp = (u(B, K, 2) * 2 - 1) * 0.05
        elif name == "border":
            kp = (0.9 + 0.1 * u(B, K, 2)) * torch.where(u(B, K, 2) < 0.5, -1.0, 1.0)
            sc = sc + 1.0
        elif name == "tiny":
            sc = torch.full((B, K, 2), -5.0) + 0.1 * n(B, K, 2)
        elif name == "huge":
            sc = torch.full((B, K, 2), 3.0) + 0.1 * n(B, K, 2)
        elif name == "ties":
            dep = torch.zeros(B, K, 1)
            obj = (u(B, K) < 0.5).float()
        elif name == "all_off":
            obj = torch.zeros(B, K)
        return kp, sc, obj, dep

    results = {}
    for name in ("random", "overlap", "border", "tiny", "huge", "ties", "all_off"):
        kp, sc, obj, dep = scen(name)
        x = torch.rand(B, 3, S, S, generator=g)
        patches = torch.rand(B, K, 4, P, P, generator=g)
        rows = []

        def run(fn, inputs, dtype):
            ins = [t.to(dev, dtype).requires_grad_(True) if t is not None else None for t in inputs]
            outs = fn(*ins)
            outs = [o for o in (outs if isinstance(outs, (tuple, list)) else (outs,)) if o is not None]
            gg = torch.Generator(device="cpu").manual_seed(1)
            loss = sum((o * torch.randn(o.shape, generator=gg).to(dev, o.dtype)).sum() for o in outs)
            grads = torch.autograd.grad(loss, [t for t in ins if t is not None], allow_unused=True)
            return [o.detach() for o in outs], [gr.detach() if gr is not None else torch.zeros_like(t) for gr, t in
                                                zip(grads, [t for t in ins if t is not None])]

        ops = []
        if name not in ("ties", "all_off"):
            ops.append(("crop", [x, kp, sc],
                        lambda a, b, c: reference.stn_crop(a, b, P, z_scale=c, padding_mode="border"),
                        lambda a, b, c: triton_backend.stn_crop(a, b, P, z_scale=c, padding_mode="border"),
                        ["out"], ["d_image", "d_kp", "d_scale"]))
            ops.append(("paste", [kp, patches, sc],
                        lambda a, b, c: reference.stn_paste(a, b, S, scale=c),
                        lambda a, b, c: triton_backend.stn_paste(a, b, S, scale=c),
                        ["out"], ["d_kp", "d_patches", "d_scale"]))
        ops.append(("composite", [patches, kp, sc, obj, dep],
                    lambda a, b, c, d, e: composite_reference(a, b, c, d, e, S, return_alpha_masks=True),
                    lambda a, b, c, d, e: composite_fused(a, b, c, d, e, S, return_alpha_masks=True),
                    ["alpha_masks", "bg_mask", "rgb"], ["d_dec_objects", "d_kp", "d_scale", "d_obj_on", "d_depth"]))
        for op, inputs, f_ref, f_new, out_names, grad_names in ops:
            o64, g64 = run(f_ref, inputs, torch.float64)
            o32, g32 = run(f_ref, inputs, torch.float32)
            on, gn = run(f_new, inputs, torch.float32)
            for nm, a, b, c in zip(out_names, on, o32, o64):
                judge(f"{op}.{nm}", a, b, c, rows)
            for nm, a, b, c in zip(grad_names, gn, g32, g64):
                judge(f"{op}.{nm}", a, b, c, rows)
            del o64, g64, o32, g32, on, gn
            torch.cuda.empty_cache()
        results[name] = rows
    return results


def part2(args, torch):
    import random
    import numpy as np
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from utils.loss_functions import LossLPIPS, calc_reconstruction_loss
    cfg = json.load(open(args.config))
    if args.n_kp_enc == "all":
        total = cfg.get("n_kp_per_patch", 1) * (cfg["image_size"] // cfg["patch_size"]) ** 2
        cfg["n_kp_enc"] = min(total, cfg.get("n_kp_prior", total))
    elif args.n_kp_enc:
        cfg["n_kp_enc"] = int(args.n_kp_enc)

    def make(fused):
        c = dict(cfg, fused_composite=fused, particle_dec_channels_last=False)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cfg.json")
            json.dump(c, open(p, "w"))
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            m, _, _ = build(p, "cuda")
        return m

    m_ref = make(False)
    m_new = make(True)
    m_new.load_state_dict(m_ref.state_dict())
    m64 = make(False)
    m64.load_state_dict(m_ref.state_dict())
    m64 = m64.double()
    info = {"n_kp_prior": m_ref.n_kp_prior, "n_kp_enc": m_ref.n_kp_enc, "n_kp_dec": m_ref.n_kp_dec,
            "batch_size": args.model_batch_size}
    ds = get_image_dataset(cfg["ds"], cfg["root"], mode="train", image_size=cfg["image_size"])
    x = torch.stack([ds[i][0] for i in range(args.model_batch_size)]).cuda().unsqueeze(1)
    if cfg["recon_loss_type"] == "vgg":
        rf32 = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).cuda()
        rf64 = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).cuda().double()
    else:
        rf32 = rf64 = calc_reconstruction_loss
    kw = dict(deterministic=True, warmup=False, with_loss=True, return_alpha_masks=True, beta_kl=cfg["beta_kl"],
              beta_rec=cfg["beta_rec"], kl_balance=cfg["kl_balance"], recon_loss_type=cfg["recon_loss_type"],
              beta_obj=cfg.get("beta_obj", 0.0))

    def run(model, backend, xin, rf):
        lpwm_stn.set_backend(backend)
        model.eval()
        model.zero_grad(set_to_none=True)
        out = model(xin, recon_loss_func=rf, **kw)
        loss = out["loss_dict"]["loss"]
        loss.backward()
        outs = {"loss": loss.detach().reshape(1)}
        for k in ("rec_rgb", "rec", "dec_objects", "alpha_masks", "obj_on", "mu_tot", "mu", "z_depth", "mu_scale"):
            if k in out and torch.is_tensor(out[k]):
                outs[k] = out[k].detach()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return outs, grads

    o32, g32 = run(m_ref, "reference", x, rf32)
    on, gn = run(m_new, "triton", x, rf32)
    info["fused_composite_calls"] = m_new.decoder_module.fused_composite_calls
    try:
        o64, g64 = run(m64, "reference", x.double(), rf64)
    except Exception as exc:  # the model may hard-code float32 somewhere: report the fp32 comparison without a verdict
        info["fp64_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        info["loss"] = {"original_fp32": o32["loss"].item(), "optimized_fp32": on["loss"].item()}
        rows = [{"tensor": f"{kind}.{k}", "relL2_optimized_vs_original_fp32": rel(b[k], a[k])}
                for kind, a, b in (("out", o32, on), ("grad", g32, gn)) for k in a if k in b]
        return info, rows
    info["loss"] = {"original_fp64": o64["loss"].item(), "original_fp32": o32["loss"].item(),
                    "optimized_fp32": on["loss"].item()}
    rows = []
    for k in o64:
        if k in o32 and k in on:
            judge(f"out.{k}", on[k], o32[k], o64[k], rows)
    missing = sorted(set(g64) ^ set(gn))
    for k in g64:
        if k in g32 and k in gn:
            judge(f"grad.{k}", gn[k], g32[k], g64[k], rows)
    info["params_with_grad"] = len(g64)
    info["grad_name_mismatch"] = missing
    return info, rows


def report(title, rows, top=8):
    bad = [r for r in rows if not r["pass"]]
    ratios = sorted((r["ratio"] for r in rows if r["ratio"] is not None), reverse=True)
    print(f"{title}: {'PASS' if not bad else 'FAIL'} | {len(rows)} tensors, {len(bad)} failing | worst err_new/err_ref32 "
          f"{ratios[0] if ratios else float('nan'):.2f}")
    for r in (bad or sorted(rows, key=lambda r: -(r["ratio"] or 0)))[:top if bad else 3]:
        print(f"    {r['tensor'][:60]:60s} err_new {r['err_new']:.2e}  err_ref32 {r['err_ref32']:.2e}  "
              f"finite {r['finite']}  {'ok' if r['pass'] else 'FAIL'}")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--particles", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4, help="part 1 batch")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--n-kp-enc", default="all", help='part 2: "all" (no filtering), a number, or "" for the config value')
    ap.add_argument("--model-batch-size", type=int, default=4)
    ap.add_argument("--skip-part2", action="store_true")
    ap.add_argument("--out", default="kernel_stress")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    sys.path.insert(0, repo)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(args.out, exist_ok=True)
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | part 1: {args.particles} particles, batch "
          f"{args.batch_size}, image {args.img_size}, glimpse {args.patch_size} | rule err_new <= {RULE_FACTOR:g} * "
          f"err_ref32 + {RULE_FLOOR:g}")
    res = {"rule": f"err_new <= {RULE_FACTOR} * err_ref32 + {RULE_FLOOR}", "part1": part1(args, torch)}
    ok = True
    for scen, rows in res["part1"].items():
        ok &= report(f"part 1 [{scen}]", rows)
    if not args.skip_part2:
        info, rows = part2(args, torch)
        res["part2"] = {"info": info, "rows": rows}
        if "fp64_error" in info:
            print(f"part 2: float64 run failed ({info['fp64_error']}); NO VERDICT. optimized vs original fp32, worst:")
            for r in sorted(rows, key=lambda r: -r["relL2_optimized_vs_original_fp32"])[:8]:
                print(f"    {r['tensor'][:60]:60s} relL2 {r['relL2_optimized_vs_original_fp32']:.2e}")
            print(f"   loss original fp32 {info['loss']['original_fp32']:.9f} | optimized fp32 {info['loss']['optimized_fp32']:.9f}")
            json.dump(res, open(os.path.join(args.out, "kernel_stress.json"), "w"), indent=1)
            print(f"OVERALL: part 1 {'PASS' if ok else 'FAIL'}, part 2 no verdict | wrote {args.out}/kernel_stress.json")
            return 0 if ok else 1
        print(f"part 2 model: particles prior {info['n_kp_prior']}, encoder {info['n_kp_enc']}, decoder "
              f"{info['n_kp_dec']}, batch {info['batch_size']}, fused calls {info['fused_composite_calls']}, "
              f"{info['params_with_grad']} params with grad, name mismatch {info['grad_name_mismatch']}")
        print(f"   loss fp64 {info['loss']['original_fp64']:.9f} | original fp32 {info['loss']['original_fp32']:.9f} | "
              f"optimized fp32 {info['loss']['optimized_fp32']:.9f}")
        ok &= report("part 2 [whole model]", rows) and not info["grad_name_mismatch"] and info["fused_composite_calls"] == 1
    json.dump(res, open(os.path.join(args.out, "kernel_stress.json"), "w"), indent=1)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'} | wrote {args.out}/kernel_stress.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
