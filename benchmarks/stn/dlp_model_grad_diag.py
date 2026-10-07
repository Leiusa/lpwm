#!/usr/bin/env python
"""
Whole-model gradient diagnosis for single-image DLP: is the optimized path's gradient difference from the original
within what rounding-level perturbations already cause, and which component (Triton STN or fused composite) is
responsible for the largest differences? Measurement only. One CUDA GPU.

Every run: same weights, same fixed batch, model.eval() + deterministic=True + with_loss=True (posterior means, no
sampling, no dropout), fp32, matmul TF32 off; cuDNN TF32 off unless stated; cudnn deterministic, benchmark off.
  orig        reference STN, unfused composite                 (the original code)
  orig_again  orig repeated                                    (run-to-run nondeterminism)
  orig_pert   orig with the input multiplied by (1 + 1e-6 * N(0,1))  (how much a rounding-sized change moves things)
  orig_tf32   orig with cuDNN TF32 ON = PyTorch's default, i.e. how the original is actually trained
  opt         Triton STN + fused composite                     (the optimized code)
  opt_again   opt repeated
  stn_only    Triton STN, unfused composite
  fused_only  reference STN, fused composite
Per parameter gradient (and loss / outputs): relL2(run - orig) = ||run - orig|| / ||orig||, plus ||orig||.

Rule, fixed before running: the optimized path is consistent if, for every tensor,
  relL2(opt - orig) <= 10 * max(relL2(orig_pert - orig), relL2(orig_again - orig)) + 1e-7.
Tensors outside it are listed with every column, so the responsible component and the gradient's size are visible.

    cd ~/dlp-lpwm-optimization && python lpwm/benchmarks/stn/dlp_model_grad_diag.py \
        --config static_bair128_vgg_lambda.json --out results/grad_diag
"""
import argparse
import json
import os
import random
import sys
import tempfile

RUNS = ("orig", "orig_again", "orig_pert", "orig_tf32", "opt", "opt_again", "stn_only", "fused_only")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--n-kp-enc", default="all", help='"all" (no filtering), a number, or "" for the config value')
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="grad_diag")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    sys.path.insert(0, repo)
    import numpy as np
    import torch
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from utils.loss_functions import LossLPIPS, calc_reconstruction_loss

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(args.out, exist_ok=True)
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

    m_unfused = make(False)
    m_fused = make(True)
    m_fused.load_state_dict(m_unfused.state_dict())
    ds = get_image_dataset(cfg["ds"], cfg["root"], mode="train", image_size=cfg["image_size"])
    x = torch.stack([ds[i][0] for i in range(args.batch_size)]).cuda()
    if x.dim() == 4:
        x = x.unsqueeze(1)
    g = torch.Generator(device="cpu").manual_seed(0)
    x_pert = x * (1 + 1e-6 * torch.randn(x.shape, generator=g).cuda())
    rf = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).cuda() if cfg["recon_loss_type"] == "vgg" else calc_reconstruction_loss
    kw = dict(deterministic=True, warmup=False, with_loss=True, return_alpha_masks=True, beta_kl=cfg["beta_kl"],
              beta_rec=cfg["beta_rec"], kl_balance=cfg["kl_balance"], recon_loss_type=cfg["recon_loss_type"],
              beta_obj=cfg.get("beta_obj", 0.0), recon_loss_func=rf)

    def run(name):
        fused = name in ("opt", "opt_again", "fused_only")
        backend = "triton" if name in ("opt", "opt_again", "stn_only") else "reference"
        model = m_fused if fused else m_unfused
        lpwm_stn.set_backend(backend)
        torch.backends.cudnn.allow_tf32 = (name == "orig_tf32")
        model.eval()
        model.zero_grad(set_to_none=True)
        calls0 = model.decoder_module.fused_composite_calls
        out = model(x_pert if name == "orig_pert" else x, **kw)
        loss = out["loss_dict"]["loss"]
        loss.backward()
        res = {"loss": loss.detach().reshape(1).clone()}
        for k in ("rec_rgb", "rec", "obj_on", "mu", "mu_tot", "z_base_var"):
            if k in out and torch.is_tensor(out[k]):
                res[f"out.{k}"] = out[k].detach().clone()
        for n, p in model.named_parameters():
            if p.grad is not None:
                res[f"grad.{n}"] = p.grad.detach().clone()
        torch.backends.cudnn.allow_tf32 = False
        assert (model.decoder_module.fused_composite_calls - calls0) == int(fused), f"{name}: fused path not as requested"
        return res

    results = {name: run(name) for name in RUNS}
    ref = results["orig"]

    def rel(a, b):
        nb = torch.linalg.vector_norm(b.double()).item()
        nd = torch.linalg.vector_norm((a - b).double()).item()
        return nd / nb if nb > 0 else nd

    table = {}
    for k, v in ref.items():
        row = {"norm_orig": torch.linalg.vector_norm(v.double()).item(), "numel": v.numel()}
        for name in RUNS[1:]:
            row[name] = rel(results[name][k], v) if k in results[name] else None
        row["yardstick"] = max(row["orig_pert"], row["orig_again"])
        row["consistent"] = row["opt"] <= 10 * row["yardstick"] + 1e-7
        table[k] = row
    bad = {k: r for k, r in table.items() if not r["consistent"]}
    print(f"{torch.cuda.get_device_name(0)} | particles prior {m_unfused.n_kp_prior}, encoder {m_unfused.n_kp_enc}, "
          f"decoder {m_unfused.n_kp_dec} | batch {args.batch_size} | {len(table)} tensors")
    print("loss: " + " | ".join(f"{n} {results[n]['loss'].item():.9f}" for n in RUNS))
    cols = ("norm_orig", "orig_again", "orig_pert", "orig_tf32", "opt", "opt_again", "stn_only", "fused_only")
    print("\nrelL2 vs orig (norm_orig = ||gradient|| of the original); sorted by opt")
    print(f"{'tensor':58s} " + " ".join(f"{c:>10s}" for c in cols))
    for k, r in sorted(table.items(), key=lambda kv: -kv[1]["opt"])[:args.top]:
        print(f"{k[:58]:58s} " + " ".join(f"{r[c]:10.2e}" for c in cols) + ("" if r["consistent"] else "  <-- outside"))
    med = lambda c: float(np.median([r[c] for r in table.values() if r[c] is not None]))  # noqa: E731
    print("\nmedian relL2 over all tensors: " + " | ".join(f"{c} {med(c):.2e}" for c in cols[1:]))
    print(f"\nVERDICT: {'CONSISTENT' if not bad else 'NOT CONSISTENT'} | {len(bad)} of {len(table)} tensors outside "
          f"10 x max(orig_pert, orig_again) + 1e-7")
    json.dump({"rule": "opt <= 10 * max(orig_pert, orig_again) + 1e-7", "table": table,
               "loss": {n: results[n]["loss"].item() for n in RUNS}}, open(os.path.join(args.out, "grad_diag.json"), "w"),
              indent=1)
    print(f"wrote {args.out}/grad_diag.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
