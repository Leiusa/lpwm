#!/usr/bin/env python
"""
Single-image DLP: how much faster could the glimpse / particle CNNs get with bf16 autocast (and channels-last)?
Measurement only; nothing in the repository changes. One CUDA GPU, nothing else running on it.

The model is built from the config exactly as dlp_profile.py does; one real training forward (eager, warmup=False)
records the input of every call to a CNN trunk (modules.vision_modules.Encoder / Decoder, i.e. the particle
encoders, the particle decoder, the background and context CNNs) at the real shapes (batch * particles patches).
Each recorded call is then benchmarked alone, forward + backward of the CNN with a fixed random output gradient,
gradients w.r.t. its input and parameters, in these variants:

  fp32       autocast off, cuDNN TF32 as PyTorch's default (allow_tf32=True) = how training runs today
  fp32_cl    fp32 with the CNN and its input in channels_last
  bf16       torch.autocast("cuda", torch.bfloat16) around the forward; backward outside it (standard AMP usage);
             GroupNorm, softmax etc. run in fp32 by autocast's own op lists
  bf16_cl    bf16 with the CNN and its input in channels_last
for each --modes entry: eager, and/or default (torch.compile of the CNN alone, mode "default").
cuDNN benchmark / deterministic and matmul TF32 come from the config (as train_dlp.py).

Reported per call and variant: median fwd+bwd ms and fwd ms (CUDA events; --repeats blocks of --iters iterations),
speedup vs fp32 in the same mode, extra peak memory of one iteration. Numerics, as context only (no threshold):
relL2 vs eager fp32 of the output, the input gradient and the median over parameter gradients; the row
"fp32_noTF32" (cuDNN TF32 off) shows how far PyTorch's default TF32 already is from strict fp32.
The total line sums all calls: compare it with the CNN share of the step in run_dlp_profile.sh's report.

    cd ~/dlp-lpwm-optimization && python lpwm/benchmarks/stn/dlp_cnn_bench.py --config static_bair128_vgg_lambda.json \
        --override n_kp_enc='"all"' --override cudnn_benchmark=true --out results/dlp_cnn_bench
"""
import argparse
import copy
import json
import os
import random
import statistics
import sys
import tempfile

import numpy as np

VARIANTS = ("fp32", "fp32_cl", "bf16", "bf16_cl")


def parse_override(s):
    k, v = s.split("=", 1)
    try:
        v = json.loads(v)
    except json.JSONDecodeError:
        pass
    return k, v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--override", action="append", default=[], help="config key=value (JSON value), repeatable")
    ap.add_argument("--modes", default="eager,default", help="comma list of eager, default")
    ap.add_argument("--warmup", type=int, default=10, help="untimed iterations (cuDNN benchmark trials, compilation)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="dlp_cnn_bench")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    sys.path.insert(0, repo)
    import torch
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from modules.vision_modules import Decoder, Encoder
    from utils.loss_functions import LossLPIPS, calc_reconstruction_loss

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(args.config))
    for o in args.override:
        k, v = parse_override(o)
        cfg[k] = v
    if cfg.get("n_kp_enc") == "all":   # same rule as models.py DLP.__init__
        total = cfg.get("n_kp_per_patch", 1) * (cfg["image_size"] // cfg["patch_size"]) ** 2
        cfg["n_kp_enc"] = min(total, cfg.get("n_kp_prior", total))
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(cfg.get("cudnn_deterministic", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("tf32_matmul", False))
    torch.backends.cudnn.allow_tf32 = True   # PyTorch default, as in training
    seed = cfg.get("seed", 0) or 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    lpwm_stn.set_backend(cfg.get("stn_backend") or "reference")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(p, "w"))
        model, _, _ = build(p, "cuda")
    model.train()

    # ---- record the inputs of every CNN trunk call in one real training forward
    names = {m: n for n, m in model.named_modules()}
    calls = []

    def hook(mod, inp):
        calls.append((names[mod], mod, inp[0].detach().clone()))

    hs = [m.register_forward_pre_hook(hook) for m in model.modules() if isinstance(m, (Encoder, Decoder))]
    rf = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).cuda() if cfg["recon_loss_type"] == "vgg" else calc_reconstruction_loss
    ds = get_image_dataset(cfg["ds"], cfg["root"], mode="train", image_size=cfg["image_size"])
    x = torch.stack([ds[i][0] for i in range(cfg["batch_size"])]).cuda()
    if x.dim() == 4:
        x = x.unsqueeze(1)
    model(x, warmup=False, with_loss=True, return_alpha_masks=False, beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"],
          kl_balance=cfg["kl_balance"], recon_loss_type=cfg["recon_loss_type"], recon_loss_func=rf,
          beta_obj=cfg.get("beta_obj", 0.0))
    for h in hs:
        h.remove()
    model.zero_grad(set_to_none=True)
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | cudnn_benchmark {torch.backends.cudnn.benchmark}"
          f" deterministic {torch.backends.cudnn.deterministic} | batch {cfg['batch_size']} | particles prior "
          f"{model.n_kp_prior}, encoder {model.n_kp_enc}, decoder {model.n_kp_dec} | {len(calls)} CNN calls")
    for i, (n, m, inp) in enumerate(calls):
        print(f"   [{i}] {n} ({type(m).__name__}): input {tuple(inp.shape)}, "
              f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params")

    def rel(a, b):
        nb = torch.linalg.vector_norm(b.double()).item()
        return torch.linalg.vector_norm((a.double() - b.double())).item() / nb if nb > 0 else float("nan")

    def run_call(idx, mod0, inp0, variant, mode, tf32=True):
        torch.backends.cudnn.allow_tf32 = tf32
        cl = variant.endswith("_cl")
        mod = copy.deepcopy(mod0)
        if cl:
            mod = mod.to(memory_format=torch.channels_last)
        inp = inp0.clone()
        if cl:
            inp = inp.contiguous(memory_format=torch.channels_last)
        inp.requires_grad_(True)
        dtype = torch.bfloat16 if variant.startswith("bf16") else None

        def fwd(t):
            if dtype is None:
                return mod(t)
            with torch.autocast("cuda", dtype=dtype):
                return mod(t)

        f = fwd
        if mode == "default":
            torch._dynamo.reset()
            f = torch.compile(fwd)
        g = torch.Generator(device="cuda").manual_seed(1000 + idx)
        out_shape = f(inp).shape
        gout = torch.randn(out_shape, device="cuda", generator=g)

        def it():
            out = f(inp)
            out.float().backward(gout)
            return out

        for _ in range(args.warmup):
            mod.zero_grad(set_to_none=True); inp.grad = None
            it()
        torch.cuda.synchronize()
        # numerics: one fresh iteration
        mod.zero_grad(set_to_none=True); inp.grad = None
        out = it().detach().float().contiguous()
        res = {"out": out, "gin": inp.grad.detach().float().contiguous(),
               "gp": [p.grad.detach().float().contiguous() if p.grad is not None else None for p in mod.parameters()]}
        # memory of one iteration
        mod.zero_grad(set_to_none=True); inp.grad = None
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        it()
        torch.cuda.synchronize()
        res["peak_mb"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        # timing
        tot, fw = [], []
        for _ in range(args.repeats):
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
            for e0, e1, e2 in ev:
                mod.zero_grad(set_to_none=True); inp.grad = None
                e0.record()
                o = f(inp)
                e1.record()
                o.float().backward(gout)
                e2.record()
            torch.cuda.synchronize()
            tot.append(statistics.median(a.elapsed_time(c) for a, b, c in ev))
            fw.append(statistics.median(a.elapsed_time(b) for a, b, c in ev))
        res["ms"], res["fwd_ms"] = statistics.median(tot), statistics.median(fw)
        torch.backends.cudnn.allow_tf32 = True
        del mod, inp
        torch.cuda.empty_cache()
        return res

    modes = [m for m in args.modes.split(",") if m]
    table, totals = {}, {}
    for i, (n, m, inp) in enumerate(calls):
        print(f"\n== [{i}] {n}  input {tuple(inp.shape)}")
        print(f"   {'variant':16s} {'fwd+bwd ms':>11s} {'fwd ms':>8s} {'speedup':>8s} {'peak MB':>9s}"
              f" {'rel out':>9s} {'rel gin':>9s} {'med rel gp':>11s}")
        ref = run_call(i, m, inp, "fp32", "eager")
        strict = run_call(i, m, inp, "fp32", "eager", tf32=False)
        rows = {}
        for mode in modes:
            for v in VARIANTS:
                key = f"{v}/{mode}"
                try:
                    r = ref if key == "fp32/eager" else run_call(i, m, inp, v, mode)
                except Exception as e:  # report and continue (e.g. a compile failure)
                    print(f"   {key:16s} FAILED: {type(e).__name__}: {str(e)[:150]}")
                    continue
                rows[key] = r
        rows["fp32_noTF32/eager"] = strict
        cell = {}
        for key, r in rows.items():
            base_key = "fp32/" + key.split("/")[1]
            speed = rows[base_key]["ms"] / r["ms"] if base_key in rows else float("nan")
            gps = [rel(a, b) for a, b in zip(r["gp"], ref["gp"]) if a is not None and b is not None]
            c = {"ms": r["ms"], "fwd_ms": r["fwd_ms"], "speedup": speed, "peak_mb": r["peak_mb"],
                 "rel_out": rel(r["out"], ref["out"]), "rel_gin": rel(r["gin"], ref["gin"]),
                 "med_rel_gp": float(np.median(gps)) if gps else float("nan")}
            cell[key] = c
            totals.setdefault(key, 0.0)
            totals[key] += r["ms"]
            print(f"   {key:16s} {c['ms']:11.2f} {c['fwd_ms']:8.2f} {c['speedup']:7.2f}x {c['peak_mb']:9.0f}"
                  f" {c['rel_out']:9.1e} {c['rel_gin']:9.1e} {c['med_rel_gp']:11.1e}")
        table[f"[{i}] {n}"] = {"input": list(inp.shape), "rows": cell}

    print("\n== total over all CNN calls (fwd+bwd ms; compare with the CNN share of the step in run_dlp_profile.sh)")
    for mode in modes:
        base = totals.get(f"fp32/{mode}")
        print("   " + mode + ": " + " | ".join(
            f"{v} {totals[f'{v}/{mode}']:.1f}" + (f" ({base / totals[f'{v}/{mode}']:.2f}x)" if base else "")
            for v in VARIANTS if f"{v}/{mode}" in totals))
    json.dump({"calls": table, "totals_ms": totals, "cfg_switches": {
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}},
        open(os.path.join(args.out, "cnn_bench.json"), "w"), indent=1)
    print(f"wrote {args.out}/cnn_bench.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
