#!/usr/bin/env python
"""
Where does an LPWM training step spend its time and memory? Measurement only; nothing in the repository changes.

Runs the train_lpwm.py training step (same model build, video DataLoader, vgg loss, Adam, forward arguments,
warmup=False, return_alpha_masks=False) eagerly for --warmup + --steps steps, then reports:

  1. per-module FORWARD time   CUDA events from forward pre-hooks / forward hooks on every module down to --depth
                               (e.g. encoder_module.particle_enc, decoder_module.particle_dec, dyn_module, ctx_module),
                               plus the vgg/LPIPS loss module; nested modules are included in their parents
  2. per-module memory         torch.cuda.memory_allocated() after minus before each module's forward = what the
                               module leaves allocated (outputs + tensors saved for backward)
  3. step split                forward / backward / optimizer, CUDA events
  4. kernel categories         torch.profiler over --profile-steps steps: GPU kernel time grouped into convolution,
                               matmul/linear, attention, the custom Triton STN / composite kernels, grid_sample,
                               elementwise/other (forward + backward + optimizer together)

Backward is reported as a whole (module-level backward timing is not attempted: several LPWM modules return dicts,
which module backward hooks do not cover). Use --override key=value to change config entries (e.g. particle count:
n_kp_per_patch=2 -> 512 prior particles with patch_size 8 at 128x128), --tf32-matmul to allow TF32 for matmul
(a precision change, measured only when asked).

    cd ~/dlp-lpwm-optimization && python lpwm/benchmarks/stn/lpwm_profile.py --config cfg_lpwm_D.json --out prof_lpwm_D
"""
import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np


def parse_override(s):
    k, v = s.split("=", 1)
    try:
        v = json.loads(v)
    except json.JSONDecodeError:
        pass
    return k, v


def kernel_category(name):
    n = name.lower()
    if "_stn_" in n or "composite" in n or "stn_crop" in n or "stn_paste" in n:
        return "custom triton (STN / composite)"
    if "grid_sampler" in n or "grid_sample" in n:
        return "grid_sample (reference STN)"
    if "flash" in n or "fmha" in n or "attention" in n or "efficient_attention" in n:
        return "attention"
    if "conv" in n or "implicit_gemm" in n or "cudnn" in n or "fft" in n or "winograd" in n or "dgrad" in n or "wgrad" in n:
        return "convolution"
    if "gemm" in n or "matmul" in n or "cutlass" in n or "xmma" in n or "sm90" in n or "sm80" in n or "cublas" in n:
        return "matmul / linear"
    if "adam" in n or "multi_tensor" in n or "foreach" in n:
        return "optimizer"
    return "elementwise / reduction / other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--profile-steps", type=int, default=2)
    ap.add_argument("--depth", type=int, default=2, help="module name depth to time (1 = encoder_module, 2 = encoder_module.x)")
    ap.add_argument("--override", action="append", default=[], help="config key=value (JSON value), repeatable")
    ap.add_argument("--tf32-matmul", action="store_true", help="allow TF32 for matmul (precision change)")
    ap.add_argument("--out", default="prof_lpwm")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    sys.path.insert(0, repo)
    import torch
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_video_dataset
    from utils.loss_functions import LossLPIPS

    torch.backends.cudnn.benchmark = False      # as train_lpwm.py
    torch.backends.cudnn.deterministic = True
    if args.tf32_matmul:
        torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(args.config))
    for o in args.override:
        k, v = parse_override(o)
        cfg[k] = v
    seed = cfg.get("seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    lpwm_stn.set_backend(cfg.get("stn_backend") or "reference")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(p, "w"))
        model, _, _ = build(p, "cuda")
    model.train()
    T = cfg["timestep_horizon"]
    recon = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["adam_betas"], eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"])
    kw = dict(warmup=False, with_loss=True, return_alpha_masks=False, beta_kl=cfg["beta_kl"], beta_dyn=cfg["beta_dyn"],
              beta_rec=cfg["beta_rec"], kl_balance=cfg["kl_balance"], dynamic_discount=None,
              recon_loss_type=cfg["recon_loss_type"], recon_loss_func=recon, beta_dyn_rec=cfg["beta_dyn_rec"],
              beta_obj=cfg.get("beta_obj", 0.0), done_mask=None, x_goal=None)
    ds = get_video_dataset(cfg["ds"], cfg["root"], seq_len=T + 1, mode="train", image_size=cfg["image_size"])
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=cfg["batch_size"], num_workers=4, pin_memory=True,
                                         drop_last=True, generator=torch.Generator().manual_seed(seed))
    it = iter(loader)

    # ---- module hooks: forward CUDA events + allocated-memory delta
    timed = {name: m for name, m in model.named_modules() if name and name.count(".") < args.depth}
    timed["loss: LossLPIPS (vgg)"] = recon
    rec = {"active": False, "scope": False, "events": defaultdict(list), "mem": defaultdict(list),
           "stack": defaultdict(list), "rf": defaultdict(list)}

    def pre(name):
        def f(mod, inp):
            if rec["active"]:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                rec["stack"][name].append((e, torch.cuda.memory_allocated()))
            if rec["scope"]:   # profiler scope so forward kernels can be attributed to this module
                r = torch.profiler.record_function("mod::" + name)
                r.__enter__()
                rec["rf"][name].append(r)
        return f

    def post(name):
        def f(mod, inp, out):
            if rec["active"] and rec["stack"][name]:
                e0, m0 = rec["stack"][name].pop()
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()
                rec["events"][name].append((e0, e1))
                rec["mem"][name].append(torch.cuda.memory_allocated() - m0)
            if rec["scope"] and rec["rf"][name]:
                rec["rf"][name].pop().__exit__(None, None, None)
        return f

    for name, m in timed.items():
        m.register_forward_pre_hook(pre(name))
        m.register_forward_hook(post(name))

    def one_step(record, scope=False):
        x = next(it)[0].to("cuda", non_blocking=True)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        rec["active"] = record
        rec["scope"] = scope
        ev[0].record()
        out = model(x, **kw)
        loss = out["loss_dict"]["loss"]
        ev[1].record()
        rec["active"] = False
        rec["scope"] = False
        opt.zero_grad()
        loss.backward()
        ev[2].record()
        opt.step()
        ev[3].record()
        return ev, out

    for _ in range(args.warmup):
        one_step(False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    steps = []
    for _ in range(args.steps):
        mem0 = torch.cuda.memory_allocated()
        ev, out = one_step(True)
        torch.cuda.synchronize()
        steps.append({"forward_ms": ev[0].elapsed_time(ev[1]), "backward_ms": ev[1].elapsed_time(ev[2]),
                      "optimizer_ms": ev[2].elapsed_time(ev[3]), "step_ms": ev[0].elapsed_time(ev[3])})
        del out
    peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    med = lambda k: statistics.median(s[k] for s in steps)  # noqa: E731
    step_ms = med("step_ms")
    modules = {}
    for name in timed:
        evs = rec["events"].get(name, [])
        if not evs:
            continue
        total = sum(a.elapsed_time(b) for a, b in evs) / args.steps
        modules[name] = {"forward_ms_per_step": total, "calls_per_step": len(evs) / args.steps,
                         "pct_of_step": 100 * total / step_ms, "pct_of_forward": 100 * total / med("forward_ms"),
                         "retained_mb_per_step": sum(rec["mem"][name]) / args.steps / 1e6}

    # ---- kernel categories over a few profiled steps
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.profile_steps):
            one_step(False, scope=True)
        torch.cuda.synchronize()

    # forward kernels attributed to module scopes (kernels launched by any op inside the module's forward)
    def scope_kernels(ev, acc):
        for k in getattr(ev, "kernels", []) or []:
            acc[kernel_category(k.name)] += k.duration
        for ch in ev.cpu_children:
            scope_kernels(ch, acc)

    module_cats = defaultdict(lambda: defaultdict(float))
    for e in prof.events():
        if e.name.startswith("mod::"):
            scope_kernels(e, module_cats[e.name[5:]])
    module_cats = {m: {c: v / 1e3 / args.profile_steps for c, v in cs.items()} for m, cs in module_cats.items()}

    cats = defaultdict(float)
    kernels = defaultdict(float)
    for e in prof.events():
        dev = getattr(e, "device_type", None)
        if dev is not None and "CUDA" in str(dev):
            t = getattr(e, "device_time", None) or getattr(e, "cuda_time", 0.0) or 0.0
            cats[kernel_category(e.name)] += t
            kernels[e.name] += t
    tot = sum(cats.values()) or 1.0
    categories = {k: {"ms_per_step": v / 1e3 / args.profile_steps, "pct_of_gpu_time": 100 * v / tot}
                  for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}
    top_kernels = [{"name": k[:140], "ms_per_step": v / 1e3 / args.profile_steps, "pct": 100 * v / tot}
                   for k, v in sorted(kernels.items(), key=lambda kv: -kv[1])[:25]]
    prof.export_chrome_trace(os.path.join(args.out, "trace.json"))

    result = {"config": os.path.abspath(args.config), "overrides": args.override, "tf32_matmul": args.tf32_matmul,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "stn_backend": lpwm_stn.get_backend_name(),
              "fused_composite": bool(cfg.get("fused_composite", False)),
              "particle_dec_channels_last": bool(cfg.get("particle_dec_channels_last", False)),
              "batch_size": cfg["batch_size"], "timestep_horizon": T, "n_kp_enc_model": model.n_kp_enc,
              "n_kp_dec_model": model.n_kp_dec, "n_kp_prior_model": model.n_kp_prior,
              "step_split_median_ms": {k: med(k) for k in ("forward_ms", "backward_ms", "optimizer_ms", "step_ms")},
              "peak_allocated_mb": peak_alloc, "modules": modules, "kernel_categories": categories, "top_kernels": top_kernels,
              "forward_kernel_categories_by_module": module_cats,
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- DLP vs LPWM-specific split (forward). DLP = encoder_module without its context encoder + decoder_module.
    def mget(name, key):
        return modules.get(name, {}).get(key, 0.0)

    lpwm_specific = [n for n in ("encoder_module.ctx_enc", "dyn_module", "ctx_module") if n in modules]
    split = {}
    for key in ("forward_ms_per_step", "retained_mb_per_step"):
        dlp = mget("encoder_module", key) - mget("encoder_module.ctx_enc", key) + mget("decoder_module", key)
        split[key] = {"DLP (encoder w/o ctx_enc + decoder)": dlp,
                      "LPWM-specific (" + " + ".join(lpwm_specific) + ")": sum(mget(n, key) for n in lpwm_specific),
                      "loss (vgg/LPIPS)": mget("loss: LossLPIPS (vgg)", key)}
    dlp_cats = defaultdict(float)
    for name, sign in (("encoder_module", 1), ("encoder_module.ctx_enc", -1), ("decoder_module", 1)):
        for c, v in module_cats.get(name, {}).items():
            dlp_cats[c] += sign * v
    result["dlp_vs_lpwm_split"] = split
    result["dlp_forward_kernel_categories"] = dict(dlp_cats)
    json.dump(result, open(os.path.join(args.out, "lpwm_profile.json"), "w"), indent=1)

    s = result["step_split_median_ms"]
    print(f"== {result['gpu']} | backend={result['stn_backend']} fused={result['fused_composite']} cl={result['particle_dec_channels_last']} "
          f"tf32_matmul={args.tf32_matmul} | batch {cfg['batch_size']} x {T + 1} frames | particles: prior {model.n_kp_prior}, "
          f"encoder {model.n_kp_enc}, decoder {model.n_kp_dec} | overrides {args.override}")
    print(f"step {s['step_ms']:.1f} ms = forward {s['forward_ms']:.1f} + backward {s['backward_ms']:.1f} + optimizer "
          f"{s['optimizer_ms']:.1f} | peak allocated {peak_alloc / 1e3:.1f} GB")
    print("\n-- forward time per module (nested modules are included in their parents)")
    print(f"{'module':52s} {'fwd ms':>8s} {'% fwd':>6s} {'% step':>7s} {'calls':>6s} {'kept MB':>9s}")
    for name, r in sorted(modules.items(), key=lambda kv: -kv[1]["forward_ms_per_step"]):
        if r["pct_of_forward"] < 0.5:
            continue
        print(f"{name[:52]:52s} {r['forward_ms_per_step']:8.1f} {r['pct_of_forward']:6.1f} {r['pct_of_step']:7.1f} "
              f"{r['calls_per_step']:6.1f} {r['retained_mb_per_step']:9.0f}")
    print("\n-- forward split: DLP vs LPWM-specific")
    fs, ms = split["forward_ms_per_step"], split["retained_mb_per_step"]
    for k in fs:
        print(f"{k:60s} {fs[k]:8.1f} ms  {100 * fs[k] / s['forward_ms']:5.1f}% of fwd  kept {ms[k] / 1e3:6.1f} GB")
    short = {"convolution": "conv", "matmul / linear": "matmul", "elementwise / reduction / other": "elemwise",
             "custom triton (STN / composite)": "triton", "grid_sample (reference STN)": "grid_smp", "attention": "attn",
             "optimizer": "optim"}
    order = list(short)
    print("\n-- forward GPU kernel ms by category, per module (scopes nest: parents include children)")
    print(f"{'module':52s} " + " ".join(f"{short[c]:>8s}" for c in order))
    rows = [("DLP total (encoder w/o ctx_enc + decoder)", dict(dlp_cats))] + \
           sorted(module_cats.items(), key=lambda kv: -sum(kv[1].values()))
    for name, cs in rows:
        if sum(cs.values()) < 0.5:
            continue
        print(f"{name[:52]:52s} " + " ".join(f"{cs.get(c, 0.0):8.1f}" for c in order))
    print("\n-- GPU kernel time by category (forward + backward + optimizer)")
    for k, v in categories.items():
        print(f"{k:36s} {v['ms_per_step']:8.1f} ms  {v['pct_of_gpu_time']:5.1f}%")
    print("\n-- top kernels")
    for k in top_kernels[:12]:
        print(f"{k['ms_per_step']:8.2f} ms {k['pct']:5.1f}%  {k['name'][:110]}")
    print(f"\nwrote {args.out}/lpwm_profile.json and {args.out}/trace.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
