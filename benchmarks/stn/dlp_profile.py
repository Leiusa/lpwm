#!/usr/bin/env python
"""
Where does a single-image DLP (train_dlp.py) training step spend its time? Measurement only; nothing in the
repository changes. One CUDA GPU, nothing else running on it.

Same model build, image DataLoader (num_workers=4, pin_memory), loss, Adam and forward arguments as train_dlp.py
(warmup=False, return_alpha_masks=False). --mode eager | default | reduce-overhead (torch.compile of the training
forward with the Triton boundary, utils/compile_utils.py, exactly as train_dlp.py). Reports:

  1. loop variants, wall clock per step (time.perf_counter over --steps steps, synchronized at the end):
       gpu_event        CUDA-event time from forward start to optimizer end = train_dlp.py's [step-timing] median
       loop_as_train    batch from the DataLoader + step + the 16 per-step .item() reads train_dlp.py does for its
                        epoch averages and progress bar (each one waits for the GPU) = what a real epoch costs per step
       loop_no_sync     batch from the DataLoader + step; the same 16 scalars kept on the GPU (detached clones),
                        read once at the end -> what removing the per-step reads would give
       loop_no_data     loop_no_sync with batches already on the GPU -> what the DataLoader costs on top
     plus the time spent waiting for the next batch in loop_as_train.
  2. GPU busy: the union of all GPU kernel / memcpy intervals over --profile-steps steps (torch.profiler), divided
     by that window's wall time. Low busy = the GPU waits for the CPU (launch overhead, syncs, data).
  3. GPU kernel time by category + top kernels. Inductor-generated kernels (triton_poi/red/per_*) are their own
     category; the custom STN / composite kernels are "custom triton". With reduce-overhead the kernels run inside
     CUDA graph replays; whether the profiler lists them depends on the driver (the kernel count is printed).
  4. eager only: per-module forward time (CUDA events, hooks down to --depth) and backward GPU kernel time per
     module (autograd sequence numbers, as lpwm_profile.py).

Config switches (stn_backend, fused_composite, cudnn_benchmark, cudnn_deterministic, tf32_matmul, n_kp_enc) come
from the config and --override key=value; '--override n_kp_enc="all"' = no particle filtering.

    cd ~/dlp-lpwm-optimization && python lpwm/benchmarks/stn/dlp_profile.py --config static_bair128_vgg_lambda.json \
        --mode reduce-overhead --override stn_backend='"triton"' --override fused_composite=true \
        --override cudnn_benchmark=true --out results/dlp_prof/ro_cb
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

# train_dlp.py reads these 16 scalars with .item() every step (9 epoch-average lists + 7 progress-bar values)
LOGGED = ("psnr", "loss", "loss_rec", "kl", "loss_kl_kp", "loss_kl_feat", "loss_kl_scale", "loss_kl_depth",
          "loss_kl_obj_on", "loss@bar", "loss_rec@bar", "kl@bar", "obj_on_l1", "a_mean", "b_mean", "mu_scale_mean")


def kernel_category(name):
    n = name.lower()
    if n.startswith("_stn_") or n.startswith("_composite"):
        return "custom triton (STN / composite)"
    if n.startswith("triton_"):   # inductor: triton_poi_/red_/per_ fused elementwise + reductions, triton_tem_ templates
        return "inductor fused (elementwise / reduction)"
    if "memcpy" in n or "memset" in n:
        return "memcpy / memset"
    if "grid_sampler" in n or "grid_sample" in n:
        return "grid_sample (reference STN)"
    if "flash" in n or "fmha" in n or "attention" in n:
        return "attention"
    if "conv" in n or "implicit_gemm" in n or "cudnn" in n or "fft" in n or "winograd" in n or "dgrad" in n or "wgrad" in n:
        return "convolution"
    if "gemm" in n or "matmul" in n or "cutlass" in n or "xmma" in n or "sm90" in n or "sm80" in n or "cublas" in n:
        return "matmul / linear"
    if "adam" in n or "multi_tensor" in n or "foreach" in n:
        return "optimizer"
    return "eager elementwise / reduction / other"


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
    ap.add_argument("--mode", default="eager", choices=("eager", "default", "reduce-overhead"))
    ap.add_argument("--warmup", type=int, default=10, help="untimed steps first (compilation, cuDNN benchmark trials)")
    ap.add_argument("--steps", type=int, default=100, help="steps per loop variant")
    ap.add_argument("--profile-steps", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3, help="eager: module name depth to time (1 = encoder_module)")
    ap.add_argument("--override", action="append", default=[], help="config key=value (JSON value), repeatable")
    ap.add_argument("--out", default="dlp_prof")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    sys.path.insert(0, repo)
    import torch
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from utils.loss_functions import LossLPIPS, calc_reconstruction_loss
    from lpwm_profile import backward_by_module, is_annotation

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(args.config))
    for o in args.override:
        k, v = parse_override(o)
        cfg[k] = v
    if cfg.get("n_kp_enc") == "all":   # same rule as models.py DLP.__init__
        total = cfg.get("n_kp_per_patch", 1) * (cfg["image_size"] // cfg["patch_size"]) ** 2
        cfg["n_kp_enc"] = min(total, cfg.get("n_kp_prior", total))
    # the switches train_dlp.py sets from the config
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(cfg.get("cudnn_deterministic", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("tf32_matmul", False))
    seed = cfg.get("seed", 0) or 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    lpwm_stn.set_backend(cfg.get("stn_backend") or "reference")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(p, "w"))
        model, _, _ = build(p, "cuda")
    model.train()
    rf = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).cuda() if cfg["recon_loss_type"] == "vgg" else calc_reconstruction_loss
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["adam_betas"], eps=cfg["adam_eps"],
                           weight_decay=cfg["weight_decay"])
    kw = dict(warmup=False, with_loss=True, return_alpha_masks=False, beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"],
              kl_balance=cfg["kl_balance"], recon_loss_type=cfg["recon_loss_type"], recon_loss_func=rf,
              beta_obj=cfg.get("beta_obj", 0.0))
    ds = get_image_dataset(cfg["ds"], cfg["root"], mode="train", image_size=cfg["image_size"])
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=cfg["batch_size"], num_workers=4, pin_memory=True,
                                         drop_last=True, generator=torch.Generator().manual_seed(seed))
    state = {"it": iter(loader)}

    def next_batch():
        try:
            b = next(state["it"])
        except StopIteration:
            state["it"] = iter(loader)
            b = next(state["it"])
        x = b[0].to("cuda", non_blocking=True)
        return x.unsqueeze(1) if x.dim() == 4 else x

    # ---- eager: per-module forward events + profiler scopes (hooks would break a compiled graph)
    timed = {n: m for n, m in model.named_modules() if n and n.count(".") < args.depth}
    rec = {"active": False, "scope": False, "events": defaultdict(list), "stack": defaultdict(list), "rf": defaultdict(list)}

    def pre(name):
        def f(mod, inp):
            if rec["active"]:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                rec["stack"][name].append(e)
            if rec["scope"]:
                r = torch.profiler.record_function("mod::" + name)
                r.__enter__()
                rec["rf"][name].append(r)
        return f

    def post(name):
        def f(mod, inp, out):
            if rec["active"] and rec["stack"][name]:
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()
                rec["events"][name].append((rec["stack"][name].pop(), e1))
            if rec["scope"] and rec["rf"][name]:
                rec["rf"][name].pop().__exit__(None, None, None)
        return f

    if args.mode == "eager":
        step_model = model
        for name, m in timed.items():
            m.register_forward_pre_hook(pre(name))
            m.register_forward_hook(post(name))
    else:
        from utils.compile_utils import compile_for_training
        step_model = compile_for_training(model, args.mode)

    def step(x, events=None):
        if args.mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        if events:
            events[0].record()
        out = step_model(x, **kw)
        loss = out["loss_dict"]["loss"]
        if events:
            events[1].record()
        opt.zero_grad()
        loss.backward()
        if events:
            events[2].record()
        opt.step()
        if events:
            events[3].record()
        return out

    def logged_scalars(out):
        d = out["loss_dict"]
        return [d["psnr"], d["loss"], d["loss_rec"], d["kl"], d["loss_kl_kp"], d["loss_kl_feat"], d["loss_kl_scale"],
                d["loss_kl_depth"], d["loss_kl_obj_on"], d["loss"], d["loss_rec"], d["kl"], d["obj_on_l1"],
                out["obj_on_a"].mean(), out["obj_on_b"].mean(), torch.sigmoid(out["mu_scale"]).mean()]

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        step(next_batch())
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0
    fused_calls0 = model.decoder_module.fused_composite_calls
    step(next_batch())
    torch.cuda.synchronize()
    path = {"stn_backend": lpwm_stn.get_backend_name(),
            "fused_composite_calls_per_step": model.decoder_module.fused_composite_calls - fused_calls0,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "mode": args.mode}
    if path["fused_composite_calls_per_step"] != int(bool(cfg.get("fused_composite", False))):
        raise RuntimeError(f"fused path not as requested: {path}")

    # ---- 1. loop variants
    torch.cuda.reset_peak_memory_stats()
    loops = {}
    ev_ms, split = [], defaultdict(list)
    for _ in range(args.steps):   # gpu_event (train_dlp.py's [step-timing])
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        rec["active"] = args.mode == "eager"
        step(next_batch(), ev)
        rec["active"] = False
        torch.cuda.synchronize()
        ev_ms.append(ev[0].elapsed_time(ev[3]))
        split["forward_ms"].append(ev[0].elapsed_time(ev[1]))
        split["backward_ms"].append(ev[1].elapsed_time(ev[2]))
        split["optimizer_ms"].append(ev[2].elapsed_time(ev[3]))
    loops["gpu_event"] = statistics.median(ev_ms)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    data_wait = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):   # loop_as_train
        td = time.perf_counter()
        x = next_batch()
        data_wait += time.perf_counter() - td
        out = step(x)
        _ = [s.data.cpu().item() for s in logged_scalars(out)]
    torch.cuda.synchronize()
    loops["loop_as_train"] = 1e3 * (time.perf_counter() - t0) / args.steps
    loops["data_wait_in_loop_as_train"] = 1e3 * data_wait / args.steps

    kept = []
    t0 = time.perf_counter()
    for _ in range(args.steps):   # loop_no_sync
        out = step(next_batch())
        kept.append(torch.stack([s.detach().float().reshape(()) for s in logged_scalars(out)]).clone())
    _ = torch.stack(kept).cpu()
    torch.cuda.synchronize()
    loops["loop_no_sync"] = 1e3 * (time.perf_counter() - t0) / args.steps

    xs = [next_batch() for _ in range(8)]
    kept = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(args.steps):   # loop_no_data
        out = step(xs[i % len(xs)])
        kept.append(torch.stack([s.detach().float().reshape(()) for s in logged_scalars(out)]).clone())
    _ = torch.stack(kept).cpu()
    torch.cuda.synchronize()
    loops["loop_no_data"] = 1e3 * (time.perf_counter() - t0) / args.steps
    del xs, kept, out

    # ---- 2./3. profiler: GPU busy, kernel categories (eager: + module scopes for backward attribution)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("window"):
            for _ in range(args.profile_steps):
                x = next_batch()
                rec["scope"] = args.mode == "eager"
                out = step(x)
                rec["scope"] = False
                _ = [s.data.cpu().item() for s in logged_scalars(out)]   # as train_dlp.py
            torch.cuda.synchronize()
    events = prof.events()
    dev = [e for e in events if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type)
           and not is_annotation(e.name) and e.name != "window"]
    win = [e for e in events if e.name == "window" and not ("CUDA" in str(getattr(e, "device_type", "")))]
    w0 = win[0].time_range.start if win else min(e.time_range.start for e in dev)
    w1 = max(([win[0].time_range.end] if win else []) + [e.time_range.end for e in dev])
    iv = sorted((e.time_range.start, e.time_range.end) for e in dev)
    busy, cur0, cur1 = 0.0, None, None
    for a, b in iv:
        if cur1 is None or a > cur1:
            if cur1 is not None:
                busy += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None:
        busy += cur1 - cur0
    gpu = {"window_ms_per_step": (w1 - w0) / 1e3 / args.profile_steps, "busy_ms_per_step": busy / 1e3 / args.profile_steps,
           "busy_pct": 100 * busy / max(w1 - w0, 1e-9), "kernels_per_step": len(dev) / args.profile_steps}
    cats, kernels = defaultdict(float), defaultdict(float)
    for e in dev:
        t = e.time_range.end - e.time_range.start
        cats[kernel_category(e.name)] += t
        kernels[e.name] += t
    tot = sum(cats.values()) or 1.0
    categories = {k: {"ms_per_step": v / 1e3 / args.profile_steps, "pct": 100 * v / tot}
                  for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}
    top = [{"name": k[:150], "ms_per_step": v / 1e3 / args.profile_steps, "pct": 100 * v / tot}
           for k, v in sorted(kernels.items(), key=lambda kv: -kv[1])[:25]]

    # ---- 4. eager: per-module forward / backward
    modules, bwd_cats, bwd_cover = {}, {}, {}
    if args.mode == "eager":
        fwd_med = statistics.median(split["forward_ms"])
        for name in timed:
            evs = rec["events"].get(name, [])
            if evs:
                t = sum(a.elapsed_time(b) for a, b in evs) / args.steps
                modules[name] = {"forward_ms": t, "pct_of_forward": 100 * t / fwd_med, "calls": len(evs) / args.steps}
        import lpwm_profile
        lpwm_profile.kernel_category = kernel_category   # same categories in the backward attribution
        bwd_cats, bwd_cover = backward_by_module(prof, args.profile_steps)

    result = {"config": os.path.abspath(args.config), "overrides": args.override, "path": path,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch_size": cfg["batch_size"],
              "particles": {"prior": model.n_kp_prior, "encoder": model.n_kp_enc, "decoder": model.n_kp_dec},
              "warmup_s": warmup_s, "loops_ms_per_step": loops,
              "step_split_median_ms": {k: statistics.median(v) for k, v in split.items()}, "peak_allocated_gb": peak_gb,
              "gpu_busy": gpu, "kernel_categories": categories, "top_kernels": top, "modules_forward": modules,
              "backward_kernel_categories_by_module": bwd_cats, "backward_attribution_coverage": bwd_cover,
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.mode != "eager":
        from torch._dynamo.utils import counters
        result["dynamo"] = {"unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
                            "graph_breaks": int(sum(counters["graph_break"].values())),
                            "graph_break_reasons": {k[:300]: int(v) for k, v in counters["graph_break"].most_common(10)}}
    json.dump(result, open(os.path.join(args.out, "dlp_profile.json"), "w"), indent=1)
    prof.export_chrome_trace(os.path.join(args.out, "trace.json"))
    report(result)
    return 0


def report(r):
    p, L, s, g = r["path"], r["loops_ms_per_step"], r["step_split_median_ms"], r["gpu_busy"]
    print(f"== {r['gpu']} | mode={p['mode']} backend={p['stn_backend']} fused_calls={p['fused_composite_calls_per_step']} "
          f"cudnn_benchmark={p['cudnn_benchmark']} deterministic={p['cudnn_deterministic']} tf32_matmul={p['tf32_matmul']} | "
          f"batch {r['batch_size']} | particles prior {r['particles']['prior']}, encoder {r['particles']['encoder']}, "
          f"decoder {r['particles']['decoder']} | warmup {r['warmup_s']:.0f} s")
    print(f"\n-- 1. per step (ms)   gpu_event {L['gpu_event']:.1f} (fwd {s['forward_ms']:.1f} + bwd {s['backward_ms']:.1f} + "
          f"opt {s['optimizer_ms']:.1f}) | peak allocated {r['peak_allocated_gb']:.1f} GB")
    print(f"   loop_as_train {L['loop_as_train']:.1f}  (of which waiting for the next batch {L['data_wait_in_loop_as_train']:.1f})")
    print(f"   loop_no_sync  {L['loop_no_sync']:.1f}  ({100 * (L['loop_no_sync'] - L['loop_as_train']) / L['loop_as_train']:+.1f}% vs loop_as_train)")
    print(f"   loop_no_data  {L['loop_no_data']:.1f}  ({100 * (L['loop_no_data'] - L['loop_as_train']) / L['loop_as_train']:+.1f}% vs loop_as_train)")
    print(f"\n-- 2. GPU busy (profiled steps, as train_dlp.py incl. per-step reads): {g['busy_ms_per_step']:.1f} of "
          f"{g['window_ms_per_step']:.1f} ms per step = {g['busy_pct']:.0f}% | {g['kernels_per_step']:.0f} GPU kernels/copies per step")
    print("\n-- 3. GPU time by category")
    for k, v in r["kernel_categories"].items():
        print(f"   {k:42s} {v['ms_per_step']:7.1f} ms {v['pct']:5.1f}%")
    print("   top kernels")
    for k in r["top_kernels"][:12]:
        print(f"   {k['ms_per_step']:7.2f} ms {k['pct']:5.1f}%  {k['name'][:110]}")
    if "dynamo" in r:
        d = r["dynamo"]
        print(f"\n   dynamo: {d['unique_graphs']} graphs, {d['graph_breaks']} graph breaks")
        for k, v in list(d["graph_break_reasons"].items())[:5]:
            print(f"   {v:4d} x {k[:140]}")
    if r["modules_forward"]:
        bwd = r["backward_kernel_categories_by_module"]
        c = r["backward_attribution_coverage"]
        print(f"\n-- 4. per module (eager; nested modules included in parents). fwd = CUDA events, bwd = GPU kernel time "
              f"attributed via autograd ({c.get('attributed_to_a_module_ms', 0):.1f} of "
              f"{c.get('backward_kernel_ms_per_step', 0):.1f} ms attributed)")
        short = {"convolution": "conv", "matmul / linear": "matmul", "eager elementwise / reduction / other": "elemwise",
                 "custom triton (STN / composite)": "triton", "grid_sample (reference STN)": "grid_smp", "attention": "attn"}
        print(f"   {'module':52s} {'fwd ms':>7s} {'% fwd':>6s} {'bwd ms':>7s} " + " ".join(f"{v:>8s}" for v in short.values()))
        names = sorted(set(r["modules_forward"]) | set(bwd), key=lambda n: -(r["modules_forward"].get(n, {}).get("forward_ms", 0)
                                                                            + sum(bwd.get(n, {}).values())))
        for n in names:
            f = r["modules_forward"].get(n, {})
            b = bwd.get(n, {})
            if f.get("forward_ms", 0) + sum(b.values()) < 0.5:
                continue
            print(f"   {n[:52]:52s} {f.get('forward_ms', 0):7.1f} {f.get('pct_of_forward', 0):6.1f} {sum(b.values()):7.1f} " +
                  " ".join(f"{b.get(k, 0.0):8.1f}" for k in short))


if __name__ == "__main__":
    sys.exit(main())
