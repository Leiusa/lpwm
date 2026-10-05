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

  5. per-module BACKWARD time  GPU kernel time of every autograd node evaluated in backward, attributed to the module
                               whose forward created that node (profiler sequence numbers link a backward node to the
                               forward op that recorded it; that op's mod:: scopes give the module). Kernel time, not
                               wall time; nested modules are included in their parents; nodes created outside every
                               timed module (top-level losses, AccumulateGrad) are listed as unattributed.

Use --override key=value to change config entries (e.g. particle count:
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


def is_annotation(name):
    # the mod:: record_function scopes are mirrored on the GPU timeline as user annotations spanning the whole module
    # (they show up among device events); they are not kernels
    return name.startswith("mod::")


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
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="cudnn.benchmark=True: time the candidate conv algorithms per shape, keep the fastest "
                         "(default False, as train_lpwm.py)")
    ap.add_argument("--cudnn-nondeterministic", action="store_true",
                    help="cudnn.deterministic=False: also allow conv algorithms whose results vary run to run at "
                         "rounding level (default True, as train_lpwm.py)")
    ap.add_argument("--compile", default=None, metavar="MODE",
                    help="profile the torch.compile'd training step (utils/compile_utils.compile_for_training, MODE = "
                         "default | reduce-overhead). Per-module timing is skipped (module boundaries are fused); step "
                         "split and kernel categories are reported. Use 'default' for the kernel breakdown: with "
                         "reduce-overhead the kernels run inside CUDA graph replays and may not be listed individually.")
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

    torch.backends.cudnn.benchmark = args.cudnn_benchmark                 # train_lpwm.py: False
    torch.backends.cudnn.deterministic = not args.cudnn_nondeterministic  # train_lpwm.py: True
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

    if args.compile:   # hooks inside a compiled graph would break it: no per-module timing in this mode
        from utils.compile_utils import compile_for_training
        step_model = compile_for_training(model, args.compile)
        args.warmup = max(args.warmup, 5)   # first step compiles; give the graph time to settle
    else:
        step_model = model
        for name, m in timed.items():
            m.register_forward_pre_hook(pre(name))
            m.register_forward_hook(post(name))

    def one_step(record, scope=False):
        x = next(it)[0].to("cuda", non_blocking=True)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        rec["active"] = record
        rec["scope"] = scope
        if args.compile == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        ev[0].record()
        out = step_model(x, **kw)
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
            if is_annotation(k.name):
                continue
            acc[kernel_category(k.name)] += k.duration
        for ch in ev.cpu_children:
            scope_kernels(ch, acc)

    module_cats = defaultdict(lambda: defaultdict(float))
    for e in prof.events():
        if e.name.startswith("mod::"):
            scope_kernels(e, module_cats[e.name[5:]])
    module_cats = {m: {c: v / 1e3 / args.profile_steps for c, v in cs.items()} for m, cs in module_cats.items()}
    bwd_cats, bwd_cover = ({}, {}) if args.compile else backward_by_module(prof, args.profile_steps)

    cats = defaultdict(float)
    kernels = defaultdict(float)
    for e in prof.events():
        dev = getattr(e, "device_type", None)
        if dev is not None and "CUDA" in str(dev) and not is_annotation(e.name):
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
              "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
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
    bwd = {n: sum(cs.values()) for n, cs in bwd_cats.items()}
    if bwd:
        split["backward_kernel_ms_per_step"] = {
            k: v for k, v in zip(split["forward_ms_per_step"],
                                 (bwd.get("encoder_module", 0.0) - bwd.get("encoder_module.ctx_enc", 0.0) + bwd.get("decoder_module", 0.0),
                                  sum(bwd.get(n, 0.0) for n in lpwm_specific), bwd.get("loss: LossLPIPS (vgg)", 0.0)))}
    result["dlp_vs_lpwm_split"] = split
    result["dlp_forward_kernel_categories"] = dict(dlp_cats)
    result["backward_kernel_categories_by_module"] = bwd_cats
    result["backward_attribution_coverage"] = bwd_cover
    if args.compile:
        from torch._dynamo.utils import counters
        result["dynamo"] = {"unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
                            "graph_breaks": int(sum(counters["graph_break"].values())),
                            "graph_break_reasons": {k[:300]: int(v) for k, v in counters["graph_break"].most_common(20)}}
    json.dump(result, open(os.path.join(args.out, "lpwm_profile.json"), "w"), indent=1)

    s = result["step_split_median_ms"]
    print(f"== {result['gpu']} | backend={result['stn_backend']} fused={result['fused_composite']} cl={result['particle_dec_channels_last']} "
          f"tf32_matmul={args.tf32_matmul} cudnn_benchmark={torch.backends.cudnn.benchmark} "
          f"cudnn_deterministic={torch.backends.cudnn.deterministic} compile={args.compile} | batch {cfg['batch_size']} x {T + 1} frames | particles: "
          f"prior {model.n_kp_prior}, encoder {model.n_kp_enc}, decoder {model.n_kp_dec} | overrides {args.override}")
    print(f"step {s['step_ms']:.1f} ms = forward {s['forward_ms']:.1f} + backward {s['backward_ms']:.1f} + optimizer "
          f"{s['optimizer_ms']:.1f} | peak allocated {peak_alloc / 1e3:.1f} GB")
    if args.compile:
        d = result["dynamo"]
        print(f"(compiled step: per-module timing skipped, module boundaries are fused) | dynamo: {d['unique_graphs']} "
              f"graphs, {d['graph_breaks']} graph breaks")
        for k, v in list(d["graph_break_reasons"].items())[:8]:
            print(f"   {v:4d} x {k[:150]}")
    else:
        print_module_tables(modules, split, s, dlp_cats, module_cats)
        print_backward_tables(modules, module_cats, bwd_cats, bwd_cover, split)
    print_kernel_tables(categories, top_kernels, args.out)
    return 0


def print_module_tables(modules, split, s, dlp_cats, module_cats):
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


BWD_ROOT = "autograd::engine::evaluate_function"


def backward_by_module(prof, n_steps):
    """Backward GPU kernel time per module (ms per step, by kernel category) + how much of it could be attributed."""
    events = prof.events()

    def under_backward(e):
        while e is not None:
            if e.name.startswith(BWD_ROOT):
                return True
            e = e.cpu_parent
        return False

    def scopes(e):
        out = []
        while e is not None:
            if e.name.startswith("mod::"):
                out.append(e.name[5:])
            e = e.cpu_parent
        return out

    # forward: sequence number -> module scopes of the op that created the autograd node. Ops that create no node
    # (no grad needed) peek the same number as the next node-creating op, possibly in another module; the creating op
    # (or one nested in it) is the LAST to start with that number, so the latest-starting op wins.
    fwd = {}
    for e in events:
        seq = getattr(e, "sequence_nr", -1)
        if seq is None or seq < 0 or e.name.startswith(BWD_ROOT) or under_backward(e):
            continue
        t = e.time_range.start
        if seq not in fwd or t >= fwd[seq][0]:
            fwd[seq] = (t, scopes(e))

    def node_seq(e):
        stack = [e]
        while stack:
            x = stack.pop(0)
            seq = getattr(x, "sequence_nr", -1)
            if seq is not None and seq >= 0:
                return seq
            stack.extend(x.cpu_children)
        return -1

    def subtree_kernels(e, acc):
        for k in getattr(e, "kernels", []) or []:
            if is_annotation(k.name):
                continue
            acc[kernel_category(k.name)] += k.duration
        for ch in e.cpu_children:
            subtree_kernels(ch, acc)

    by_mod = defaultdict(lambda: defaultdict(float))
    cover = {"backward_kernel_ms_per_step": 0.0, "attributed_to_a_module_ms": 0.0, "unattributed_ms": 0.0,
             "nodes": 0, "nodes_without_forward_match": 0}
    unattributed = defaultdict(float)
    for e in events:
        if not e.name.startswith(BWD_ROOT) or under_backward(e.cpu_parent):
            continue
        acc = defaultdict(float)
        subtree_kernels(e, acc)
        ms = sum(acc.values()) / 1e3 / n_steps
        cover["backward_kernel_ms_per_step"] += ms
        cover["nodes"] += 1
        seq = node_seq(e)
        mods = fwd.get(seq, (0, []))[1] if seq >= 0 else []
        if seq < 0 or seq not in fwd:
            cover["nodes_without_forward_match"] += 1
        if not mods:
            cover["unattributed_ms"] += ms
            unattributed[e.name[len(BWD_ROOT) + 2:].split(" ")[0][:60]] += ms
            continue
        cover["attributed_to_a_module_ms"] += ms
        for m in mods:
            for c, v in acc.items():
                by_mod[m][c] += v / 1e3 / n_steps
    cover["nodes"] //= n_steps
    cover["nodes_without_forward_match"] //= n_steps
    cover["unattributed_top_nodes_ms"] = dict(sorted(unattributed.items(), key=lambda kv: -kv[1])[:10])
    return {m: dict(cs) for m, cs in by_mod.items()}, cover


def print_backward_tables(modules, fwd_cats, bwd_cats, cover, split):
    if not bwd_cats:
        print("\n-- backward by module: nothing attributed (profiler events carry no sequence numbers?)")
        return
    print(f"\n-- backward GPU kernel time per module (attributed via autograd sequence numbers; nested modules are "
          f"included in their parents)")
    print(f"   coverage: {cover['backward_kernel_ms_per_step']:.1f} ms backward kernel time/step, "
          f"{cover['attributed_to_a_module_ms']:.1f} ms attributed to a module, {cover['unattributed_ms']:.1f} ms "
          f"unattributed; {cover['nodes']} nodes/step, {cover['nodes_without_forward_match']} without a forward match")
    for k, v in list(cover["unattributed_top_nodes_ms"].items())[:5]:
        print(f"     unattributed {v:7.1f} ms  {k}")
    short = {"convolution": "conv", "matmul / linear": "matmul", "elementwise / reduction / other": "elemwise",
             "custom triton (STN / composite)": "triton", "grid_sample (reference STN)": "grid_smp", "attention": "attn"}
    print(f"{'module':52s} {'fwd ms':>8s} {'bwd ms':>8s} {'bwd/fwd':>7s} " + " ".join(f"{v:>8s}" for v in short.values()))
    for name, cs in sorted(bwd_cats.items(), key=lambda kv: -sum(kv[1].values())):
        b = sum(cs.values())
        if b < 0.5:
            continue
        f = sum(fwd_cats.get(name, {}).values())
        print(f"{name[:52]:52s} {f:8.1f} {b:8.1f} {b / f if f else float('nan'):7.2f} " +
              " ".join(f"{cs.get(c, 0.0):8.1f}" for c in short))
    if "backward_kernel_ms_per_step" in split:
        print("\n-- backward split: DLP vs LPWM-specific (GPU kernel ms per step)")
        for k, v in split["backward_kernel_ms_per_step"].items():
            print(f"{k:60s} {v:8.1f} ms")


def print_kernel_tables(categories, top_kernels, out):
    print("\n-- GPU kernel time by category (forward + backward + optimizer)")
    for k, v in categories.items():
        print(f"{k:36s} {v['ms_per_step']:8.1f} ms  {v['pct_of_gpu_time']:5.1f}%")
    print("\n-- top kernels")
    for k in top_kernels[:12]:
        print(f"{k['ms_per_step']:8.2f} ms {k['pct']:5.1f}%  {k['name'][:110]}")
    print(f"\nwrote {out}/lpwm_profile.json and {out}/trace.json")


if __name__ == "__main__":
    sys.exit(main())
