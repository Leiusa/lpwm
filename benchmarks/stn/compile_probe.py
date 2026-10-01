#!/usr/bin/env python
"""
torch.compile probe for static DLP: wrap the model in torch.compile() and see what happens.

For ONE path (reference = A, fused_cl = D), in one process on one verified GPU, through the same training step
full_stack_scan.py uses (and train_dlp.py does): real BAIR batches, the vgg loss (LossLPIPS), Adam with the config's
settings, warmup=False, return_alpha_masks=False. Phases, in order, all on the same model instance:

  1. eager_before : the model as is -- the baseline, same process and GPU
  2. explain      : torch._dynamo.explain(model)(x, **kw) on one batch -- graphs captured, graph breaks, their reasons
  3. compiled     : cmodel = torch.compile(model), nothing else changed. Warm-up runs until no new graph has been
                    compiled for --settle-steps steps (at most --max-compiled-warmup-steps), then --timed-steps are
                    timed. Every step's time and the dynamo graph count after it are kept, so the first compilation
                    and any recompilation are visible.
  4. eager_after  : the original module again -- drift check of the eager baseline
  5. numerics     : eager vs compiled deterministic inference (model.eval(), no_grad, deterministic=True), one batch

Nothing in the repository changes. Errors are not suppressed (torch._dynamo.config.suppress_errors stays False):
a failure is recorded as what happened and the remaining phases still run where they can. The dynamo logs
(TORCH_LOGS=graph_breaks,recompiles, set by the job script) go to this process's stderr.

    python compile_probe.py --repo-root R --config CFG --path reference|fused_cl --out probe.json
"""
import argparse
import importlib.metadata as md
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
import traceback

import numpy as np


def _version(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", choices=("reference", "fused_cl"), required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eager-warmup-steps", type=int, default=10)
    ap.add_argument("--timed-steps", type=int, default=30)
    ap.add_argument("--settle-steps", type=int, default=5, help="compiled warm-up ends after this many steps without a new graph")
    ap.add_argument("--max-compiled-warmup-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disable-custom-ops", action="store_true",
                    help="keep the custom Triton ops out of the compiled graph: wrap lpwm_stn.triton_backend.stn_crop/stn_paste "
                         "and modules.modules.composite_fused with torch.compiler.disable (after the eager baseline, before "
                         "explain). No repository file changes; the reference backend is not touched.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    import bench_e2e as be
    be._bootstrap(repo)
    import torch
    import torch._dynamo
    from torch._dynamo.utils import counters
    import lpwm_stn
    import m1_common as MC
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from utils.loss_functions import LossLPIPS

    hw = MC.verify_hardware()
    MC.assert_agreed_environment(MC.set_agreed_environment())
    B, seed, path = args.batch_size, args.seed, args.path
    backend = "triton" if path == "fused_cl" else "reference"
    fused = cl = path == "fused_cl"
    cfg = json.load(open(args.config))
    cfg.update(batch_size=B, stn_backend=backend, fused_composite=fused, particle_dec_channels_last=cl, seed=seed)
    result = {"kind": "torch.compile probe (default torch.compile(model)); short measurement, not a convergence run",
              "path": path, "batch_size": B, "hardware": hw, "settings": MC.effective_settings(), "code_hashes": MC.code_hashes(repo),
              "script_sha256": be.file_sha256(os.path.abspath(__file__)), "config_sha256": be.file_sha256(args.config),
              "versions": {"torch": torch.__version__, "triton": _version("triton"),
                           "torch_requires_triton": [r for r in (md.requires("torch") or []) if r.lower().startswith("triton")],
                           "dynamo_cache_size_limit": torch._dynamo.config.cache_size_limit,
                           "dynamo_suppress_errors": torch._dynamo.config.suppress_errors,
                           "inline_inbuilt_nn_modules": getattr(torch._dynamo.config, "inline_inbuilt_nn_modules", None)},
              "env": {k: os.environ.get(k) for k in ("TORCH_LOGS", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "SLURM_JOB_ID")},
              "compile_call": "cmodel = torch.compile(model)   # default mode, fullgraph=False, dynamic=None",
              "phases": {}, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def save():
        json.dump(result, open(args.out, "w"), indent=2, default=str)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lpwm_stn.set_backend(backend)
    dataset = get_image_dataset(cfg["ds"], cfg["root"], mode="train", image_size=cfg["image_size"])

    def batch_stream():
        g = torch.Generator().manual_seed(seed)
        while True:
            loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=B, num_workers=4, pin_memory=True,
                                                 drop_last=True, generator=g)
            for b in loader:
                yield b[0]
    batches = batch_stream()

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(cfg_path, "w"))
        model, _, _ = build(cfg_path, "cuda")
    model.train()
    recon_loss_func = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["adam_betas"], eps=cfg["adam_eps"],
                           weight_decay=cfg["weight_decay"])
    kw = dict(warmup=False, with_loss=True, beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"], kl_balance=cfg["kl_balance"],
              recon_loss_type=cfg["recon_loss_type"], recon_loss_func=recon_loss_func, beta_obj=cfg.get("beta_obj", 0.0),
              return_alpha_masks=False)

    def step(fn, x):
        out = fn(x, **kw)                      # model_output stays alive until the step ends, as in train_dlp.py
        loss = out["loss_dict"]["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss)

    def model_counters():
        dec = model.decoder_module
        return {"fused_composite_calls": dec.fused_composite_calls,
                "channels_last_calls": getattr(dec.particle_dec, "channels_last_calls", 0)}

    def dyn():
        return {k: {str(kk): vv for kk, vv in v.items()} for k, v in counters.items()}

    def graphs():
        return counters["stats"].get("unique_graphs", 0)

    def run_phase(name, fn, n_warm, n_timed, settle=None, max_warm=None):
        """settle/max_warm given: warm up until `settle` consecutive steps compiled no new graph (at most max_warm)."""
        rec = {"status": "started"}
        result["phases"][name] = rec
        counters.clear()
        mc0 = model_counters()
        times, graphs_after, losses = [], [], []
        warm_done, settled = None, None
        try:
            quiet = 0
            while True:
                i = len(times)
                if warm_done is None:
                    if settle is None:
                        if i == n_warm:
                            warm_done = i
                    elif quiet >= settle or i >= max_warm:
                        warm_done, settled = i, quiet >= settle
                    if warm_done is not None:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                if warm_done is not None and i >= warm_done + n_timed:
                    break
                x = next(batches).to("cuda")
                g0 = graphs()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                losses.append(step(fn, x))
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
                graphs_after.append(graphs())
                quiet = quiet + 1 if graphs() == g0 else 0
            timed = times[warm_done:]
            rec.update(status="ok", step_ms_median=statistics.median(timed), step_ms_mean=statistics.fmean(timed),
                       step_ms_min=min(timed), step_ms_max=max(timed),
                       peak_allocated_mb=torch.cuda.max_memory_allocated() / 1e6,
                       peak_reserved_mb=torch.cuda.max_memory_reserved() / 1e6,
                       new_graphs_during_timed_steps=graphs_after[-1] - graphs_after[warm_done - 1] if warm_done else graphs_after[-1],
                       compiled_warmup_settled=settled)
        except Exception as exc:  # noqa: BLE001 - what happened is the result
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:3000], traceback=traceback.format_exc()[-8000:])
        mc1 = model_counters()
        rec.update(warmup_steps=warm_done, steps_completed=len(times), step_ms_all=[round(t, 2) for t in times],
                   graphs_after_each_step=graphs_after, loss_first=losses[0] if losses else None,
                   loss_last=losses[-1] if losses else None, all_losses_finite=all(math.isfinite(v) for v in losses),
                   model_counter_delta={k: mc1[k] - mc0[k] for k in mc0}, dynamo_counters=dyn(),
                   backend_after=lpwm_stn.get_backend_name())
        save()
        print(f"[{name}] {rec['status']} | steps {len(times)} (warm-up {warm_done}) | median "
              f"{rec.get('step_ms_median', float('nan')):.1f} ms | first step {times[0] if times else float('nan'):.0f} ms | "
              f"graphs {graphs()} | counters {rec['model_counter_delta']}", flush=True)
        return rec

    # 1. eager baseline
    run_phase("eager_before", model, args.eager_warmup_steps, args.timed_steps)

    if args.disable_custom_ops:
        import lpwm_stn.triton_backend as tb
        import modules.modules as mm
        wrapped = []
        for mod, name in ((tb, "stn_crop"), (tb, "stn_paste"), (mm, "composite_fused")):
            if getattr(mod, name, None) is not None:
                setattr(mod, name, torch.compiler.disable(getattr(mod, name)))
                wrapped.append(f"{mod.__name__}.{name}")
        result["custom_ops_disabled"] = wrapped
        save()
        print(f"[disable-custom-ops] {wrapped}", flush=True)

    # 2. graph-break report
    ex_rec = {"status": "started"}
    result["phases"]["explain"] = ex_rec
    try:
        torch._dynamo.reset()
        counters.clear()
        x = next(batches).to("cuda")
        t0 = time.perf_counter()
        ex = torch._dynamo.explain(model)(x, **kw)

        def where(frames):
            return [f"{os.path.relpath(f.filename, repo) if f.filename.startswith(repo) else f.filename}:{f.lineno} in {f.name}"
                    for f in (frames or [])][-6:]
        ex_rec.update(status="ok", seconds=time.perf_counter() - t0, graph_count=ex.graph_count,
                      graph_break_count=ex.graph_break_count, op_count=ex.op_count,
                      ops_per_graph=[len(o) for o in (ex.ops_per_graph or [])],
                      break_reasons=[{"reason": str(r.reason)[:1500], "user_stack": where(r.user_stack)} for r in ex.break_reasons])
        with open(os.path.splitext(args.out)[0] + "_explain.txt", "w") as fh:
            fh.write(str(ex))
        del ex
    except Exception as exc:  # noqa: BLE001
        ex_rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:3000], traceback=traceback.format_exc()[-8000:])
    finally:
        opt.zero_grad(set_to_none=True)
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    save()
    print(f"[explain] {ex_rec['status']} | graphs {ex_rec.get('graph_count')} | graph breaks {ex_rec.get('graph_break_count')}", flush=True)

    # 3. the probe itself
    cmodel = torch.compile(model)
    run_phase("compiled", cmodel, None, args.timed_steps, settle=args.settle_steps, max_warm=args.max_compiled_warmup_steps)

    # 4. eager again (drift check)
    run_phase("eager_after", model, 3, 20)

    # 5. numerics: compiled vs eager, deterministic inference
    num = {"status": "started"}
    result["phases"]["numerics"] = num
    try:
        counters.clear()
        x = next(batches).to("cuda")
        model.eval()
        with torch.no_grad():
            ref = model(x, deterministic=True, with_loss=False)["rec_rgb"].float()
            got = cmodel(x, deterministic=True, with_loss=False)["rec_rgb"].float()
        d = got - ref
        mse = float((d ** 2).mean())
        num.update(status="ok", max_abs_diff=float(d.abs().max()), mean_abs_diff=float(d.abs().mean()), mse_between=mse,
                   psnr_between_db=(-10 * math.log10(mse) if mse > 0 else float("inf")), bitwise_equal=bool(torch.equal(got, ref)),
                   compiled_call_compiled_new_graph=graphs() > 0, dynamo_counters=dyn())
    except Exception as exc:  # noqa: BLE001
        num.update(status="error", error=f"{type(exc).__name__}: {exc}"[:3000], traceback=traceback.format_exc()[-8000:])
    finally:
        model.train()
    try:
        result["compile_times"] = torch._dynamo.utils.compile_times(repr="str")
    except Exception as exc:  # noqa: BLE001
        result["compile_times"] = f"unavailable: {exc}"
    result["status"] = "done"
    save()
    print(f"[numerics] {num['status']} | max |compiled - eager| {num.get('max_abs_diff')} | new graph compiled: "
          f"{num.get('compiled_call_compiled_new_graph')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
