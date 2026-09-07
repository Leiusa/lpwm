#!/usr/bin/env python
"""
End-to-end LPWM benchmark for one STN backend.

One backend per process: a backend that OOMs must not leave a fragmented
allocator behind and poison the other's numbers.

    python benchmarks/stn/bench_e2e.py --repo-root /workspace/lpwm \
      --config configs/bair.json --backend triton --batch-size 1 \
      --out /workspace/experiments/<stamp>/bair_bs1_triton.json

Metrics, deliberately named so they cannot be confused:

  D_clean_wall_ms                       median of the UNINSTRUMENTED timed loop.
                                        The only number end-to-end speedup may
                                        be quoted from, and the only valid input
                                        to an Amdahl argument.
  D_prof_wall_ms                        wall time of the profiled step. Slower,
                                        because instrumentation costs time.
  total_device_kernel_ms                summed self device time of CUDA kernels.
  stn_device_kernel_ms                  the STN part of that.
  stn_share_of_device_kernel_time_pct   ratio of the two above.

`stn_share_of_device_kernel_time_pct` is a share of *kernel time*, not of wall
time: kernels overlap and the two are not interchangeable. No wall-time share is
reported, because making one defensible needs concurrency analysis this harness
does not do.

Attribution lives in attribution.py and is unit-tested on CPU. Scopes are
installed by monkeypatching the bindings in `modules.modules`, only for the
duration of the profiled step -- no production file is edited, and D_clean runs
through a completely uninstrumented path.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from attribution import (  # noqa: E402
    SCOPE_PREFIX, attribute, evaluate_gradx_expectation, records_from_profiler,
)


def _bootstrap(repo_root):
    """Put the real repository on sys.path. A copied harness must not change
    what the repo root means -- hence --repo-root rather than a path guess."""
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(os.path.join(repo_root, ".git")):
        raise SystemExit(f"--repo-root {repo_root!r} is not a git repository")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _sync():
    torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# scope instrumentation -- profiled step only
# --------------------------------------------------------------------------- #
class ScopeProbe:
    """Wrap the STN entry points in `modules.modules` with record_function scopes.

    `modules.modules` does `from lpwm_stn import stn_crop, ...`, binding the
    dispatcher into its own namespace, so patching that namespace intercepts the
    model's calls while leaving the registry and every production file untouched.
    Installed on __enter__, removed on __exit__, so D_clean never pays for it.
    """

    TARGETS = {"stn_crop": "crop", "stn_paste": "paste",
               "create_masks_fast": "masks_fast", "create_masks_with_scale": "masks_scale"}

    def __init__(self):
        self._saved = {}
        self.calls = {}
        self.grad_flags = {}

    def _wrap(self, name, scope, fn):
        from torch.profiler import record_function

        def wrapper(*a, **kw):
            self.calls[name] = self.calls.get(name, 0) + 1
            if name == "stn_crop" and a:
                x = a[0]
                self.grad_flags.setdefault("stn_crop.x.requires_grad", []).append(
                    bool(getattr(x, "requires_grad", False)))
            with record_function(f"{SCOPE_PREFIX}{scope}"):
                return fn(*a, **kw)

        return wrapper

    def __enter__(self):
        import modules.modules as M
        for name, scope in self.TARGETS.items():
            fn = getattr(M, name, None)
            if fn is not None:
                self._saved[name] = fn
                setattr(M, name, self._wrap(name, scope, fn))
        return self

    def __exit__(self, *exc):
        import modules.modules as M
        for name, fn in self._saved.items():
            setattr(M, name, fn)
        self._saved.clear()
        return False


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def timed(fn, iters, warmup):
    """D_clean: uninstrumented, sync around every iteration, warmup excluded."""
    for _ in range(warmup):
        fn()
    _sync()
    s = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        s.append((time.perf_counter() - t0) * 1e3)
    s.sort()
    return {"D_clean_wall_ms": statistics.median(s), "mean_ms": statistics.fmean(s),
            "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))], "min_ms": s[0], "max_ms": s[-1],
            "spread_pct": 100.0 * (s[-1] - s[0]) / statistics.median(s) if s else 0.0,
            "iters": iters, "warmup": warmup, "instrumented": False}


def peak_of(fn):
    """Dedicated step, counters reset immediately before. Impact measurement only --
    never proof that a particular tensor was or was not allocated."""
    _sync()
    torch.cuda.reset_peak_memory_stats()
    fn()
    _sync()
    return {"peak_alloc_mb": torch.cuda.max_memory_allocated() / 1e6,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6}


def profile_step(fn, with_scopes=True):
    """D_prof: one profiled step with memory profiling and scope attribution."""
    from torch.profiler import ProfilerActivity, profile

    fn()                       # warm: never profile a compile
    _sync()
    probe = ScopeProbe() if with_scopes else None
    ctx = probe if probe is not None else _Null()
    with ctx:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True, record_shapes=False) as prof:
            t0 = time.perf_counter()
            fn()
            _sync()
            wall_ms = (time.perf_counter() - t0) * 1e3

    records, adapter_diag = records_from_profiler(prof)
    attr = attribute(records)
    out = {"D_prof_wall_ms": wall_ms,
           "adapter_strategy": adapter_diag["strategy"],
           "adapter_diagnostics": adapter_diag,
           "scopes_installed": bool(with_scopes)}
    out.update(attr.as_dict())
    if probe is not None:
        out["scope_call_counts"] = dict(probe.calls)
        out["grad_flags"] = {k: sorted(set(v)) for k, v in probe.grad_flags.items()}
    out["memory_events"] = _memory_events(prof)
    return out


def _memory_events(prof):
    """Memory rows from ``profile_memory=True``. Supporting evidence, not tensor identity.

    Three separations the first version got wrong (B7):
    * ranking is by **self** device memory, since aggregate rows include children
      and double-count nested operators;
    * only **positive** self-allocations are ranked -- negative rows are frees;
    * aggregate and deallocation rows are reported separately and labelled.
    """
    allocs, frees, aggregates = [], [], []
    try:
        for ev in prof.key_averages():
            agg = int(getattr(ev, "device_memory_usage", 0)
                      or getattr(ev, "cuda_memory_usage", 0) or 0)
            own = int(getattr(ev, "self_device_memory_usage", 0)
                      or getattr(ev, "self_cuda_memory_usage", 0) or 0)
            if own > 0:
                allocs.append({"op": ev.key, "self_device_memory_bytes": own})
            elif own < 0:
                frees.append({"op": ev.key, "self_device_memory_bytes": own})
            if agg:
                aggregates.append({"op": ev.key, "aggregate_device_memory_bytes": agg})
    except Exception as exc:                                        # pragma: no cover
        return {"available": False, "error": f"memory events unavailable: {exc}"}

    allocs.sort(key=lambda r: -r["self_device_memory_bytes"])
    frees.sort(key=lambda r: r["self_device_memory_bytes"])
    aggregates.sort(key=lambda r: -abs(r["aggregate_device_memory_bytes"]))
    return {
        "available": True,
        "note": "self-device allocations; aggregates include children and may double-count. "
                "Not proof of any particular tensor's identity.",
        "top_self_allocations": allocs[:15],
        "top_self_deallocations": frees[:10],
        "top_aggregate_rows_informational": aggregates[:10],
    }


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# crop grad-x microprofile
# --------------------------------------------------------------------------- #
def resolve_patch_size(cfg):
    """Glimpse side for the crop, config first (B3).

    BAIR sets ``patch_size: 8`` explicitly while
    ``round(anchor_s * (image_size - 1))`` gives 16, so deriving it would
    microprofile a shape the model never runs. The derivation is only a
    documented fallback for configs that omit the key.
    """
    if cfg.get("patch_size"):
        return int(cfg["patch_size"]), "config:patch_size"
    derived = int(round(cfg["anchor_s"] * (cfg["image_size"] - 1)))
    return derived, "fallback:round(anchor_s*(image_size-1))"


def crop_gradx_microprofile(cfg, backend, seed=0):
    """Isolate crop-backward and check the BACKEND'S OWN expectation (B2).

    In the BAIR end-to-end graph the encoder's image input does not require
    grad, so ``_StnCrop.backward`` never launches ``_stn_crop_grad_x_kernel``.
    Its absence there is correct behaviour, not a fallback -- which is exactly
    why "kernel implemented and tested" and "kernel exercised by this graph"
    have to be separated. Expectations differ per backend; see
    :func:`attribution.evaluate_gradx_expectation`.
    """
    import lpwm_stn
    from torch.profiler import ProfilerActivity, profile

    patch, patch_source = resolve_patch_size(cfg)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    bs, n_kp = 4, cfg["n_kp_enc"]
    S, ch = cfg["image_size"], cfg["ch"]

    def run(requires_grad_x):
        x = torch.rand(bs, ch, S, S, generator=g, device="cuda", requires_grad=requires_grad_x)
        kp = (2 * torch.rand(bs, n_kp, 2, generator=g, device="cuda") - 1).requires_grad_(True)
        zs = (4 * torch.rand(bs, n_kp, 2, generator=g, device="cuda") - 2).requires_grad_(True)
        lpwm_stn.stn_crop(x, kp, patch, z_scale=zs, padding_mode="border").sum().backward()
        _sync()

    results = {"backend": backend, "patch_size": patch, "patch_size_source": patch_source,
               "shape": [bs, ch, S, S], "n_kp": n_kp, "cases": {}}
    all_met = True
    for rg in (False, True):
        run(rg)                                    # warm; never profile a compile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run(rg)
        recs, diag = records_from_profiler(prof)
        names = sorted({r.name for r in recs if r.is_device})
        relevant = [n for n in names if "_stn_" in n or "grid_sampler" in n.lower()]
        verdict = evaluate_gradx_expectation(backend, rg, relevant)
        verdict["adapter_strategy"] = diag["strategy"]
        verdict["all_device_kernel_names"] = names
        results["cases"][f"x_requires_grad_{rg}"] = verdict
        all_met = all_met and bool(verdict["expectation_met"])
    results["expectation_met"] = all_met
    return results


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def loss_of(out):
    ld = out.get("loss_dict") if isinstance(out, dict) else None
    if isinstance(ld, dict):
        for k in ("loss", "total_loss", "elbo"):
            if torch.is_tensor(ld.get(k)):
                return ld[k]
        for v in ld.values():
            if torch.is_tensor(v) and v.ndim == 0:
                return v
    raise RuntimeError(f"no scalar loss; loss_dict keys={list(ld) if isinstance(ld, dict) else ld}")


def env_record():
    import triton
    return {"gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "triton": triton.__version__, "cuda_runtime": torch.version.cuda}


def repo_record(repo_root):
    """Provenance of the REAL repository, not of wherever the harness was copied."""
    def git(*a):
        # stderr captured: a deliberately bad root must not print `fatal: ...` (B8)
        return subprocess.check_output(["git", "-C", repo_root, *a], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return {"repo_root": repo_root, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "commit": git("rev-parse", "HEAD"),
                "clean": git("status", "--porcelain") == "",
                "remotes": git("remote", "-v").splitlines()}
    except Exception as exc:
        return {"repo_root": repo_root, "error": str(exc)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(_HERE)),
                    help="the real LPWM repository; imports, configs and git provenance "
                         "all resolve against it, so the harness may live anywhere")
    ap.add_argument("--config", required=True, help="absolute, or relative to --repo-root")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--gradx-microprofile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    repo_root = _bootstrap(args.repo_root)
    config = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    if not os.path.exists(config):
        raise SystemExit(f"config not found: {config}")

    import lpwm_stn
    from build_model import build, sequence_length

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, cfg, _ = build(config)

    g = torch.Generator(device="cuda")
    g.manual_seed(args.seed)
    x = torch.rand(args.batch_size or cfg["batch_size"], sequence_length(cfg), cfg["ch"],
                   cfg["image_size"], cfg["image_size"], generator=g, device="cuda")

    betas = {"beta_kl": cfg.get("beta_kl", 0.1), "beta_dyn": cfg.get("beta_dyn", 0.1),
             "beta_rec": cfg.get("beta_rec", 1.0)}
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 2e-4))

    def infer():
        model.eval()
        with torch.no_grad():
            model(x, deterministic=True, with_loss=False)

    def train():
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_of(model(x, deterministic=False, with_loss=True, **betas)).backward()
        opt.step()

    lpwm_stn.set_backend(args.backend)

    results = {"shapes": {"input": list(x.shape), "batch_size": x.shape[0],
                          "seq_len": x.shape[1], "timestep_horizon": cfg["timestep_horizon"],
                          "image_size": cfg["image_size"], "n_kp_enc": cfg["n_kp_enc"],
                          "anchor_s": cfg["anchor_s"], "ch": cfg["ch"], "dtype": str(x.dtype),
                          "patch_size": resolve_patch_size(cfg)[0],
                          "patch_size_source": resolve_patch_size(cfg)[1],
                          "model_params_M": sum(p.numel() for p in model.parameters()) / 1e6}}
    for mode, fn in (("inference", infer), ("training", train)):
        results[mode] = {"clean": timed(fn, args.iters, args.warmup),
                         "memory": peak_of(fn),
                         "profile": profile_step(fn)}
        d = results[mode]["clean"]["D_clean_wall_ms"]
        results[mode]["throughput_seq_per_s"] = x.shape[0] / (d / 1e3)
        results[mode]["throughput_frames_per_s"] = (x.shape[0] * x.shape[1]) / (d / 1e3)

    if args.gradx_microprofile:
        results["crop_gradx_microprofile"] = crop_gradx_microprofile(cfg, args.backend, args.seed)

    report = {"schema_version": 2,
              "experiment": f"lpwm-e2e-{os.path.basename(config).replace('.json','')}-{args.backend}",
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "config": config, "backend": args.backend, "seed": args.seed,
              "sync": "torch.cuda.synchronize() around every measured region",
              "metric_semantics": {
                  "D_clean_wall_ms": "median of the uninstrumented timed loop; the ONLY basis for "
                                     "end-to-end speedup and for any Amdahl argument",
                  "D_prof_wall_ms": "wall time of the profiled step; attribution only",
                  "total_device_kernel_ms": "summed self device time of CUDA kernels; NOT wall time",
                  "stn_share_of_device_kernel_time_pct": "share of kernel time, not of wall time"},
              "harness_path": _HERE,
              "env": env_record(), "repo": repo_record(repo_root), "results": results}

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
