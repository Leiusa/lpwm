#!/usr/bin/env python
"""
Memory and time attribution of ONE static-DLP training step, for one STN backend.

Answers "where does the step's memory go, and how much of it is the paste ->
alpha/depth composite chain" without touching production code:

  1. Phase memory of the real step: peak in forward, tensors retained for backward,
     peak in backward, peak in the optimizer step, all measured against a baseline taken
     after zero_grad (weights + optimizer state).
  2. Saved tensors: every tensor autograd saves for backward is recorded through
     torch.autograd.graph.saved_tensors_hooks, deduplicated by storage, excluding weights
     and the input, and attributed to the innermost active scope
     (enc:crop, encoder, dec:particle_dec, dec:paste, dec:composite, dec:bg_dec, other).
  3. Isolated blocks: the paste (translate_patches) and the composite
     (get_objects_alpha_rgb_with_depth) run alone on tensors captured from the real
     forward, with forward time, backward time, and their own peak / retained / saved
     memory. Autograd-thread backward kernels cannot be attributed by profiler scopes, so
     the backward cost of a block is measured this way instead.

One backend per process. Compare runs made inside the same Slurm allocation only.

    python attribute_memory.py --repo-root /path/to/lpwm --config static_bair128.json \
        --backend triton --batch-size 16 --no-return-alpha-masks --out attr.json
"""

import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_e2e as be  # noqa: E402

MB = 1e6


class Ledger:
    """Saved-for-backward tensors, deduplicated by storage, attributed to a scope stack."""

    def __init__(self, exclude_ptrs=(), cuda_only=True):
        self.exclude = set(exclude_ptrs)
        self.cuda_only = cuda_only
        self.stack = ["other"]
        self.seen = set()
        self.nbytes = defaultdict(int)
        self.count = defaultdict(int)
        self.largest = []

    def push(self, label):
        self.stack.append(label)

    def pop(self):
        self.stack.pop()

    @contextmanager
    def scope(self, label):
        self.push(label)
        try:
            yield
        finally:
            self.pop()

    def pack(self, tensor):
        if tensor.is_cuda or not self.cuda_only:
            storage = tensor.untyped_storage()
            ptr = storage.data_ptr()
            if ptr not in self.exclude and ptr not in self.seen:
                self.seen.add(ptr)
                label = self.stack[-1]
                self.nbytes[label] += storage.nbytes()
                self.count[label] += 1
                self.largest.append((storage.nbytes(), label, list(tensor.shape), str(tensor.dtype)))
        return tensor

    @staticmethod
    def unpack(tensor):
        return tensor

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def summary(self, top=8):
        self.largest.sort(key=lambda r: -r[0])
        groups = defaultdict(lambda: [0, 0])
        for nbytes, scope, shape, dtype in self.largest:
            g = groups[(scope, tuple(shape), dtype)]
            g[0] += 1
            g[1] += nbytes
        grouped = sorted(({"scope": s, "shape": list(sh), "dtype": dt, "n_storages": n, "total_mb": b / MB,
                           "each_mb": b / n / MB} for (s, sh, dt), (n, b) in groups.items()),
                         key=lambda r: (r["scope"], -r["total_mb"]))
        return {"total_mb": sum(self.nbytes.values()) / MB,
                "by_scope_mb": {k: v / MB for k, v in sorted(self.nbytes.items(), key=lambda kv: -kv[1])},
                "n_storages_by_scope": dict(self.count),
                "largest": [{"mb": n / MB, "scope": s, "shape": sh, "dtype": dt} for n, s, sh, dt in self.largest[:top]],
                "grouped_complete": grouped}


def install_scopes(ledger, model, modules_ns):
    """Label decoder methods, encoder/decoder submodules and stn_crop. Returns an undo function."""
    undo = []
    dec = model.decoder_module
    for name, label in (("translate_patches", "dec:paste"),
                        ("get_objects_alpha_rgb_with_depth", "dec:composite")):
        original = getattr(dec, name)

        def wrapper(*a, _o=original, _l=label, **k):
            with ledger.scope(_l):
                return _o(*a, **k)

        setattr(dec, name, wrapper)
        undo.append(lambda n=name: delattr(dec, n))
    for module, label in ((model.encoder_module, "encoder"), (dec.particle_dec, "dec:particle_dec"),
                          (dec.bg_dec, "dec:bg_dec")):
        pre = module.register_forward_pre_hook(lambda m, i, _l=label: ledger.push(_l))
        post = module.register_forward_hook(lambda m, i, o: ledger.pop())
        undo += [pre.remove, post.remove]
    original_crop = modules_ns.stn_crop

    def crop(*a, **k):
        with ledger.scope("enc:crop"):
            return original_crop(*a, **k)

    modules_ns.stn_crop = crop
    undo.append(lambda: setattr(modules_ns, "stn_crop", original_crop))

    def restore():
        for fn in undo:
            fn()
    return restore


def bench_block(fn, leaves, iters, warmup, seed=0):
    """Time and memory of ``fn(*leaves)`` forward and backward, alone on the given leaves."""
    outs = [o for o in fn(*leaves) if o is not None]
    torch.manual_seed(seed)
    cots = [torch.rand_like(o) for o in outs]
    del outs

    def clear():
        for leaf in leaves:
            leaf.grad = None

    fwd, bwd = [], []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        outs = [o for o in fn(*leaves) if o is not None]
        e1.record()
        torch.autograd.backward(outs, cots)
        e2.record()
        torch.cuda.synchronize()
        if i >= warmup:
            fwd.append(e0.elapsed_time(e1))
            bwd.append(e1.elapsed_time(e2))
        del outs
        clear()

    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ledger = Ledger({leaf.untyped_storage().data_ptr() for leaf in leaves})
    with ledger.hooks():
        outs = [o for o in fn(*leaves) if o is not None]
    torch.cuda.synchronize()
    fwd_peak, retained = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.autograd.backward(outs, cots)
    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated()
    del outs
    clear()
    return {"fwd_ms_median": statistics.median(fwd), "bwd_ms_median": statistics.median(bwd),
            "fwd_ms_min": min(fwd), "fwd_ms_max": max(fwd), "bwd_ms_min": min(bwd), "bwd_ms_max": max(bwd),
            "iters": iters, "warmup": warmup,
            "fwd_peak_above_input_mb": (fwd_peak - base) / MB,
            "retained_after_fwd_above_input_mb": (retained - base) / MB,
            "saved_for_backward_mb": ledger.summary()["total_mb"],
            "bwd_peak_above_input_mb": (bwd_peak - base) / MB}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--block-iters", type=int, default=20)
    ap.add_argument("--return-alpha-masks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    repo_root = be._bootstrap(args.repo_root)
    config = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)

    import lpwm_stn
    import modules.modules as M
    from build_model import build, sequence_length

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, cfg, _ = build(config)
    if model.is_dynamics_model:
        raise SystemExit("this attribution is for the static DLP path (timestep_horizon == 1)")
    g = torch.Generator(device="cuda")
    g.manual_seed(args.seed)
    x = torch.rand(args.batch_size or cfg["batch_size"], sequence_length(cfg), cfg["ch"],
                   cfg["image_size"], cfg["image_size"], generator=g, device="cuda")
    betas = {"beta_kl": cfg.get("beta_kl", 0.1), "beta_dyn": cfg.get("beta_dyn", 0.1),
             "beta_rec": cfg.get("beta_rec", 1.0)}
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 2e-4))
    alpha = args.return_alpha_masks
    weights_sha, input_sha = be.tensors_sha256(model.parameters()), be.tensors_sha256([x])
    lpwm_stn.set_backend(args.backend)

    def forward():
        return be.loss_of(model(x, deterministic=False, with_loss=True, return_alpha_masks=alpha, **betas))

    def step():
        model.train()
        opt.zero_grad(set_to_none=True)
        forward().backward()
        opt.step()

    for _ in range(args.warmup_steps):
        step()

    exclude = {p.untyped_storage().data_ptr() for p in model.parameters()} | {x.untyped_storage().data_ptr()}
    ledger = Ledger(exclude)
    restore = install_scopes(ledger, model, M)
    model.train()
    opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        with ledger.hooks():
            loss = forward()
        torch.cuda.synchronize()
        fwd_peak, after_fwd = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        bwd_peak, after_bwd = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        opt.step()
        torch.cuda.synchronize()
        opt_peak = torch.cuda.max_memory_allocated()
    finally:
        restore()
    del loss
    saved = ledger.summary()
    step_rec = {"baseline_allocated_mb": base / MB,
                "forward_peak_mb": fwd_peak / MB, "forward_peak_above_baseline_mb": (fwd_peak - base) / MB,
                "retained_after_forward_above_baseline_mb": (after_fwd - base) / MB,
                "backward_peak_mb": bwd_peak / MB, "backward_peak_above_baseline_mb": (bwd_peak - base) / MB,
                "after_backward_above_baseline_mb": (after_bwd - base) / MB,
                "optimizer_peak_mb": opt_peak / MB,
                "step_peak_mb": max(fwd_peak, bwd_peak, opt_peak) / MB,
                "peak_phase": max((("forward", fwd_peak), ("backward", bwd_peak), ("optimizer", opt_peak)),
                                  key=lambda kv: kv[1])[0]}
    saved["coverage_of_retained"] = (saved["total_mb"] / ((after_fwd - base) / MB)
                                     if after_fwd > base else None)

    cap = {}
    dec = model.decoder_module
    o_paste, o_comp = dec.translate_patches, dec.get_objects_alpha_rgb_with_depth

    def cap_paste(*a, **k):
        out = o_paste(*a, **k)
        cap["paste_in"] = [t.detach() for t in a[:3]]
        cap["stack"] = out.detach()
        return out

    def cap_comp(a_obj, rgb_obj, *a, **k):
        cap["obj_on"], cap["z_depth"] = k["obj_on"].detach(), k["z_depth"].detach()
        return o_comp(a_obj, rgb_obj, *a, **k)

    dec.translate_patches, dec.get_objects_alpha_rgb_with_depth = cap_paste, cap_comp
    try:
        model.train()
        torch.manual_seed(args.seed)
        with torch.no_grad():
            model(x, deterministic=False, with_loss=False, return_alpha_masks=alpha)
    finally:
        del dec.translate_patches, dec.get_objects_alpha_rgb_with_depth
    kp, patches, scale = (t.clone().requires_grad_(True) for t in cap["paste_in"])
    stack = cap["stack"].clone().requires_grad_(True)
    obj_on, z_depth = (cap[k].clone().requires_grad_(True) for k in ("obj_on", "z_depth"))
    logical = {"paste_stack_shape": list(stack.shape), "paste_stack_mb": stack.numel() * 4 / MB,
               "per_particle_planes": {"rgba_obj_mb": stack.numel() // 4 * 3 * 4 / MB,
                                       "importance_or_alpha_plane_mb": stack.numel() // 4 * 4 / MB}}
    del cap

    def paste_fn(k_, p_, s_):
        return (dec.translate_patches(k_, p_, s_),)

    def composite_fn(st_, on_, dp_):
        a_obj, rgb_obj = torch.split(st_, [1, st_.shape[2] - 1], dim=2)
        return dec.get_objects_alpha_rgb_with_depth(a_obj, rgb_obj, obj_on=on_, z_depth=dp_,
                                                    return_alpha_masks=alpha)

    blocks = {"paste": bench_block(paste_fn, [kp, patches, scale], args.block_iters, 5, args.seed),
              "composite": bench_block(composite_fn, [stack, obj_on, z_depth], args.block_iters, 5, args.seed)}

    report = {"schema_version": 1, "kind": "memory/time attribution, synthetic-input static DLP training step",
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "backend": args.backend, "return_alpha_masks": alpha, "seed": args.seed,
              "batch_size": x.shape[0], "input_shape": list(x.shape),
              "config": config, "config_sha256": be.file_sha256(config), "config_content": cfg,
              "script_sha256": be.file_sha256(os.path.abspath(__file__)),
              "initial_weights_sha256": weights_sha, "input_sha256": input_sha,
              "env": be.env_record(), "repo": be.repo_record(repo_root),
              "step": step_rec, "saved_tensors": saved, "logical_sizes": logical, "isolated_blocks": blocks}
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    print(json.dumps({"step": step_rec, "saved_by_scope_mb": saved["by_scope_mb"],
                      "blocks": {k: {kk: round(vv, 3) for kk, vv in v.items() if kk.endswith(("_ms_median", "_mb"))}
                                 for k, v in blocks.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
