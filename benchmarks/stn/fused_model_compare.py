#!/usr/bin/env python
"""
Whole-model comparison of three static-DLP paths, on synthetic inputs (torch.rand images; not a dataset):

  reference  PyTorch STN reference, fused_composite off
  triton     Triton crop/paste (lpwm_stn backend), fused_composite off
  fused      Triton crop + fused paste/composite (decoder switch fused_composite=True)

  dump / diffdump   one train-mode step saved to a file and compared across two source trees; used for the
                    "switch off == 91fcae1" regression (outputs and loss must be bit-identical; gradients,
                    which use atomics, are reported as metrics)
  numerics          same initial weights, same input, same random state: loss and parameter gradients for
                    every path, each path run twice so the run-to-run floor is visible. Reported, not gated.
  bench             clean wall time, throughput and peak allocated memory of the full training step, one path
                    per process. model_output stays alive until backward ends, as in train_dlp.py.
  smoke             a short training run from identical initial weights on one fixed batch. It checks that
                    training runs and stays finite; it says nothing about convergence or quality.

Compare paths only inside one Slurm allocation.
"""

import argparse
import json
import os
import socket
import sys
import tempfile
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_e2e as be  # noqa: E402

PATHS = ("reference", "triton", "fused")


class Ctx:
    def __init__(self, args):
        self.repo_root = be._bootstrap(args.repo_root)
        self.config = args.config if os.path.isabs(args.config) else os.path.join(self.repo_root, args.config)
        import lpwm_stn
        from build_model import build, sequence_length
        self.lpwm_stn, self.build, self.sequence_length = lpwm_stn, build, sequence_length
        self.seed = args.seed
        self.tmp = tempfile.mkdtemp(prefix="fused_cmp_")

    def model(self, path):
        with open(self.config) as f:
            cfg = json.load(f)
        cfg["fused_composite"] = path == "fused"          # ignored by a source tree that predates the switch
        p = os.path.join(self.tmp, f"cfg_{path}.json")
        with open(p, "w") as f:
            json.dump(cfg, f)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        model, cfg, _ = self.build(p, "cuda")
        self.lpwm_stn.set_backend("reference" if path == "reference" else "triton")
        return model, cfg

    def inputs(self, cfg, batch_size):
        g = torch.Generator(device="cuda")
        g.manual_seed(self.seed)
        x = torch.rand(batch_size, self.sequence_length(cfg), cfg["ch"], cfg["image_size"], cfg["image_size"],
                       generator=g, device="cuda")
        betas = {"beta_kl": cfg.get("beta_kl", 0.1), "beta_dyn": cfg.get("beta_dyn", 0.1),
                 "beta_rec": cfg.get("beta_rec", 1.0)}
        return x, betas


def one_step(model, x, betas, seed):
    """Forward + backward with model_output alive until backward finishes."""
    model.train()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    out = model(x, deterministic=False, with_loss=True, return_alpha_masks=False, **betas)
    loss = be.loss_of(out)
    loss.backward()
    torch.cuda.synchronize()
    return {"loss_dict": {k: float(v) for k, v in out["loss_dict"].items() if torch.is_tensor(v) and v.ndim == 0},
            "rec": out["rec"].detach().clone(), "dec_objects": out["dec_objects"].detach().clone(),
            "grads": {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in model.named_parameters()}}


def tensor_metrics(a, b):
    d = a.double() - b.double()
    nb = float(torch.linalg.vector_norm(b.double()))
    return {"finite": bool(torch.isfinite(a).all()), "max_abs": float(d.abs().max()),
            "rel_l2": (float(torch.linalg.vector_norm(d)) / nb) if nb > 0 else None, "bit_identical": bool(torch.equal(a, b))}


MATERIAL_FRACTION = 1e-6   # a parameter is "material" if its gradient norm is at least this fraction of the total


def grad_metrics(ga, gb):
    """Aggregate and per-parameter gradient differences. Near-zero gradients make per-parameter relative L2
    meaningless, so the worst value is also reported over the material parameters only."""
    total_sq = sum(float((b.double() ** 2).sum()) for b in gb.values() if b is not None and bool(torch.isfinite(b).all()))
    num = den = na = dot = 0.0
    worst, worst_name, worst_mat, worst_mat_name, n_mat = 0.0, None, 0.0, None, 0
    none_mismatch = nonfinite = with_grad = 0
    for n, b in gb.items():
        a = ga[n]
        if (a is None) != (b is None):
            none_mismatch += 1
            continue
        if a is None:
            continue
        with_grad += 1
        if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())):
            nonfinite += 1
            continue
        a64, b64 = a.double(), b.double()
        s, t = float(((a64 - b64) ** 2).sum()), float((b64 ** 2).sum())
        num, den, na, dot = num + s, den + t, na + float((a64 ** 2).sum()), dot + float((a64 * b64).sum())
        if t > 0:
            r = (s / t) ** 0.5
            if r > worst:
                worst, worst_name = r, n
            if t >= (MATERIAL_FRACTION ** 2) * total_sq:
                n_mat += 1
                if r > worst_mat:
                    worst_mat, worst_mat_name = r, n
    return {"global_rel_l2": (num / den) ** 0.5 if den > 0 else None,
            "cosine": dot / ((na ** 0.5) * (den ** 0.5)) if na > 0 and den > 0 else None,
            "worst_param_rel_l2": worst, "worst_param": worst_name,
            "worst_material_param_rel_l2": worst_mat, "worst_material_param": worst_mat_name, "n_material_params": n_mat,
            "n_params": len(gb), "n_params_with_grad": with_grad,
            "n_grad_presence_mismatch": none_mismatch, "n_params_nonfinite": nonfinite}


def compare_steps(a, b):
    lo = {k: (abs(a["loss_dict"][k] - b["loss_dict"][k]) / abs(b["loss_dict"][k]) if b["loss_dict"][k] != 0 else abs(a["loss_dict"][k]))
          for k in b["loss_dict"]}
    return {"loss_rel_diff": lo["loss"], "loss_dict_max_rel_diff": max(lo.values()),
            "rec": tensor_metrics(a["rec"], b["rec"]), "dec_objects": tensor_metrics(a["dec_objects"], b["dec_objects"]),
            "grads": grad_metrics(a["grads"], b["grads"])}


def provenance(ctx, args):
    return {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "env": be.env_record(),
            "repo": be.repo_record(ctx.repo_root), "config": ctx.config, "config_sha256": be.file_sha256(ctx.config),
            "seed": args.seed, "script_sha256": be.file_sha256(os.path.abspath(__file__)), "command": sys.argv}


# ------------------------------------------------------------------ dump / diffdump
def cmd_dump(args):
    ctx = Ctx(args)
    model, cfg = ctx.model(args.path)
    x, betas = ctx.inputs(cfg, args.batch_size)
    rec = one_step(model, x, betas, args.seed)
    rec["grads"] = {n: (g.cpu() if g is not None else None) for n, g in rec["grads"].items()}
    rec.update(path=args.path, batch_size=args.batch_size, weights_sha256=be.tensors_sha256(model.parameters()),
               input_sha256=be.tensors_sha256([x]), repo_root=ctx.repo_root, repo=be.repo_record(ctx.repo_root),
               fused_composite_calls=model.decoder_module.fused_composite_calls if hasattr(model.decoder_module, "fused_composite_calls") else None)
    rec["rec"], rec["dec_objects"] = rec["rec"].cpu(), rec["dec_objects"].cpu()
    torch.save(rec, args.out)
    print("wrote", args.out, "| loss", rec["loss_dict"]["loss"])


def cmd_diffdump(args):
    a, b = torch.load(args.a, weights_only=False), torch.load(args.b, weights_only=False)
    print(f"A: {a['repo_root']} commit {a['repo'].get('commit', '?')[:8]} | B: {b['repo_root']} commit {b['repo'].get('commit', '?')[:8]}")
    print("weights identical:", a["weights_sha256"] == b["weights_sha256"], "| input identical:", a["input_sha256"] == b["input_sha256"])
    loss_equal = all(a["loss_dict"][k] == b["loss_dict"][k] for k in b["loss_dict"]) and set(a["loss_dict"]) == set(b["loss_dict"])
    rec_equal, dec_equal = torch.equal(a["rec"], b["rec"]), torch.equal(a["dec_objects"], b["dec_objects"])
    print(f"loss_dict bit-identical: {loss_equal} | rec bit-identical: {rec_equal} | dec_objects bit-identical: {dec_equal}")
    gm = grad_metrics(a["grads"], b["grads"])
    print("grad metrics (atomics make bit-identity inappropriate):", json.dumps(gm))
    res = {"weights_identical": a["weights_sha256"] == b["weights_sha256"], "input_identical": a["input_sha256"] == b["input_sha256"],
           "loss_bit_identical": loss_equal, "rec_bit_identical": rec_equal, "dec_objects_bit_identical": dec_equal, "grads": gm,
           "a": {"repo_root": a["repo_root"], "repo": a["repo"], "path": a["path"]}, "b": {"repo_root": b["repo_root"], "repo": b["repo"], "path": b["path"]}}
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
    return 0 if (res["weights_identical"] and res["input_identical"] and loss_equal and rec_equal and dec_equal
                 and gm["n_grad_presence_mismatch"] == 0 and gm["n_params_nonfinite"] == 0) else 1


# ------------------------------------------------------------------ numerics
def cmd_numerics(args):
    ctx = Ctx(args)
    report = {"kind": "whole-model numerics: same initial weights, input and random state; reported, not gated",
              "provenance": provenance(ctx, args), "batches": {}}
    for bs in args.batch_sizes:
        runs, meta = {}, {}
        for path in PATHS:
            model, cfg = ctx.model(path)
            x, betas = ctx.inputs(cfg, bs)
            meta[path] = {"weights_sha256": be.tensors_sha256(model.parameters()), "input_sha256": be.tensors_sha256([x])}
            runs[path] = [one_step(model, x, betas, args.seed) for _ in range(2)]
            dec = model.decoder_module
            meta[path]["fused_composite_calls"] = dec.fused_composite_calls
            meta[path]["forward_passes"] = 2
            del model
            torch.cuda.empty_cache()
        pairs = {"triton_vs_reference": ("triton", "reference"), "fused_vs_reference": ("fused", "reference"),
                 "fused_vs_triton": ("fused", "triton")}
        rec = {"meta": meta,
               "same_weights_and_input_everywhere": len({m["weights_sha256"] for m in meta.values()}) == 1 and len({m["input_sha256"] for m in meta.values()}) == 1,
               "cross_path": {n: compare_steps(runs[a][0], runs[b][0]) for n, (a, b) in pairs.items()},
               "run_to_run_floor": {p: compare_steps(runs[p][1], runs[p][0]) for p in PATHS}}
        report["batches"][str(bs)] = rec
        print(f"\n=== B={bs}: same weights/input across paths: {rec['same_weights_and_input_everywhere']} | fused calls "
              f"{ {p: meta[p]['fused_composite_calls'] for p in PATHS} } (expected fused=2, others=0)")
        print("%-24s %-10s %-11s %-11s %-11s %-9s %-26s %s" % ("pair", "loss rel", "rec relL2", "trans relL2", "grad relL2", "cosine", "worst material param (relL2)", "presence/nonfinite"))
        for name, m in list(rec["cross_path"].items()) + [(f"floor: {p}", m) for p, m in rec["run_to_run_floor"].items()]:
            g = m["grads"]
            f = lambda v: "n/a" if v is None else "%.2e" % v  # noqa: E731
            print("%-24s %-10s %-11s %-11s %-11s %-9s %-26s %d/%d" % (name, f(m["loss_rel_diff"]), f(m["rec"]["rel_l2"]), f(m["dec_objects"]["rel_l2"]),
                                                                       f(g["global_rel_l2"]), "%.9f" % g["cosine"] if g["cosine"] else "n/a",
                                                                       "%s (%s)" % (g["worst_material_param"], f(g["worst_material_param_rel_l2"])), g["n_grad_presence_mismatch"], g["n_params_nonfinite"]))
    json.dump(report, open(args.out, "w"), indent=2)
    print("wrote", args.out)


# ------------------------------------------------------------------ bench
def cmd_bench(args):
    ctx = Ctx(args)
    model, cfg = ctx.model(args.path)
    x, betas = ctx.inputs(cfg, args.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 2e-4))
    calls = [0]
    model.decoder_module.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + 1))
    weights_sha, input_sha = be.tensors_sha256(model.parameters()), be.tensors_sha256([x])
    init_loss = be.initial_loss(model, x, betas, False, args.seed)

    def train():
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(x, deterministic=False, with_loss=True, return_alpha_masks=False, **betas)  # alive until backward ends
        be.loss_of(out).backward()
        opt.step()

    clean = be.timed(train, args.iters, args.warmup)
    mem = be.peak_of(train)
    dec = model.decoder_module
    res = {"kind": "synthetic-input static DLP whole-step benchmark (torch.rand images; not real-dataset training)",
           "path": args.path, "batch_size": args.batch_size, "input_shape": list(x.shape), "return_alpha_masks": False,
           "model_output_retention": "alive until backward ends, as in train_dlp.py",
           "initial_weights_sha256": weights_sha, "input_sha256": input_sha, "initial_loss": init_loss,
           "clean": clean, "memory": mem,
           "throughput_frames_per_s": args.batch_size / (clean["D_clean_wall_ms"] / 1e3),
           "decoder_forward_passes": calls[0], "fused_composite_calls": dec.fused_composite_calls if hasattr(dec, "fused_composite_calls") else None,
           "provenance": provenance(ctx, args)}
    json.dump(res, open(args.out, "w"), indent=2)
    print("wrote", args.out, "| %.2f ms/step, peak %.0f MB" % (clean["D_clean_wall_ms"], mem["peak_alloc_mb"]))


# ------------------------------------------------------------------ smoke
def cmd_smoke(args):
    ctx = Ctx(args)
    report = {"kind": "short training smoke on ONE fixed synthetic batch: checks the run stays finite; not convergence, not quality",
              "provenance": provenance(ctx, args), "steps": args.steps, "paths": {}}
    for path in PATHS:
        model, cfg = ctx.model(path)
        x, betas = ctx.inputs(cfg, args.batch_size)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 2e-4))
        losses, gnorms, first_bad = [], [], None
        for i in range(args.steps):
            model.train()
            opt.zero_grad(set_to_none=True)
            out = model(x, deterministic=False, with_loss=True, return_alpha_masks=False, **betas)
            loss = be.loss_of(out)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
            opt.step()
            losses.append(float(loss))
            gnorms.append(gn)
            if first_bad is None and not (torch.isfinite(loss).item() and gn == gn and gn != float("inf")):
                first_bad = i
        report["paths"][path] = {"loss": losses, "grad_norm": gnorms, "first_non_finite_step": first_bad,
                                 "fused_composite_calls": model.decoder_module.fused_composite_calls,
                                 "final_loss": losses[-1], "first_loss": losses[0]}
        del model, opt
        torch.cuda.empty_cache()
    ref = report["paths"]["reference"]["loss"]
    for path in ("triton", "fused"):
        l = report["paths"][path]["loss"]
        report["paths"][path]["max_rel_loss_deviation_from_reference"] = max(abs(a - b) / abs(b) for a, b in zip(l, ref))
    json.dump(report, open(args.out, "w"), indent=2)
    for path in PATHS:
        p = report["paths"][path]
        print("%-10s loss %.3f -> %.3f | non-finite step: %s | max rel dev from reference: %s | fused calls %d" % (
            path, p["first_loss"], p["final_loss"], p["first_non_finite_step"], p.get("max_rel_loss_deviation_from_reference", "-"), p["fused_composite_calls"]))
    print("wrote", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--repo-root", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)

    p = sub.add_parser("dump"); common(p); p.add_argument("--path", choices=PATHS, required=True); p.add_argument("--batch-size", type=int, default=2)
    p = sub.add_parser("diffdump"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--out", default=None)
    p = sub.add_parser("numerics"); common(p); p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    p = sub.add_parser("bench"); common(p); p.add_argument("--path", choices=PATHS, required=True)
    p.add_argument("--batch-size", type=int, required=True); p.add_argument("--iters", type=int, default=30); p.add_argument("--warmup", type=int, default=10)
    p = sub.add_parser("smoke"); common(p); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--steps", type=int, default=100)
    args = ap.parse_args(argv)
    return {"dump": cmd_dump, "diffdump": cmd_diffdump, "numerics": cmd_numerics, "bench": cmd_bench, "smoke": cmd_smoke}[args.cmd](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
