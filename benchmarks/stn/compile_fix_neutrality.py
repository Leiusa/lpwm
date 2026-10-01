#!/usr/bin/env python
"""
Neutrality check for the torch.compile-friendliness fixes (patch sizes as Python int instead of numpy.int64;
explicit List[int] instead of torch.Size(...) in lpwm_stn/reference.py). The fixes must change nothing: the same
seeded eager computations in the old tree and the fixed tree have to give the same bytes.

`run` (one tree, one path; eager only, no torch.compile), repeated --reps times from the same initial state:
  * initial-weights hash (same seed -> must match across trees)
  * deterministic inference on one fixed batch (model.eval(), no_grad, deterministic=True):
    rec_rgb, dec_objects, bg_rgb, alpha_masks, mu_tot, obj_on
  * one seeded training forward + backward (train mode, the vgg loss, warmup=False): loss and every parameter gradient
`compare` (old vs new): per tensor, max |old - new| next to each tree's own repeat-to-repeat max |diff|.
  Verdict: "bitwise_identical" if every tensor matches byte for byte; "within_repeatability" if some differ but
  never by more than the larger of the two trees' own repeat differences; otherwise "DIFFERS". Fixed in advance.

    python compile_fix_neutrality.py run --repo-root R --config CFG --path reference|fused_cl --out X.pt
    python compile_fix_neutrality.py compare OLD.pt NEW.pt --out result.json
"""
import argparse
import json
import os
import random
import sys
import tempfile

import numpy as np

INFER_KEYS = ("rec_rgb", "dec_objects", "bg_rgb", "alpha_masks", "mu_tot", "obj_on")


def run(args):
    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    import bench_e2e as be
    be._bootstrap(repo)
    import torch
    import lpwm_stn
    import m1_common as MC
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from utils.loss_functions import LossLPIPS

    hw = MC.verify_hardware()
    MC.assert_agreed_environment(MC.set_agreed_environment())
    seed, path = args.seed, args.path
    backend = "triton" if path == "fused_cl" else "reference"
    fused = cl = path == "fused_cl"
    cfg = json.load(open(args.config))
    cfg.update(batch_size=args.batch_size, stn_backend=backend, fused_composite=fused, particle_dec_channels_last=cl, seed=seed)

    def seed_all():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_all()
    lpwm_stn.set_backend(backend)
    ds = get_image_dataset(cfg["ds"], cfg["root"], mode="train", image_size=cfg["image_size"])
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=args.batch_size, num_workers=0, drop_last=True,
                                         generator=torch.Generator().manual_seed(seed))
    x = next(iter(loader))[0].cuda()
    with tempfile.TemporaryDirectory() as tmp:
        cp = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(cp, "w"))
        model, _, _ = build(cp, "cuda")
    init_hash = be.tensors_sha256(model.parameters())
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    recon_loss_func = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).to("cuda")
    kw = dict(warmup=False, with_loss=True, beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"], kl_balance=cfg["kl_balance"],
              recon_loss_type=cfg["recon_loss_type"], recon_loss_func=recon_loss_func, beta_obj=cfg.get("beta_obj", 0.0),
              return_alpha_masks=False)
    patch_types = {"DLP.obj_patch_size": type(model.obj_patch_size).__name__,
                   "DLPDecoder.obj_patch_size": type(model.decoder_module.obj_patch_size).__name__}
    reps = []
    for _ in range(args.reps):
        model.load_state_dict(init_state)
        seed_all()
        model.eval()
        with torch.no_grad():
            mo = model(x, deterministic=True, with_loss=False)
        infer = {k: mo[k].detach().float().cpu().clone() for k in INFER_KEYS if mo.get(k) is not None}
        del mo
        model.train()
        model.zero_grad(set_to_none=True)
        seed_all()
        out = model(x, **kw)
        loss = out["loss_dict"]["loss"]
        loss.backward()
        grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        reps.append({"infer": infer, "loss": loss.detach().cpu().clone(), "grads": grads})
        del out, loss
    torch.save({"meta": {"repo": repo, "path": path, "seed": seed, "gpu": hw["nvidia_smi"][0], "node": hw["node"],
                         "init_weights_sha256": init_hash, "batch_sha256": be.tensors_sha256([x]), "patch_size_types": patch_types,
                         "code_hashes": MC.code_hashes(repo), "script_sha256": be.file_sha256(os.path.abspath(__file__))},
                "reps": reps}, args.out)
    print(f"[run] {path} @ {repo}: init {init_hash[:16]} batch {be.tensors_sha256([x])[:16]} patch types {patch_types} "
          f"loss {[float(r['loss']) for r in reps]}", flush=True)
    return 0


def compare(args):
    import torch

    old, new = torch.load(args.old, weights_only=False), torch.load(args.new, weights_only=False)

    def flat(rep):
        d = {f"infer/{k}": v for k, v in rep["infer"].items()}
        d["loss"] = rep["loss"]
        d.update({f"grad/{k}": v for k, v in rep["grads"].items()})
        return d

    def maxdiff(a, b):
        if a.shape != b.shape:
            return float("inf")
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0

    o0, n0 = flat(old["reps"][0]), flat(new["reps"][0])
    o1 = flat(old["reps"][1]) if len(old["reps"]) > 1 else o0
    n1 = flat(new["reps"][1]) if len(new["reps"]) > 1 else n0
    rows, keys_missing = {}, sorted(set(o0) ^ set(n0))
    for k in sorted(set(o0) & set(n0)):
        cross = maxdiff(o0[k], n0[k])
        rows[k] = {"cross_max_abs": cross, "old_repeat_max_abs": maxdiff(o0[k], o1[k]), "new_repeat_max_abs": maxdiff(n0[k], n1[k]),
                   "bitwise_equal": bool(torch.equal(o0[k], n0[k]))}
    same_init = old["meta"]["init_weights_sha256"] == new["meta"]["init_weights_sha256"]
    same_batch = old["meta"]["batch_sha256"] == new["meta"]["batch_sha256"]
    if not (same_init and same_batch) or keys_missing:
        verdict = "INVALID_COMPARISON"
    elif all(r["bitwise_equal"] for r in rows.values()):
        verdict = "bitwise_identical"
    elif all(r["cross_max_abs"] <= max(r["old_repeat_max_abs"], r["new_repeat_max_abs"]) for r in rows.values()):
        verdict = "within_repeatability"
    else:
        verdict = "DIFFERS"
    worst = sorted(rows.items(), key=lambda kv: -kv[1]["cross_max_abs"])[:5]
    result = {"verdict": verdict, "path": new["meta"]["path"], "same_initial_weights": same_init, "same_batch": same_batch,
              "keys_missing_in_one_tree": keys_missing, "tensors_compared": len(rows),
              "tensors_bitwise_equal": sum(r["bitwise_equal"] for r in rows.values()),
              "old_patch_size_types": old["meta"]["patch_size_types"], "new_patch_size_types": new["meta"]["patch_size_types"],
              "old_code_hashes": old["meta"]["code_hashes"], "new_code_hashes": new["meta"]["code_hashes"],
              "largest_cross_differences": dict(worst), "per_tensor": rows}
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"[compare] {result['path']}: {verdict} | {result['tensors_bitwise_equal']}/{len(rows)} tensors bitwise equal | "
          f"patch types old {result['old_patch_size_types']} new {result['new_patch_size_types']}", flush=True)
    return 0 if verdict in ("bitwise_identical", "within_repeatability") else 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo-root", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--path", choices=("reference", "fused_cl"), required=True)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
