#!/usr/bin/env python
"""
Neutrality check for a code change that must not change what LPWM computes (here: the numpy-int -> int patch-size fix,
which lets torch.compile trace the particle decoder). Eager only, no torch.compile; the same script is run against the
old tree and the new tree (--repo-root), then the two dumps are compared.

`run`, repeated --reps times from the same initial weights:
  * initial-weights hash (same seed -> must match across trees)
  * deterministic inference on one fixed batch (model.eval(), no_grad, deterministic=True): rec_rgb, dec_objects,
    bg_rgb, mu_tot, obj_on (+ alpha_masks when returned)
  * one seeded training forward + backward (train mode, vgg loss, warmup=False, the train_lpwm.py loss arguments):
    loss and every parameter gradient
`compare` (old vs new): per tensor, max |old - new| next to each tree's own repeat-to-repeat max |diff|.
  Verdict, fixed in advance: "bitwise_identical" if every tensor matches byte for byte; "within_repeatability" if some
  differ but never by more than the larger of the two trees' own repeat differences; otherwise "DIFFERS".

    python lpwm_fix_neutrality.py run --repo-root lpwm_nofix --config CFG --out old.pt
    python lpwm_fix_neutrality.py run --repo-root lpwm --config CFG --out new.pt
    python lpwm_fix_neutrality.py compare old.pt new.pt --out neutrality.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
import tempfile

import numpy as np

INFER_KEYS = ("rec_rgb", "dec_objects", "bg_rgb", "mu_tot", "obj_on", "alpha_masks")


def run(args):
    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    sys.path.insert(0, repo)
    import torch
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_video_dataset
    from utils.loss_functions import LossLPIPS

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False   # strict fp32 matmul for a byte comparison
    cfg = json.load(open(args.config))
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    seed = args.seed

    def seed_all():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    seed_all()
    lpwm_stn.set_backend(cfg.get("stn_backend") or "reference")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(p, "w"))
        model, _, _ = build(p, "cuda")
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    h = hashlib.sha256()
    for k in sorted(init):
        h.update(k.encode()); h.update(init[k].cpu().contiguous().numpy().tobytes())
    T = cfg["timestep_horizon"]
    ds = get_video_dataset(cfg["ds"], cfg["root"], seq_len=T + 1, mode="train", image_size=cfg["image_size"])
    x = torch.stack([ds[i][0][:T + 1] for i in range(cfg["batch_size"])]).cuda().contiguous()
    recon = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).to("cuda")
    kw = dict(warmup=False, with_loss=True, return_alpha_masks=False, beta_kl=cfg["beta_kl"], beta_dyn=cfg["beta_dyn"],
              beta_rec=cfg["beta_rec"], kl_balance=cfg["kl_balance"], dynamic_discount=None,
              recon_loss_type=cfg["recon_loss_type"], recon_loss_func=recon, beta_dyn_rec=cfg["beta_dyn_rec"],
              beta_obj=cfg.get("beta_obj", 0.0), done_mask=None, x_goal=None)
    reps = []
    for _ in range(args.reps):
        model.load_state_dict(init)
        r = {"infer": {}, "grads": {}}
        model.eval()
        with torch.no_grad():
            seed_all()
            mo = model(x, deterministic=True, with_loss=False)
            for k in INFER_KEYS:
                if k in mo and torch.is_tensor(mo[k]):
                    r["infer"][k] = mo[k].float().cpu()
        model.train()
        model.zero_grad(set_to_none=True)
        seed_all()
        loss = model(x, **kw)["loss_dict"]["loss"]
        loss.backward()
        r["loss"] = loss.detach().double().cpu()
        for n, prm in model.named_parameters():
            if prm.grad is not None:
                r["grads"][n] = prm.grad.detach().float().cpu()
        reps.append(r)
        print(f"rep {len(reps)}: loss {float(loss):.6f}", flush=True)
    torch.save({"repo": repo, "init_sha256": h.hexdigest(), "batch_shape": list(x.shape), "reps": reps,
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}, args.out)
    print(f"init sha256 {h.hexdigest()[:16]} | wrote {args.out}")


def compare(args):
    import torch
    a, b = torch.load(args.old, weights_only=False), torch.load(args.new, weights_only=False)

    def tensors(d):
        out = []
        for i, r in enumerate(d["reps"]):
            t = {f"infer/{k}": v for k, v in r["infer"].items()}
            t.update({f"grad/{k}": v for k, v in r["grads"].items()})
            t["loss"] = r["loss"]
            out.append(t)
        return out

    ta, tb = tensors(a), tensors(b)

    def maxdiff(u, v):
        return 0.0 if torch.equal(u, v) else float((u.double() - v.double()).abs().max())

    def repeat(ts, k):
        return max((maxdiff(ts[0][k], t[k]) for t in ts[1:]), default=0.0)

    rows, verdict = {}, "bitwise_identical"
    keys = sorted(set(ta[0]) | set(tb[0]))
    for k in keys:
        if k not in ta[0] or k not in tb[0] or ta[0][k].shape != tb[0][k].shape:
            rows[k] = {"status": "missing or shape mismatch"}
            verdict = "DIFFERS"
            continue
        d = maxdiff(ta[0][k], tb[0][k])
        ra, rb = repeat(ta, k), repeat(tb, k)
        rows[k] = {"old_vs_new_max_abs": d, "old_repeat_max_abs": ra, "new_repeat_max_abs": rb}
        if d > 0:
            if d <= max(ra, rb):
                verdict = verdict if verdict == "DIFFERS" else "within_repeatability"
            else:
                verdict = "DIFFERS"
    same_init = a["init_sha256"] == b["init_sha256"]
    if not same_init:
        verdict = "DIFFERS"
    nz = {k: v for k, v in rows.items() if v.get("old_vs_new_max_abs", 1) != 0}
    res = {"verdict": verdict, "init_weights_identical": same_init, "n_tensors": len(rows),
           "n_tensors_bitwise_identical": len(rows) - len(nz), "loss_old": float(ta[0]["loss"]),
           "loss_new": float(tb[0]["loss"]), "non_identical": nz, "old": a["repo"], "new": b["repo"], "gpu": a["gpu"]}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"verdict: {verdict} | init weights identical: {same_init} | {res['n_tensors_bitwise_identical']}/{len(rows)} "
          f"tensors bitwise identical | loss old {res['loss_old']:.6f} new {res['loss_new']:.6f}")
    for k, v in sorted(nz.items(), key=lambda kv: -kv[1].get("old_vs_new_max_abs", 0))[:10]:
        print(f"  {k[:70]:70s} {v}")
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo-root", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--batch-size", type=int, default=None)
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    c.add_argument("--out", default="neutrality.json")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
