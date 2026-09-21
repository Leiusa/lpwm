#!/usr/bin/env python
"""
Quality-validation run for ONE seed: the three paths (reference / triton / fused) are trained for --epochs epochs
through train_dlp.py, one after the other inside ONE process and ONE Slurm allocation, from the same initial
weights, data order and sampling noise (all fixed by --seed). Nothing is chosen by outcome: the reported checkpoint
is the final one (after the last epoch), per-epoch curves come from the training logs, and the final checkpoints are
evaluated on the reference path over ALL validation images.

It reuses the committed functions of benchmarks/stn/train_dlp_compare.py (train_one, reconstruction_check) unchanged.

    python quality_run.py --repo-root <snapshot> --config <cfg> --seed 0 --epochs 8 \
        --order reference,triton,fused --lpips-head <vgg.pth> --out <dir>
"""

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--order", default="reference,triton,fused")
    ap.add_argument("--lpips-head", required=True)
    ap.add_argument("--recon-images", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.set = {}

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    import bench_e2e as be
    be._bootstrap(repo)
    import train_dlp_compare as C

    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "eval", "lpips"), exist_ok=True)
    shutil.copy(args.lpips_head, os.path.join(out, "eval", "lpips", "vgg.pth"))
    os.chdir(out)                                       # train_dlp.py writes its run directories to the cwd
    base_cfg = json.load(open(args.config))
    order = args.order.split(",")
    header = {"kind": "quality-validation run for one seed (paths trained in one allocation from identical initial state)",
              "seed": args.seed, "epochs": args.epochs, "order": order, "host": socket.gethostname(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "env": be.env_record(), "repo": be.repo_record(repo), "config": args.config, "config_sha256": be.file_sha256(args.config),
              "config_content": base_cfg, "script_sha256": be.file_sha256(os.path.abspath(__file__)), "command": sys.argv,
              "dataset": base_cfg["ds"], "recon_loss_type": base_cfg["recon_loss_type"], "synthetic_procedural_dataset": False}
    runs = []
    for path in order:
        t0 = time.time()
        rec = C.train_one(repo, base_cfg, path, "_p1", out, args)
        runs.append(rec)
        json.dump(runs, open(os.path.join(out, "train_runs.json"), "w"), indent=2)          # progress survives a later failure
        losses = [e["train_loss"] for e in rec["epochs"]] + list(rec["validation_loss"])
        if len(rec["epochs"]) != args.epochs or not all(math.isfinite(v) for v in losses):
            json.dump({**header, "runs": runs, "stopped": f"{path}: epochs={len(rec['epochs'])}, finite={all(math.isfinite(v) for v in losses)}"},
                      open(os.path.join(out, "train_report_STOPPED.json"), "w"), indent=2)
            print(f"STOPPED after {path}: missing epochs or non-finite loss", flush=True)
            return 4
        print(f"  {path} done in {time.time() - t0:.0f}s", flush=True)
    reconstruction = C.reconstruction_check(repo, base_cfg, runs, out, args)
    json.dump({**header, "runs": runs, "reconstruction": reconstruction}, open(os.path.join(out, "train_report.json"), "w"), indent=2)
    print("wrote", os.path.join(out, "train_report.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
