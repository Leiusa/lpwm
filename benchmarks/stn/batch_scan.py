#!/usr/bin/env python
"""
One (path, batch size) short performance measurement through the real BAIR static-DLP training flow.

It reproduces what train_dlp.py does per step -- real BAIR frames from the BAIRImage DataLoader (seeded shuffling,
4 workers, pinned memory, drop_last), the original vgg loss (LossLPIPS), Adam with the config's settings, the
same forward arguments, model_output kept alive until the step ends -- for ONE path and ONE batch size, in its own
process, so an out-of-memory in one configuration cannot disturb another.

Paths: reference (stn_backend=reference), triton (stn_backend=triton), fused (triton + fused_composite=True).
Training stage: post-warmup steady state (warmup=False), return_alpha_masks=False for the timed steps; one extra
step with return_alpha_masks=True is measured afterwards because train_dlp requests the masks on the last batch of
every plotting epoch.

Excluded from timing: the first WARMUP steps (compilation, cudnn/allocator warm-up, dataloader start-up). The same
number of steps is timed for every configuration. An OOM is recorded once (stage, allocator peak) and the process ends
normally; nothing is changed to make it pass.
"""

import argparse
import json
import os
import random
import socket
import statistics
import sys
import tempfile
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", choices=("reference", "triton", "fused"), required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--timed-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo_root, "benchmarks", "stn"))
    import bench_e2e as be
    be._bootstrap(repo_root)
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from utils.loss_functions import LossLPIPS

    torch.backends.cudnn.benchmark = False       # as train_dlp.py
    torch.backends.cudnn.deterministic = True
    B, seed, path = args.batch_size, args.seed, args.path
    backend = "reference" if path == "reference" else "triton"
    cfg = json.load(open(args.config))
    cfg.update(batch_size=B, stn_backend=backend, fused_composite=(path == "fused"), seed=seed)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
    result = {"kind": "short performance measurement through the real BAIR static-DLP training flow; not a convergence run",
              "path": path, "batch_size": B, "warmup_steps": args.warmup_steps, "timed_steps": args.timed_steps,
              "training_stage": "post-warmup steady state (warmup=False)", "alpha_masks_timed_steps": False,
              "gpu": torch.cuda.get_device_name(0), "gpu_total_mb": total_mem, "host": socket.gethostname(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "seed": seed, "config_sha256": be.file_sha256(args.config), "script_sha256": be.file_sha256(os.path.abspath(__file__)),
              "repo": be.repo_record(repo_root), "env": be.env_record(), "status": "started", "stage": "setup"}

    def save():
        json.dump(result, open(args.out, "w"), indent=2)

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        lpwm_stn.set_backend(backend)
        dataset = get_image_dataset(cfg["ds"], cfg["root"], mode="train", image_size=cfg["image_size"])
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=B, num_workers=4, pin_memory=True, drop_last=True,
                                             generator=torch.Generator().manual_seed(seed))
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "cfg.json")
            json.dump(cfg, open(cfg_path, "w"))
            model, _, _ = build(cfg_path, "cuda")
        result["initial_weights_sha256"] = be.tensors_sha256(model.parameters())
        recon_loss_func = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).to("cuda")
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["adam_betas"], eps=cfg["adam_eps"],
                               weight_decay=cfg["weight_decay"])
        kw = dict(warmup=False, with_loss=True, beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"], kl_balance=cfg["kl_balance"],
                  recon_loss_type=cfg["recon_loss_type"], recon_loss_func=recon_loss_func, beta_obj=cfg.get("beta_obj", 0.0))

        def step(x, masks):
            model.train()
            model_output = model(x, return_alpha_masks=masks, **kw)       # alive until the step ends, as in train_dlp.py
            loss = model_output["loss_dict"]["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            return float(loss)

        it = iter(loader)
        records, first_loss, first_batch_sha = [], None, None
        for i in range(args.warmup_steps + args.timed_steps):
            result["stage"] = "warmup" if i < args.warmup_steps else "timed"
            if i == args.warmup_steps:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            x = next(it)[0].to("cuda")
            t_fetch = time.perf_counter() - t0
            if i == 0:
                first_batch_sha = be.tensors_sha256([x])
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            loss = step(x, False)
            e1.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i == 0:
                first_loss = loss
            if i >= args.warmup_steps:
                records.append({"wall_ms": (t1 - t0) * 1e3, "event_ms": e0.elapsed_time(e1), "fetch_ms": t_fetch * 1e3})
        med = lambda k: statistics.median(r[k] for r in records)  # noqa: E731
        result.update(stage="masks_step", first_step_loss=first_loss, first_batch_sha256=first_batch_sha,
                      steady_peak_allocated_mb=torch.cuda.max_memory_allocated() / 1e6,
                      steady_peak_reserved_mb=torch.cuda.max_memory_reserved() / 1e6,
                      wall_ms_median=med("wall_ms"), wall_ms_mean=statistics.fmean(r["wall_ms"] for r in records),
                      event_ms_median=med("event_ms"), fetch_ms_median=med("fetch_ms"),
                      wall_ms_min=min(r["wall_ms"] for r in records), wall_ms_max=max(r["wall_ms"] for r in records))
        result["throughput_images_per_s"] = B / (result["wall_ms_median"] / 1e3)
        result["ms_per_image_wall"] = result["wall_ms_median"] / B
        torch.cuda.reset_peak_memory_stats()
        x = next(it)[0].to("cuda")
        step(x, True)                                                     # the plotting batch: masks requested
        torch.cuda.synchronize()
        result.update(masks_step_peak_allocated_mb=torch.cuda.max_memory_allocated() / 1e6, stage="done")
        calls = model.decoder_module.fused_composite_calls
        steps_done = args.warmup_steps + args.timed_steps + 1
        result["path_check"] = {"backend": lpwm_stn.get_backend_name(), "fused_composite_calls": calls, "forward_passes": steps_done,
                                "ok": lpwm_stn.get_backend_name() == backend and calls == (steps_done if path == "fused" else 0)}
        result["status"] = "ok" if result["path_check"]["ok"] else "PATH_CHECK_FAILED"
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="OOM", oom_stage=result["stage"], oom_message=str(exc).split("\n")[0][:300],
                      allocator_peak_allocated_mb=torch.cuda.max_memory_allocated() / 1e6,
                      allocator_peak_reserved_mb=torch.cuda.max_memory_reserved() / 1e6)
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised so the driver sees a failure
        result.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    save()
    print(json.dumps({k: result.get(k) for k in ("path", "batch_size", "status", "wall_ms_median", "event_ms_median",
                                                  "throughput_images_per_s", "steady_peak_allocated_mb", "masks_step_peak_allocated_mb",
                                                  "oom_stage")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
