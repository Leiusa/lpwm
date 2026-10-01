#!/usr/bin/env python
"""
Where does a training step of static DLP spend its wall time OUTSIDE the GPU math? A monitor at several places of
the real train_dlp.py step (same DataLoader settings, model, vgg loss, Adam, forward arguments), for small wins in
the input pipeline and the per-step bookkeeping. Measurement only: nothing in the repository changes.

Per step it records
  wait_ms    CPU blocked in next(loader)            -> the DataLoader cannot keep up
  h2d_ms     batch .to(cuda) + sync                 -> host-to-device copy
  gpu_ms     CUDA events around forward+backward+Adam -> the GPU math itself
  log_ms     the ~16 per-step .item() calls train_dlp.py makes (loss, psnr, kl terms, progress bar)
  wall_ms    the whole step, fetch to end
and, in a background thread, nvidia-smi GPU utilization / memory / power every 0.2 s, plus the CPU load average.

Variants (each with its own warm-up, then timed steps):
  workers=W, log=per_step   DataLoader num_workers W (train_dlp.py uses 4), the logging train_dlp.py does
  workers=4, log=none       same as train_dlp.py but without per-step .item() calls (what logging costs)
plus one-off: the start-up time of a fresh DataLoader iterator (train_dlp.py's workers are NOT persistent, so this
is paid at the start of every epoch), and a short torch.profiler run (top CUDA kernels / CPU ops, chrome trace).

Run from the folder that holds eval/lpips/vgg.pth, with TORCH_HOME set, e.g. on Lambda:
    cd ~/dlp-lpwm-optimization && python lpwm/benchmarks/stn/d_pipeline_probe.py --config cfg_D.json --out probe_d
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time


def gpu_sampler(stop, rows):
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits", "-lms", "200"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for line in p.stdout:
            if stop.is_set():
                break
            try:
                u, m, w = (float(v) for v in line.strip().split(","))
                rows.append((u, m, w))
            except ValueError:
                pass
    finally:
        p.terminate()


def summarize(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {"median": statistics.median(s), "mean": statistics.fmean(s), "p90": s[int(0.9 * (len(s) - 1))], "max": s[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--workers", default="4,8,16")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--timed-steps", type=int, default=40)
    ap.add_argument("--profile-steps", type=int, default=5)
    ap.add_argument("--out", default="probe_d")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    sys.path.insert(0, repo)
    import torch
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from utils.loss_functions import LossLPIPS

    torch.backends.cudnn.benchmark = False      # as train_dlp.py
    torch.backends.cudnn.deterministic = True
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(args.config))
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    lpwm_stn.set_backend(cfg.get("stn_backend") or "reference")
    model, _, _ = build(args.config, "cuda")
    model.train()
    recon_loss_func = LossLPIPS(normalized_rgb=cfg["normalize_rgb"]).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["adam_betas"], eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"])
    kw = dict(warmup=False, with_loss=True, beta_kl=cfg["beta_kl"], beta_rec=cfg["beta_rec"], kl_balance=cfg["kl_balance"],
              recon_loss_type=cfg["recon_loss_type"], recon_loss_func=recon_loss_func, beta_obj=cfg.get("beta_obj", 0.0),
              return_alpha_masks=False)
    dataset = get_image_dataset(cfg["ds"], cfg["root"], mode="train", image_size=cfg["image_size"])
    B = cfg["batch_size"]

    def make_loader(workers):
        return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=B, num_workers=workers, pin_memory=True,
                                           drop_last=True, generator=torch.Generator().manual_seed(seed))

    def log_like_train_dlp(out, loss):
        al = out["loss_dict"]
        vals = [al["psnr"], loss, al["loss_rec"], al["kl"], al["loss_kl_kp"], al["loss_kl_feat"], al["loss_kl_scale"],
                al["loss_kl_depth"], al["loss_kl_obj_on"], loss, al["loss_rec"], al["kl"], al["obj_on_l1"],
                out["obj_on_a"].mean(), out["obj_on_b"].mean(), torch.sigmoid(out["mu_scale"]).mean()]
        return [v.data.cpu().item() for v in vals]

    result = {"config": os.path.abspath(args.config), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "stn_backend": lpwm_stn.get_backend_name(), "fused_composite": bool(cfg.get("fused_composite", False)),
              "particle_dec_channels_last": bool(cfg.get("particle_dec_channels_last", False)), "batch_size": B,
              "cpu_count": os.cpu_count(), "train_dlp_num_workers": 4, "variants": {}}

    # one-off: start-up of a fresh iterator (paid at every epoch start in train_dlp.py, workers not persistent)
    startup = []
    for _ in range(3):
        t0 = time.perf_counter()
        it = iter(make_loader(4))
        next(it)
        startup.append((time.perf_counter() - t0) * 1e3)
        del it
    result["fresh_iterator_first_batch_ms_workers4"] = startup

    def run_variant(name, workers, per_step_log):
        it = iter(make_loader(workers))
        stop, smi = threading.Event(), []
        recs = []
        th = None
        for i in range(args.warmup_steps + args.timed_steps):
            timed = i >= args.warmup_steps
            if i == args.warmup_steps:
                torch.cuda.synchronize()
                th = threading.Thread(target=gpu_sampler, args=(stop, smi), daemon=True)
                th.start()
                load0 = os.getloadavg()[0]
            t0 = time.perf_counter()
            x = next(it)[0]
            t1 = time.perf_counter()
            x = x.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            out = model(x, **kw)
            loss = out["loss_dict"]["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            e1.record()
            t3 = time.perf_counter()          # CPU finished enqueueing the step
            torch.cuda.synchronize()          # train_dlp.py's first .item() waits here too
            t3b = time.perf_counter()
            if per_step_log:
                log_like_train_dlp(out, loss)
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            if timed:
                recs.append({"wait_ms": (t1 - t0) * 1e3, "h2d_ms": (t2 - t1) * 1e3, "gpu_ms": e0.elapsed_time(e1),
                             "cpu_enqueue_ms": (t3 - t2) * 1e3, "log_ms": (t4 - t3b) * 1e3, "wall_ms": (t4 - t0) * 1e3})
            del out, loss
        stop.set()
        load1 = os.getloadavg()[0]
        th.join(timeout=2)
        r = {k: summarize([rec[k] for rec in recs]) for k in recs[0]}
        wall = sum(rec["wall_ms"] for rec in recs)
        r["gpu_busy_fraction_of_wall"] = sum(rec["gpu_ms"] for rec in recs) / wall
        r["data_wait_fraction_of_wall"] = sum(rec["wait_ms"] for rec in recs) / wall
        r["images_per_s"] = B * len(recs) / (wall / 1e3)
        r["nvidia_smi_util_pct"] = summarize([s[0] for s in smi])
        r["nvidia_smi_mem_mb_max"] = max((s[1] for s in smi), default=None)
        r["nvidia_smi_power_w"] = summarize([s[2] for s in smi])
        r["cpu_loadavg_1min"] = [load0, load1]
        result["variants"][name] = r
        json.dump(result, open(os.path.join(args.out, "pipeline_probe.json"), "w"), indent=1)
        print(f"[{name}] wall {r['wall_ms']['median']:.1f} ms | gpu {r['gpu_ms']['median']:.1f} | wait {r['wait_ms']['median']:.2f} "
              f"(p90 {r['wait_ms']['p90']:.2f}) | h2d {r['h2d_ms']['median']:.2f} | log {r['log_ms']['median']:.2f} | "
              f"gpu busy {100 * r['gpu_busy_fraction_of_wall']:.1f}% | smi util {r['nvidia_smi_util_pct']['mean'] if r['nvidia_smi_util_pct'] else float('nan'):.0f}% | "
              f"{r['images_per_s']:.1f} img/s", flush=True)

    for w in (int(v) for v in args.workers.split(",")):
        run_variant(f"workers={w},log=per_step", w, True)
    run_variant("workers=4,log=none", 4, False)

    # short profile of the train_dlp.py-like step (workers=4, per-step logging)
    it = iter(make_loader(4))
    for _ in range(3):
        x = next(it)[0].to("cuda")
        out = model(x, **kw); loss = out["loss_dict"]["loss"]; opt.zero_grad(); loss.backward(); opt.step()
        log_like_train_dlp(out, loss)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(args.profile_steps):
            x = next(it)[0].to("cuda", non_blocking=True)
            out = model(x, **kw); loss = out["loss_dict"]["loss"]; opt.zero_grad(); loss.backward(); opt.step()
            log_like_train_dlp(out, loss)
        torch.cuda.synchronize()
    prof.export_chrome_trace(os.path.join(args.out, "trace.json"))
    ka = prof.key_averages()
    with open(os.path.join(args.out, "profile_top.txt"), "w") as fh:
        fh.write(f"{args.profile_steps} steps\n\n== top 30 by CUDA time ==\n")
        fh.write(ka.table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))
        fh.write("\n\n== top 30 by CPU time ==\n")
        fh.write(ka.table(sort_by="cpu_time_total", row_limit=30, max_name_column_width=70))
    print(f"[profile] wrote {args.out}/profile_top.txt and {args.out}/trace.json (open the trace at https://ui.perfetto.dev)", flush=True)
    print(f"[startup] fresh DataLoader iterator, first batch (workers=4): {[round(v) for v in startup]} ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
