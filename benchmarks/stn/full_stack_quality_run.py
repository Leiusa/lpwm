#!/usr/bin/env python
"""
Quality validation of the FULL integrated optimization stack for ONE seed: reference (A) and the complete optimized
path (D = triton crop/paste + composite fusion + particle_dec channels_last) are trained for --epochs epochs through
train_dlp.py, one after the other inside ONE process and ONE Slurm allocation, from identical initial weights, data
order and sampling noise (all fixed by --seed). This is the DIRECT A-vs-D comparison; it does not retrain B or C
(their incremental contribution was already validated separately: fused vs triton and fused_cl vs fused).

Reuses benchmarks/stn/train_dlp_compare.py's train_one() and reconstruction_check() unchanged. Final checkpoints are
evaluated two ways:
  1. Standard: both checkpoints (A-trained and D-trained) run through the REFERENCE evaluation path, over the whole
     validation set -- this isolates what TRAINING on each path did to the model, independent of how it is later run.
  2. Deployment cross-check: the D-trained checkpoint is ALSO evaluated by actually running the D path itself
     (stn_backend=triton, fused_composite=True, particle_dec_channels_last=True) and compared to its own
     reference-path evaluation, to confirm the deployed inference path does not introduce a further detectable
     difference beyond what training already produced.

    python full_stack_quality_run.py --repo-root <snapshot> --config <cfg> --seed 0 --epochs 8 \
        --order reference,fused_cl --lpips-head <vgg.pth> --out <dir>
"""
import argparse
import json
import math
import os
import re
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
    ap.add_argument("--order", default="reference,fused_cl")
    ap.add_argument("--lpips-head", required=True)
    ap.add_argument("--recon-images", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.set = {}

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    import bench_e2e as be
    be._bootstrap(repo)
    import torch
    import train_dlp_compare as C

    C.PATHS = dict(C.PATHS)
    C.PATHS["fused_cl"] = dict(stn_backend="triton", fused_composite=True, particle_dec_channels_last=True)

    def load_final_model_eval_path(run, out, path_override):
        """Same evaluation configuration for both checkpoints (reference path), OR the run's own path for the cross-check."""
        from build_model import build
        cfg = json.load(open(run["cfg_path"]))
        cfg.update(C.PATHS[path_override])
        tmp = os.path.join(out, f"recon_cfg_{run['path']}{run['tag']}_evalas_{path_override}.json")
        json.dump(cfg, open(tmp, "w"))
        model, _, _ = build(tmp, "cuda")
        ckpt = os.path.join(run["log_dir"], "saves", f"{cfg['ds']}_gdlp{cfg['run_prefix']}.pth")
        model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False))
        return model
    C.load_final_model = lambda run, out: load_final_model_eval_path(run, out, "reference")  # standard: all checkpoints via the reference path

    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "eval", "lpips"), exist_ok=True)
    shutil.copy(args.lpips_head, os.path.join(out, "eval", "lpips", "vgg.pth"))
    os.chdir(out)
    base_cfg = json.load(open(args.config))
    order = args.order.split(",")
    header = {"kind": "quality validation of the FULL integrated optimization stack: reference (A) vs triton+fusion+channels_last (D), "
                      "one seed, one allocation, identical initial state; does not retrain the intermediate B/C paths",
              "seed": args.seed, "epochs": args.epochs, "order": order, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "env": be.env_record(), "repo": be.repo_record(repo), "config": args.config,
              "config_sha256": be.file_sha256(args.config), "config_content": base_cfg, "script_sha256": be.file_sha256(os.path.abspath(__file__)), "command": sys.argv,
              "dataset": base_cfg["ds"], "recon_loss_type": base_cfg["recon_loss_type"], "synthetic_procedural_dataset": False}
    runs = []
    for path in order:
        t0 = time.time()
        rec = C.train_one(repo, base_cfg, path, "_p1", out, args)
        text = open(os.path.join(rec["log_dir"], "log.txt")).read()
        m1, m2 = re.search(r"particle_dec_channels_last: (\S+)", text), re.search(r"particle_dec_channels_last_calls=(\d+)", text)
        rec["layout_startup_flag"] = m1.group(1) if m1 else None
        rec["layout_first_step_calls"] = int(m2.group(1)) if m2 else None
        runs.append(rec)
        json.dump(runs, open(os.path.join(out, "train_runs.json"), "w"), indent=2)
        losses = [e["train_loss"] for e in rec["epochs"]] + list(rec["validation_loss"])
        if len(rec["epochs"]) != args.epochs or not all(math.isfinite(v) for v in losses):
            json.dump({**header, "runs": runs, "stopped": f"{path}: epochs={len(rec['epochs'])}"}, open(os.path.join(out, "train_report_STOPPED.json"), "w"), indent=2)
            print(f"STOPPED after {path}", flush=True)
            return 4
        print(f"  {path} done in {time.time() - t0:.0f}s", flush=True)

    reconstruction = C.reconstruction_check(repo, base_cfg, runs, out, args)   # both checkpoints via the reference path
    # reconstruction_check()'s own weight-diff pairs don't include fused_cl; compute A-vs-D directly (both loaded via the
    # reference-path build -- architecture/state_dict shapes are unaffected by particle_dec_channels_last, only strides)
    a_run = next(r for r in runs if r["path"] == "reference")
    d_run_for_weights = next(r for r in runs if r["path"] == "fused_cl")
    m_a = load_final_model_eval_path(a_run, out, "reference")
    m_d = load_final_model_eval_path(d_run_for_weights, out, "reference")
    w_a = torch.cat([p.detach().flatten().double().cpu() for p in m_a.parameters()])
    w_d = torch.cat([p.detach().flatten().double().cpu() for p in m_d.parameters()])
    reconstruction["weight_relative_l2_between_final_models"] = {"fused_cl_vs_reference": float(torch.linalg.vector_norm(w_d - w_a) / torch.linalg.vector_norm(w_a))}
    del m_a, m_d
    torch.cuda.empty_cache()
    # persist the core A-vs-D result NOW: it is expensive (the full training) and must survive even if the
    # supplementary deployment cross-check below fails for an unrelated reason
    json.dump({**header, "runs": runs, "reconstruction": reconstruction}, open(os.path.join(out, "train_report.json"), "w"), indent=2)
    print("wrote (core, pre-deployment-check)", os.path.join(out, "train_report.json"), flush=True)

    # deployment cross-check: the D (fused_cl) checkpoint, evaluated through its own real path. Supplementary --
    # any failure here is recorded, not fatal, and never discards the reconstruction_check result written above.
    deployment_check = {"kind": "the D-trained checkpoint evaluated through its own real inference path (stn_backend=triton, "
                                "fused_composite=True, particle_dec_channels_last=True) instead of the standard reference-path "
                                "evaluation used for the main comparison"}
    try:
        d_run = next(r for r in runs if r["path"] == "fused_cl")
        from datasets.get_dataset import get_image_dataset
        ds_valid_full = get_image_dataset(base_cfg["ds"], base_cfg["root"], mode="valid", image_size=base_cfg["image_size"])
        x4, idx = C.fixed_validation_batch(ds_valid_full, args.recon_images)
        x4 = x4.cuda()
        import lpwm_stn
        model_d_native = load_final_model_eval_path(d_run, out, "fused_cl")
        lpwm_stn.set_backend("triton")
        rec_native = C.reconstruct(model_d_native, x4)
        mse_native = float(((rec_native - x4.clamp(0, 1)) ** 2).mean())
        full_native = C.full_validation_metrics(model_d_native, ds_valid_full)
        fused_calls_seen = model_d_native.decoder_module.fused_composite_calls
        cl_calls_seen = getattr(model_d_native.decoder_module.particle_dec, "channels_last_calls", None)
        lpwm_stn.set_backend("reference")
        deployment_check.update(
            fixed_batch={"mse": mse_native, "psnr_db": float(-10 * math.log10(mse_native)), "finite": bool(torch.isfinite(rec_native).all())},
            full_validation_set=full_native, fused_composite_calls_during_eval=fused_calls_seen, channels_last_calls_during_eval=cl_calls_seen,
            reference_path_result_for_comparison=reconstruction["runs"][d_run["path"] + d_run["tag"]], failed=False)
        del model_d_native
        torch.cuda.empty_cache()
        deployment_check["diff_native_minus_reference_path"] = {
            "fixed_batch_mse_pct": 100 * (deployment_check["fixed_batch"]["mse"] - deployment_check["reference_path_result_for_comparison"]["fixed_batch"]["mse"]) / deployment_check["reference_path_result_for_comparison"]["fixed_batch"]["mse"],
            "full_val_mse_pct": 100 * (deployment_check["full_validation_set"]["mse"] - deployment_check["reference_path_result_for_comparison"]["full_validation_set"]["mse"]) / deployment_check["reference_path_result_for_comparison"]["full_validation_set"]["mse"],
            "full_val_psnr_db": deployment_check["full_validation_set"]["psnr_db"] - deployment_check["reference_path_result_for_comparison"]["full_validation_set"]["psnr_db"]}
        print("deployment cross-check (D native path vs D-checkpoint-via-reference-path):", json.dumps(deployment_check["diff_native_minus_reference_path"]), flush=True)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal: the core A-vs-D result above already survived
        deployment_check.update(failed=True, error=f"{type(exc).__name__}: {exc}")
        print(f"deployment cross-check FAILED (core result unaffected): {type(exc).__name__}: {exc}", flush=True)

    json.dump({**header, "runs": runs, "reconstruction": reconstruction, "deployment_check": deployment_check}, open(os.path.join(out, "train_report.json"), "w"), indent=2)
    print("wrote", os.path.join(out, "train_report.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
