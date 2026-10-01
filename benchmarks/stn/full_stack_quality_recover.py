#!/usr/bin/env python
"""
Recovery evaluation for a full_stack_quality_run.py seed whose training finished (both checkpoints saved) but whose
evaluation crashed before writing train_report.json (root cause: a pre-existing non-contiguous z_depth in
deterministic/eval mode, fixed in modules.py's _decode_objects_fused; see docs/dlp_full_stack_report.md). Does NOT
retrain: reads the already-saved runs/checkpoints from --crashed-out (its train_runs.json + saves/*.pth) and redoes
reconstruction_check() and the deployment cross-check against the FIXED code in --repo-root, writing a fresh
train_report.json into --out.

    python full_stack_quality_recover.py --repo-root <fixed snapshot> --crashed-out <old out dir> --out <new out dir>
"""
import argparse
import json
import math
import os
import shutil
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--crashed-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recon-images", type=int, default=16)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    import bench_e2e as be
    be._bootstrap(repo)
    import torch
    import train_dlp_compare as C

    C.PATHS = dict(C.PATHS)
    C.PATHS["fused_cl"] = dict(stn_backend="triton", fused_composite=True, particle_dec_channels_last=True)

    def load_final_model_eval_path(run, out, path_override):
        from build_model import build
        cfg = json.load(open(run["cfg_path"]))
        cfg.update(C.PATHS[path_override])
        tmp = os.path.join(out, f"recon_cfg_{run['path']}{run['tag']}_evalas_{path_override}.json")
        json.dump(cfg, open(tmp, "w"))
        model, _, _ = build(tmp, "cuda")
        ckpt = os.path.join(run["log_dir"], "saves", f"{cfg['ds']}_gdlp{cfg['run_prefix']}.pth")
        model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False))
        return model
    C.load_final_model = lambda run, out: load_final_model_eval_path(run, out, "reference")

    crashed = os.path.abspath(args.crashed_out)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "eval", "lpips"), exist_ok=True)
    shutil.copy(os.path.join(crashed, "eval", "lpips", "vgg.pth"), os.path.join(out, "eval", "lpips", "vgg.pth"))
    runs = json.load(open(os.path.join(crashed, "train_runs.json")))
    for r in runs:  # cfg_path / log_dir were absolute paths into the crashed run's own directory; point at fixed-code cfgs but keep the log_dir/checkpoint as-is
        r["cfg_path"] = os.path.join(crashed, os.path.basename(r["cfg_path"]))
    base_cfg = json.load(open(next(r["cfg_path"] for r in runs if r["path"] == "reference")))
    args.set = {}
    os.chdir(out)

    header = {"kind": "RECOVERY evaluation (no retraining): reuses checkpoints trained by a crashed full_stack_quality_run.py, "
                      "re-evaluates against the FIXED code (non-contiguous z_depth in deterministic mode fixed in "
                      "_decode_objects_fused). Training itself was unaffected by the bug and is unchanged from the original run.",
              "crashed_out": crashed, "seed": runs[0].get("cfg_path"), "host": be.env_record().get("hostname"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "env": be.env_record(), "repo": be.repo_record(repo), "script_sha256": be.file_sha256(os.path.abspath(__file__)),
              "dataset": base_cfg["ds"], "recon_loss_type": base_cfg["recon_loss_type"]}

    reconstruction = C.reconstruction_check(repo, base_cfg, runs, out, args)
    a_run = next(r for r in runs if r["path"] == "reference")
    d_run_for_weights = next(r for r in runs if r["path"] == "fused_cl")
    m_a = load_final_model_eval_path(a_run, out, "reference")
    m_d = load_final_model_eval_path(d_run_for_weights, out, "reference")
    w_a = torch.cat([p.detach().flatten().double().cpu() for p in m_a.parameters()])
    w_d = torch.cat([p.detach().flatten().double().cpu() for p in m_d.parameters()])
    reconstruction["weight_relative_l2_between_final_models"] = {"fused_cl_vs_reference": float(torch.linalg.vector_norm(w_d - w_a) / torch.linalg.vector_norm(w_a))}
    del m_a, m_d
    torch.cuda.empty_cache()
    json.dump({**header, "runs": runs, "reconstruction": reconstruction}, open(os.path.join(out, "train_report.json"), "w"), indent=2)
    print("wrote (core, pre-deployment-check)", os.path.join(out, "train_report.json"), flush=True)

    deployment_check = {"kind": "the D-trained checkpoint evaluated through its own real inference path (stn_backend=triton, "
                                "fused_composite=True, particle_dec_channels_last=True) instead of the standard reference-path "
                                "evaluation used for the main comparison; this is the exact call that crashed before the fix"}
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
    except Exception as exc:  # noqa: BLE001
        deployment_check.update(failed=True, error=f"{type(exc).__name__}: {exc}")
        print(f"deployment cross-check FAILED again (core result unaffected): {type(exc).__name__}: {exc}", flush=True)

    json.dump({**header, "runs": runs, "reconstruction": reconstruction, "deployment_check": deployment_check}, open(os.path.join(out, "train_report.json"), "w"), indent=2)
    print("wrote", os.path.join(out, "train_report.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
