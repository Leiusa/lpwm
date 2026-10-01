#!/usr/bin/env python
"""
Milestone-1 controlled study, ONE seed: train A and D for --epochs epochs each, one after the other in this one process on
this one verified RTX A6000, from identical initial weights / data order / RNG (all fixed by --seed), then run the four
agreed evaluations (m1_eval.py, separate process, same GPU) on the saved epoch-8 checkpoints.

  A: stn_backend=reference, fused_composite=False, particle_dec_channels_last=False   (explicitly written to the config)
  D: stn_backend=triton,    fused_composite=True,  particle_dec_channels_last=True

Before any work: GPU must be exactly "NVIDIA RTX A6000" on a node other than grogu-4-13 (else exit). The agreed environment
is set explicitly and asserted (fp32, no AMP, cudnn benchmark False / deterministic True, cuDNN TF32 allowed, matmul TF32 off).
Runtime evidence (STN implementation functions that actually ran, composite_fused calls, model counters) is recorded per
training run and checked against what the run was supposed to be. Nothing here changes a kernel, threshold or model file.

    python m1_train_pair.py --repo-root R --config CFG --seed S --order reference,fused_cl --lpips-head vgg.pth --out DIR [--smoke]
"""
import argparse
import contextlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

PATHS = {"reference": dict(stn_backend="reference", fused_composite=False, particle_dec_channels_last=False),
         "fused_cl": dict(stn_backend="triton", fused_composite=True, particle_dec_channels_last=True)}


def training_evidence_ok(path, ev_snapshot, mc, backend_after):
    tri = path == "fused_cl"
    needle = "triton" if tri else "reference"
    keys = lambda op: [k for k in ev_snapshot if k.startswith(op + "|")]  # noqa: E731
    crop, paste = keys("stn_crop"), keys("stn_paste")
    fused_wrapper = ev_snapshot.get("composite_fused|call", 0)
    checks = {
        "stn_crop_ran_and_only_expected_impl": bool(crop) and all(needle in k for k in crop),
        "stn_paste_absent_when_fused_else_expected_impl": (not paste) if tri else (bool(paste) and all(needle in k for k in paste)),
        "composite_fused_calls_match_model_counter": fused_wrapper == mc["fused_composite_calls"],
        "fusion_ran_iff_D": (fused_wrapper > 0) if tri else (fused_wrapper == 0),
        "channels_last_ran_iff_D": ((mc["channels_last_calls"] or 0) > 0) if tri else ((mc["channels_last_calls"] or 0) == 0),
        "channels_last_calls_equal_fused_calls_for_D": (mc["channels_last_calls"] == mc["fused_composite_calls"]) if tri else True,
        "flags_as_requested": mc["fused_composite_flag"] == tri and bool(mc["particle_dec_channels_last_flag"]) == tri,
        "backend_name_after": backend_after == ("triton" if tri else "reference"),
    }
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--order", default="reference,fused_cl")
    ap.add_argument("--lpips-head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny-data smoke run: root override, evaluation expects the smoke validation size")
    ap.add_argument("--root-override", default=None)
    ap.add_argument("--expect-val-images", type=int, default=1200)
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo_root)
    here = os.path.join(repo, "benchmarks", "stn")
    sys.path.insert(0, here)
    import bench_e2e as be
    be._bootstrap(repo)
    import torch
    import lpwm_stn
    import m1_common as MC
    import train_dlp_compare as C            # reused unchanged: parse_log

    t_start = time.time()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "eval", "lpips"), exist_ok=True)
    shutil.copy(args.lpips_head, os.path.join(out, "eval", "lpips", "vgg.pth"))
    hw = MC.verify_hardware()                                   # raises HardwareMismatch -> non-zero exit before any work
    env0 = MC.set_agreed_environment()
    MC.assert_agreed_environment(env0)
    ev = MC.Evidence()
    MC.install_evidence(ev)
    base_cfg = json.load(open(args.config))
    if args.root_override:
        base_cfg["root"] = args.root_override
    order = args.order.split(",")
    assert sorted(order) == sorted(PATHS), order
    os.chdir(out)                                                # train_dlp writes run directories to the cwd
    T = importlib.import_module("train_dlp")                     # sets cudnn flags at import; re-assert afterwards
    env_after_import = MC.effective_settings()
    MC.assert_agreed_environment(env_after_import)

    report = {"kind": "m1 controlled pair: A vs D, one seed, one verified A6000, one process", "seed": args.seed, "epochs": args.epochs, "order": order,
              "smoke": args.smoke, "hardware": hw, "settings_set": env0, "settings_after_train_dlp_import": env_after_import, "code_hashes": MC.code_hashes(repo),
              "config": args.config, "config_sha256": MC.sha256_file(args.config), "command": sys.argv, "gpu_state_start": MC.gpu_state(), "runs": []}

    def save():
        json.dump(report, open(os.path.join(out, "train_report.json"), "w"), indent=2, default=str)

    save()
    for path in order:
        cfg = dict(base_cfg)
        cfg.update(PATHS[path])
        cfg.update(seed=args.seed, num_epochs=args.epochs, log_step_timing=True, eval_im_metrics=False, run_prefix=f"_{path}_m1")
        cfg_path = os.path.join(out, f"cfg_{path}_m1.json")
        json.dump(cfg, open(cfg_path, "w"), indent=1)
        ev.reset()
        before = set(os.listdir(out))
        gs0 = MC.gpu_state()
        t0 = time.perf_counter()
        with open(os.path.join(out, f"stdout_{path}_m1.log"), "w") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            model = T.train_dlp(cfg_path)
        wall = time.perf_counter() - t0
        gs1 = MC.gpu_state()
        new = sorted(d for d in set(os.listdir(out)) - before if d.endswith(f"_gdlp_{path}_m1"))
        log_dir = os.path.join(out, new[-1])
        rec = C.parse_log(log_dir)
        text = open(os.path.join(log_dir, "log.txt")).read()
        m2 = re.search(r"particle_dec_channels_last_calls=(\d+)", text)
        mc = MC.module_counters(model)
        snap = ev.snapshot()
        checks = training_evidence_ok(path, snap, mc, lpwm_stn.get_backend_name())
        ck = os.path.join(log_dir, "saves", f"bair_gdlp_{path}_m1.pth")
        env_end = MC.effective_settings()
        rec.update(path=path, cfg_path=cfg_path, log_dir=log_dir, total_wall_s=wall, first_step_layout_calls=int(m2.group(1)) if m2 else None,
                   runtime_evidence={"stn_and_fusion_call_counts": snap, "model_counters": mc, "backend_name_after": lpwm_stn.get_backend_name(), "checks": checks,
                                     "evidence_ok": bool(all(checks.values()))},
                   final_checkpoint=ck, final_checkpoint_exists=os.path.exists(ck), final_checkpoint_sha256=MC.sha256_file(ck) if os.path.exists(ck) else None,
                   gpu_state_before=gs0, gpu_state_after=gs1, settings_at_end=env_end, settings_unchanged=(env_end == env_after_import or {k: env_end[k] for k in MC.AGREED} == {k: env_after_import[k] for k in MC.AGREED}),
                   epochs_completed=len(rec["epochs"]), validation_lines=len(rec.get("validation_loss", [])))
        report["runs"].append(rec)
        save()
        print(f"{path}: {len(rec['epochs'])} epochs in {wall:.0f}s | evidence_ok={rec['runtime_evidence']['evidence_ok']} {[k for k, v in checks.items() if not v]} | "
              f"init={rec.get('startup', {}).get('init_weights_sha256')} first_batch={rec.get('first_step', {}).get('first_batch_sha256')}", flush=True)
        del model
        torch.cuda.empty_cache()
        if len(rec["epochs"]) != args.epochs or not rec["final_checkpoint_exists"]:
            print("STOP: incomplete training run", flush=True)
            report["stopped"] = f"{path}: epochs={len(rec['epochs'])}"
            save()
            return 4
        if not rec["runtime_evidence"]["evidence_ok"]:
            print("STOP: runtime evidence does not match the requested path", flush=True)
            report["stopped"] = f"{path}: evidence mismatch"
            save()
            return 6
    runs = {r["path"]: r for r in report["runs"]}
    a, d = runs["reference"], runs["fused_cl"]
    report["pairing"] = {"init_weights_identical": a["startup"]["init_weights_sha256"] == d["startup"]["init_weights_sha256"],
                         "first_batch_identical": a["first_step"]["first_batch_sha256"] == d["first_step"]["first_batch_sha256"]}
    report["training_wall_s"] = time.time() - t_start
    save()
    cmd = [sys.executable, os.path.join(here, "m1_eval.py"), "--repo-root", repo, "--config", args.config, "--ckpt-a", a["final_checkpoint"], "--ckpt-d", d["final_checkpoint"],
           "--out", os.path.join(out, "eval_m1.json"), "--expect-val-images", str(args.expect_val_images)]
    if args.root_override:
        cmd += ["--root-override", args.root_override]
    print("evaluation:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, stdout=open(os.path.join(out, "eval_m1.log"), "w"), stderr=subprocess.STDOUT).returncode
    report["evaluation_returncode"] = rc
    report["total_wall_s"] = time.time() - t_start
    save()
    print("evaluation returncode", rc, "| total wall s", round(report["total_wall_s"]), flush=True)
    return 0 if (rc == 0 and report["pairing"]["init_weights_identical"] and report["pairing"]["first_batch_identical"]) else 7


if __name__ == "__main__":
    sys.exit(main())
