#!/usr/bin/env python
"""
Milestone-1 final evaluation of one seed's saved epoch-8 checkpoints, in ONE consistent environment, with every switch set
explicitly and runtime evidence recorded for every evaluation. Evaluation only; nothing is trained.

  E1  A checkpoint : reference STN, fusion off, channels-last off
  E2  D checkpoint : triton STN,    fusion on,  channels-last on     (primary comparison is E2 - E1)
  E3  D checkpoint : reference STN, fusion off, channels-last on     (inference diagnostic)
  E4  D checkpoint : triton STN,    fusion on,  channels-last off    (inference diagnostic)

Model inference is deterministic (posterior means, model.eval(), no_grad); cuDNN is deterministic too (agreed environment).
Metrics over the full validation set: pooled MSE, pooled PSNR (= -10 log10 pooled MSE, the convention of the earlier
results), mean of per-image PSNR (supplementary), plus the fixed 16-image batch. The evidence check of every evaluation
compares the STN implementation functions that actually ran and the fusion / channels-last counters with what was requested.

    python m1_eval.py --repo-root R --config CFG --ckpt-a A.pth --ckpt-d D.pth --out eval.json [--expect-val-images 1200]
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

SPECS = (("E1", "A", "reference", False, False), ("E2", "D", "triton", True, True), ("E3", "D", "reference", False, True), ("E4", "D", "triton", True, False))


def expected_evidence(backend, fused, cl, n_forward):
    """What the counters must show for a completed evaluation of n_forward forward passes."""
    return {"stn_impl_needle": "triton" if backend == "triton" else "reference", "composite_fused_calls": n_forward if fused else 0,
            "model_fused_calls": n_forward if fused else 0, "channels_last_calls": n_forward if cl else 0, "stn_paste_calls_zero": fused}


def check_evidence(ev, mc, backend, fused, cl, n_forward, active_backend_after):
    exp = expected_evidence(backend, fused, cl, n_forward)
    tag = exp["stn_impl_needle"]
    checks = {
        "backend_name_after_run": active_backend_after == backend,
        "stn_crop_ran": ev.total("stn_crop|") > 0,
        "stn_crop_only_expected_impl": ev.only_impl("stn_crop", tag),
        "stn_paste_absent_when_fused": (ev.total("stn_paste|") == 0) if fused else (ev.only_impl("stn_paste", tag)),
        "composite_fused_wrapper_calls": ev.c.get("composite_fused|call", 0) == exp["composite_fused_calls"],
        "model_fused_counter": mc["fused_composite_calls"] == exp["model_fused_calls"],
        "channels_last_counter": (mc["channels_last_calls"] or 0) == exp["channels_last_calls"],
        "flags_as_requested": mc["fused_composite_flag"] == fused and bool(mc["particle_dec_channels_last_flag"]) == cl,
    }
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-val-images", type=int, default=1200)
    ap.add_argument("--root-override", default=None, help="dataset root override (smoke test only)")
    ap.add_argument("--recon-images", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    import bench_e2e as be
    be._bootstrap(repo)
    import numpy as np
    import torch
    import lpwm_stn
    import m1_common as MC
    import train_dlp_compare as C            # reused unchanged: fixed_validation_batch
    from build_model import build
    from datasets.get_dataset import get_image_dataset

    hw = MC.verify_hardware()
    env_before = MC.set_agreed_environment()
    MC.assert_agreed_environment(env_before)
    ev = MC.Evidence()
    MC.install_evidence(ev)

    base_cfg = json.load(open(args.config))
    if args.root_override:
        base_cfg["root"] = args.root_override
    ds = get_image_dataset(base_cfg["ds"], base_cfg["root"], mode="valid", image_size=base_cfg["image_size"])
    n_val = len(ds)
    if args.expect_val_images and n_val != args.expect_val_images:
        raise RuntimeError(f"validation set has {n_val} images, expected {args.expect_val_images}")
    x4, fixed_idx = C.fixed_validation_batch(ds, args.recon_images)
    x4 = x4.cuda()
    n_forward = 1 + math.ceil(n_val / args.batch_size)
    ckpts = {"A": args.ckpt_a, "D": args.ckpt_d}
    ck_sha = {k: MC.sha256_file(v) for k, v in ckpts.items()}
    result = {"kind": "m1 final evaluation (evaluation only)", "hardware": hw, "settings_set_before": env_before, "code_hashes": MC.code_hashes(repo),
              "config_sha256": MC.sha256_file(args.config), "checkpoints": {k: {"path": ckpts[k], "sha256": ck_sha[k]} for k in ckpts}, "validation_images": n_val,
              "fixed_batch_indices": fixed_idx, "expected_forward_passes_per_evaluation": n_forward, "evaluations": {}, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def save():
        json.dump(result, open(args.out, "w"), indent=2)

    def psnr(mse):
        return float(-10 * math.log10(mse)) if mse > 0 else float("inf")

    for name, which, backend, fused, cl in SPECS:
        cfg = dict(base_cfg)
        cfg.update(fused_composite=fused, particle_dec_channels_last=cl, stn_backend=backend, batch_size=args.batch_size)
        with tempfile.TemporaryDirectory() as tmp:
            cp = os.path.join(tmp, "cfg.json")
            json.dump(cfg, open(cp, "w"))
            model, _, _ = build(cp, "cuda")
        model.load_state_dict(torch.load(ckpts[which], map_location="cuda", weights_only=False), strict=True)
        model.eval()
        prev = lpwm_stn.set_backend(backend)
        assert lpwm_stn.get_backend_name() == backend
        env_now = MC.effective_settings()
        MC.assert_agreed_environment(env_now)
        ev.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            rec = model(x4.unsqueeze(1), deterministic=True, with_loss=False)["rec_rgb"].reshape(x4.shape).clamp(0, 1)
            fixed_mse = float(((rec - x4.clamp(0, 1)) ** 2).mean())
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
            sq, cnt, per_image_psnr = 0.0, 0, []
            for batch in loader:
                x = batch[0].cuda().reshape(-1, *batch[0].shape[-3:])
                r = model(x.unsqueeze(1), deterministic=True, with_loss=False)["rec_rgb"].reshape(x.shape).clamp(0, 1)
                se = (r - x.clamp(0, 1)) ** 2
                sq += float(se.sum())
                cnt += se.numel()
                per_image_mse = se.flatten(1).mean(1).double()
                per_image_psnr.extend((-10 * torch.log10(per_image_mse)).tolist())
        torch.cuda.synchronize()
        mse = sq / cnt
        mc = MC.module_counters(model)
        checks = check_evidence(ev, mc, backend, fused, cl, n_forward, lpwm_stn.get_backend_name())
        conv_cl = all(m.weight.is_contiguous(memory_format=torch.channels_last) for m in model.decoder_module.particle_dec.modules() if isinstance(m, torch.nn.Conv2d))
        conv_nchw = all(m.weight.is_contiguous() for m in model.decoder_module.particle_dec.modules() if isinstance(m, torch.nn.Conv2d))
        checks["conv_weight_layout_as_requested"] = conv_cl if cl else conv_nchw
        finite = bool(math.isfinite(mse) and all(math.isfinite(p) for p in per_image_psnr))
        result["evaluations"][name] = {
            "checkpoint": which, "requested": {"stn_backend": backend, "fusion": fused, "channels_last": cl, "model_deterministic": True},
            "full_validation": {"images": cnt // (3 * base_cfg["image_size"] ** 2), "mse": mse, "psnr_pooled_db": psnr(mse), "psnr_per_image_mean_db": float(np.mean(per_image_psnr)),
                                "psnr_per_image_std_db": float(np.std(per_image_psnr, ddof=1))},
            "fixed_batch": {"mse": fixed_mse, "psnr_db": psnr(fixed_mse)}, "finite": finite,
            "runtime_evidence": {"stn_and_fusion_call_counts": ev.snapshot(), "model_counters": mc, "active_backend_after": lpwm_stn.get_backend_name(), "backend_before": prev,
                                 "settings_during": env_now, "gpu_state": MC.gpu_state()},
            "evidence_checks": checks, "evidence_ok": bool(all(checks.values()) and finite), "eval_seconds": time.perf_counter() - t0}
        print(f"[{name}] {which}: backend={backend} fusion={fused} cl={cl} | pooled PSNR {psnr(mse):.4f} dB, MSE {mse:.6e}, per-image mean PSNR "
              f"{np.mean(per_image_psnr):.4f} | evidence_ok={result['evaluations'][name]['evidence_ok']} {[k for k, v in checks.items() if not v]}", flush=True)
        del model
        torch.cuda.empty_cache()
        save()
    result["all_evidence_ok"] = all(v["evidence_ok"] for v in result["evaluations"].values())
    result["settings_after"] = MC.effective_settings()
    save()
    print("wrote", args.out, "| all_evidence_ok =", result["all_evidence_ok"], flush=True)
    return 0 if result["all_evidence_ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
