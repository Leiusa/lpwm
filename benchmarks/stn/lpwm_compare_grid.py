#!/usr/bin/env python
"""
Visual side-by-side check of two LPWM checkpoints (e.g. trained with fp32 vs TF32 matmuls) on the same fixed
validation episodes. Evaluation only, deterministic inference (model.eval(), no_grad, deterministic=True).

  1. video prediction    exactly how train_lpwm.py's image metrics predict: condition on --cond-steps frames
                         (config cond_steps) and roll out to --horizon frames (config animation_horizon) with
                         model.sample_from_x(..., deterministic=True). Rows: ground truth, A, B; per-frame PSNR
                         against ground truth is printed under every predicted frame. One PNG per episode.
  2. decomposition       one forward pass over the first timestep_horizon+1 frames (deterministic posterior means):
                         GT | A, B reconstruction | foreground | background | mask | particles (obj_on > 0.5,
                         encoder particles, count shown) for the selected frames. One PNG per episode.

Each checkpoint is evaluated with its own matmul setting (--tf32-a / --tf32-b, default: A fp32, B TF32), i.e. the
way it was trained. Episodes are fixed (evenly spread over the validation set), so A and B see identical inputs.

    cd ~/dlp-lpwm-optimization && python lpwm/benchmarks/stn/lpwm_compare_grid.py --config cfg_lpwm_C_s1_fp32.json \
        --ckpt-a RUN_A/saves/bair_gddlp_lpwm_C_s1_fp32.pth --ckpt-b RUN_B/saves/bair_gddlp_lpwm_C_s1_tf32.pth \
        --labels fp32,tf32 --out cmp_s1
"""
import argparse
import json
import math
import os
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="LPWM config used for training (model shape, data root, flags)")
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--tf32-a", type=int, default=0, help="1 = evaluate A with TF32 matmul")
    ap.add_argument("--tf32-b", type=int, default=1, help="1 = evaluate B with TF32 matmul")
    ap.add_argument("--n-episodes", type=int, default=4)
    ap.add_argument("--cond-steps", type=int, default=None, help="default: config cond_steps")
    ap.add_argument("--horizon", type=int, default=None, help="default: config animation_horizon")
    ap.add_argument("--pred-frames", default="0,3,6,9,12,15,18,21,24,27,29", help="frames shown in the prediction grid")
    ap.add_argument("--decomp-frames", default="0,4,8,12,16", help="frames shown in the decomposition grid (< T+1)")
    ap.add_argument("--out", default="cmp")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    sys.path.insert(0, repo)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import lpwm_stn
    from build_model import build
    from datasets.get_dataset import get_video_dataset
    from utils.util_func import plot_keypoints_on_image

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(args.config))
    lab = args.labels.split(",")
    T = cfg["timestep_horizon"]
    cond = args.cond_steps if args.cond_steps is not None else cfg.get("cond_steps", 1)
    horizon = args.horizon if args.horizon is not None else cfg.get("animation_horizon", 30)
    lpwm_stn.set_backend(cfg.get("stn_backend") or "reference")

    ds = get_video_dataset(cfg["ds"], cfg["root"], seq_len=horizon, mode="val", image_size=cfg["image_size"])
    n = len(ds)
    ep_idx = [round(i * (n - 1) / max(args.n_episodes - 1, 1)) for i in range(args.n_episodes)]
    x = torch.stack([ds[i][0][:horizon] for i in ep_idx]).cuda()          # [E, horizon, 3, H, W]
    E = x.shape[0]

    def run(ckpt, tf32):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cfg.json")
            json.dump(cfg, open(p, "w"))
            model, _, _ = build(p, "cuda")
        model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False), strict=True)
        model.eval()
        with torch.no_grad():
            gen = model.sample_from_x(x, cond_steps=cond, num_steps=horizon - cond, deterministic=True)
            gen = gen.reshape(E, -1, *x.shape[-3:])[:, :horizon].clamp(0, 1).cpu()
            mo = model(x[:, :T + 1], deterministic=True, with_loss=False)
            Tm = T + 1
            shp = (E, Tm) + tuple(x.shape[-3:])
            dec = {"rec": mo["rec_rgb"].reshape(shp).clamp(0, 1).cpu(),
                   "fg": mo["dec_objects"].reshape(shp).clamp(0, 1).cpu(),
                   "bg": mo["bg_rgb"].reshape(shp).clamp(0, 1).cpu()}
            am = mo["alpha_masks"]
            dec["mask"] = am.reshape(E * Tm, -1, *x.shape[-2:]).sum(1).reshape(E, Tm, *x.shape[-2:]).clamp(0, 1).cpu()
            obj_on = mo["obj_on"].reshape(E, Tm, -1).cpu()
            kr = model.kp_range
            mu = mo["mu_tot"].reshape(E, Tm, obj_on.shape[-1], 2).clamp(kr[0], kr[1]).cpu()
            dec["obj_on"], dec["mu"], dec["kp_range"] = obj_on, mu, kr
        del model
        torch.cuda.empty_cache()
        return gen, dec

    gen_a, dec_a = run(args.ckpt_a, args.tf32_a)
    gen_b, dec_b = run(args.ckpt_b, args.tf32_b)
    gt = x.cpu()

    def psnr(a, b):
        mse = float(((a - b) ** 2).mean())
        return -10 * math.log10(mse) if mse > 0 else float("inf")

    def hwc(t):
        return t.clamp(0, 1).permute(1, 2, 0).numpy()

    pf = [f for f in (int(v) for v in args.pred_frames.split(",")) if f < horizon]
    df = [f for f in (int(v) for v in args.decomp_frames.split(",")) if f < T + 1]
    summary = {"labels": lab, "episodes": ep_idx, "cond_steps": cond, "horizon": horizon, "tf32": [args.tf32_a, args.tf32_b],
               "pred_psnr_mean_over_predicted_frames": {}, "recon_psnr_mean": {}, "active_particles_mean": {}}
    for e in range(E):
        # ---- prediction grid
        fig, axes = plt.subplots(3, len(pf), figsize=(1.6 * len(pf), 5.3), dpi=110)
        for c, f in enumerate(pf):
            for r, (name, seq) in enumerate((("GT", gt), (lab[0], gen_a), (lab[1], gen_b))):
                ax = axes[r, c]
                ax.imshow(hwc(seq[e, f]))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"t={f}" + (" (cond)" if f < cond else ""), fontsize=8)
                if c == 0:
                    ax.set_ylabel(name, fontsize=9)
                if r > 0 and f >= cond:
                    ax.text(0.5, -0.1, f"{psnr(seq[e, f], gt[e, f]):.1f} dB", transform=ax.transAxes, ha="center",
                            va="top", fontsize=7)
        pa = sum(psnr(gen_a[e, f], gt[e, f]) for f in range(cond, horizon)) / (horizon - cond)
        pb = sum(psnr(gen_b[e, f], gt[e, f]) for f in range(cond, horizon)) / (horizon - cond)
        summary["pred_psnr_mean_over_predicted_frames"][ep_idx[e]] = {lab[0]: pa, lab[1]: pb}
        fig.suptitle(f"video prediction, val episode {ep_idx[e]} | cond {cond} frame(s), deterministic | mean PSNR over "
                     f"predicted frames: {lab[0]} {pa:.2f} dB, {lab[1]} {pb:.2f} dB", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(args.out, f"pred_ep{ep_idx[e]}.png"))
        plt.close(fig)

        # ---- decomposition grid
        cols = [("GT", None, None)]
        for key, nm in (("rec", "recon"), ("fg", "foreground"), ("bg", "background"), ("mask", "mask"),
                        ("particles", "particles")):
            cols += [(f"{lab[0]} {nm}", dec_a, key), (f"{lab[1]} {nm}", dec_b, key)]
        fig, axes = plt.subplots(len(df), len(cols), figsize=(1.45 * len(cols), 1.5 * len(df)), dpi=110)
        for r, f in enumerate(df):
            for c, (title, d, key) in enumerate(cols):
                ax = axes[r, c]
                if d is None:
                    ax.imshow(hwc(gt[e, f]))
                elif key == "mask":
                    ax.imshow(d["mask"][e, f].numpy(), cmap="gray", vmin=0, vmax=1)
                elif key == "particles":
                    active = d["obj_on"][e, f] > 0.5
                    img = plot_keypoints_on_image(d["mu"][e, f][active], gt[e, f], radius=2, thickness=1,
                                                  kp_range=d["kp_range"], plot_numbers=False)
                    ax.imshow(img)
                    ax.text(0.5, -0.08, f"n={int(active.sum())}/{active.numel()}", transform=ax.transAxes,
                            ha="center", va="top", fontsize=7)
                else:
                    ax.imshow(hwc(d[key][e, f]))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"t={f}", fontsize=8)
        ra = sum(psnr(dec_a["rec"][e, f], gt[e, f]) for f in range(T + 1)) / (T + 1)
        rb = sum(psnr(dec_b["rec"][e, f], gt[e, f]) for f in range(T + 1)) / (T + 1)
        summary["recon_psnr_mean"][ep_idx[e]] = {lab[0]: ra, lab[1]: rb}
        summary["active_particles_mean"][ep_idx[e]] = {lab[0]: float((dec_a["obj_on"][e] > 0.5).sum(-1).float().mean()),
                                                       lab[1]: float((dec_b["obj_on"][e] > 0.5).sum(-1).float().mean())}
        fig.suptitle(f"decomposition, val episode {ep_idx[e]} | deterministic | mean reconstruction PSNR over {T + 1} "
                     f"frames: {lab[0]} {ra:.2f} dB, {lab[1]} {rb:.2f} dB | particles: encoder slots, obj_on > 0.5",
                     fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(args.out, f"decomp_ep{ep_idx[e]}.png"))
        plt.close(fig)
        print(f"episode {ep_idx[e]}: prediction PSNR {lab[0]} {pa:.2f} / {lab[1]} {pb:.2f} dB | reconstruction PSNR "
              f"{lab[0]} {ra:.2f} / {lab[1]} {rb:.2f} dB | active particles {summary['active_particles_mean'][ep_idx[e]]}",
              flush=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"wrote {args.out}/pred_ep*.png, {args.out}/decomp_ep*.png, {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
