#!/usr/bin/env python
"""
Milestone-1 visual quality check: for one seed's saved A and D checkpoints, decompose the same fixed
validation images (the ones already used for the fixed-16 PSNR metric, per seed) the way the model itself
does internally -- ground truth, full reconstruction, composited foreground/objects layer, background layer,
the aggregate object mask, and the particles the model actually placed on the image -- side by side for A and
D. This mirrors the output-dict keys the official DLPv3 tutorial notebook names (rec_rgb, dec_objects, bg_rgb,
alpha_masks); it does not touch that notebook, which only prints those keys' shapes and does not itself plot
them.

Particle count/location: the model always encodes a fixed pool of n_kp_enc latent particle slots per image
(from the config; static DLP does not filter particles in the decoder since timestep_horizon=1, so every
encoded particle is also a decode candidate). Which of them the model actually uses to explain the image is
governed by 'obj_on', a per-particle Beta-distributed activity gate; under deterministic=True (used here,
matching the eval harness) this is the Beta-mean activity probability, not a stochastic sample. A particle is
counted/plotted as "used" when obj_on > 0.5 -- the midpoint of a probability, fixed before looking at any
seed's images, not tuned from them. Location is the model's own plotting convention: mu_tot = z_base +
mu_offset, clamped to kp_range, drawn with the repo's own plot_keypoints_on_image (utils/util_func.py) --
the same function and the same keypoint tensor train_dlp.py's own training-time figures use.

Evaluation only, deterministic inference (model.eval(), no_grad, deterministic=True), same as m1_eval.py.
Reads eval_m1.json for the checkpoint paths and the exact fixed_batch_indices already recorded for the seed,
so the images match the ones the fixed-16 PSNR number in the report was computed from.

    python m1_decomp_grid.py --repo-root R --config CFG --eval-json seedN/eval_m1.json --out grid.png
"""
import argparse
import json
import os
import sys

import numpy as np


def to_hwc(t):
    return t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-json", default=None, help="milestone-1 mode: seed's eval_m1.json (ckpt paths + fixed_batch_indices)")
    ap.add_argument("--ckpt-a", default=None, help="any-checkpoint mode: first checkpoint (left column of each pair)")
    ap.add_argument("--ckpt-b", default=None, help="any-checkpoint mode: second checkpoint")
    ap.add_argument("--path-a", default="fused_cl", choices=("reference", "fused_cl"), help="inference path for --ckpt-a")
    ap.add_argument("--path-b", default="fused_cl", choices=("reference", "fused_cl"), help="inference path for --ckpt-b")
    ap.add_argument("--labels", default=None, help="column labels 'X,Y' (default A,D in milestone-1 mode, a,b otherwise)")
    ap.add_argument("--skip-hw-check", action="store_true", help="do not require the milestone-1 study GPU; the GPU is recorded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None, help="only for the figure title; eval_m1.json does not record it")
    args = ap.parse_args(argv)
    if (args.eval_json is None) == (args.ckpt_a is None or args.ckpt_b is None):
        ap.error("give either --eval-json, or both --ckpt-a and --ckpt-b")

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, os.path.join(repo, "benchmarks", "stn"))
    import bench_e2e as be
    be._bootstrap(repo)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import lpwm_stn
    import m1_common as MC
    from build_model import build
    from datasets.get_dataset import get_image_dataset
    from utils.util_func import plot_keypoints_on_image

    PATH_FLAGS = {"reference": dict(stn_backend="reference", fused_composite=False, particle_dec_channels_last=False),
                  "fused_cl": dict(stn_backend="triton", fused_composite=True, particle_dec_channels_last=True)}
    if args.eval_json:
        ev = json.load(open(args.eval_json))
        idx = ev["fixed_batch_indices"][: args.n_images]
        ckpts = {"A": ev["checkpoints"]["A"]["path"], "D": ev["checkpoints"]["D"]["path"]}
        SPECS = {"A": PATH_FLAGS["reference"], "D": PATH_FLAGS["fused_cl"]}
        labels = (args.labels or "A,D").split(",")
    else:
        idx = None
        ckpts = {"A": args.ckpt_a, "D": args.ckpt_b}
        SPECS = {"A": PATH_FLAGS[args.path_a], "D": PATH_FLAGS[args.path_b]}
        labels = (args.labels or "a,b").split(",")
    lab = {"A": labels[0], "D": labels[1]}
    for k, p in ckpts.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"checkpoint {k} missing on disk: {p}")

    if args.skip_hw_check:
        hw = {"nvidia_smi": [{"name": torch.cuda.get_device_name(0)}], "node": os.uname().nodename}
    else:
        hw = MC.verify_hardware()
    MC.set_agreed_environment()
    base_cfg = json.load(open(args.config))
    ds = get_image_dataset(base_cfg["ds"], base_cfg["root"], mode="valid", image_size=base_cfg["image_size"])
    if idx is None:   # the same fixed images the milestone-1 study used (spread over episodes, deterministic)
        import train_dlp_compare as C
        _, idx = C.fixed_validation_batch(ds, args.n_images)
    x = torch.stack([ds[i][0] for i in idx])
    x = x.reshape(-1, *x.shape[-3:])[: len(idx)].cuda()  # same reshape train_dlp_compare.fixed_validation_batch uses
    out = {}
    for which, flags in SPECS.items():
        cfg = dict(base_cfg)
        cfg.update(flags, batch_size=len(idx))
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cp = os.path.join(tmp, "cfg.json")
            json.dump(cfg, open(cp, "w"))
            model, _, _ = build(cp, "cuda")
        model.load_state_dict(torch.load(ckpts[which], map_location="cuda", weights_only=False), strict=True)
        model.eval()
        lpwm_stn.set_backend(flags["stn_backend"])
        n_kp_enc = model.n_kp_enc
        kp_range = model.kp_range
        with torch.no_grad():
            mo = model(x.unsqueeze(1), deterministic=True, with_loss=False)
            rec = mo["rec_rgb"].reshape(x.shape).clamp(0, 1)
            fg = mo["dec_objects"].reshape(x.shape).clamp(0, 1)
            bg = mo["bg_rgb"].reshape(x.shape).clamp(0, 1)
            am = mo["alpha_masks"]                                    # [N, n_kp, 1, H, W], per-particle importance*alpha
            mask = am.sum(dim=1).reshape(x.shape[0], 1, *x.shape[-2:]).clamp(0, 1)   # aggregate foreground visibility
            # particles: obj_on under deterministic=True is the Beta-posterior MEAN activity probability (not a
            # sample); a particle counts as "used" when that probability is > 0.5. mu_tot = z_base + mu_offset is
            # the model's own final particle location, the same tensor train_dlp.py's own figures plot.
            n = x.shape[0]
            obj_on = mo["obj_on"].reshape(n, -1).cpu()                # [N, n_kp_enc]
            mu_tot = mo["mu_tot"].reshape(n, obj_on.shape[1], 2).clamp(kp_range[0], kp_range[1]).cpu()
            active = obj_on > 0.5
            n_active = active.sum(dim=1).tolist()
            particle_imgs = []
            for r in range(n):
                kp_active = mu_tot[r][active[r]]
                img_np = plot_keypoints_on_image(kp_active, x[r], radius=2, thickness=2, kp_range=kp_range, plot_numbers=False)
                particle_imgs.append(torch.tensor(img_np).float().permute(2, 0, 1) / 255.0)
            particle_imgs = torch.stack(particle_imgs)
        out[which] = dict(rec=rec.cpu(), fg=fg.cpu(), bg=bg.cpu(), mask=mask.cpu(), particles=particle_imgs,
                          n_active=n_active, n_kp_enc=n_kp_enc)
        del model
        torch.cuda.empty_cache()

    cols = [("GT", None, None)]
    for key, name in (("rec", "recon"), ("fg", "foreground"), ("bg", "background"), ("mask", "mask"), ("particles", "particles")):
        cols += [(f"{lab['A']} {name}", "A", key), (f"{lab['D']} {name}", "D", key)]
    n = x.shape[0]
    fig, axes = plt.subplots(n, len(cols), figsize=(1.5 * len(cols), 1.5 * n), dpi=110)
    for r in range(n):
        for c, (title, which, key) in enumerate(cols):
            ax = axes[r, c]
            if which is None:
                img = to_hwc(x[r])
                ax.imshow(img)
            elif key == "mask":
                ax.imshow(out[which][key][r, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            elif key == "particles":
                ax.imshow(to_hwc(out[which][key][r]))
                ax.text(0.5, -0.08, f"n={out[which]['n_active'][r]}/{out[which]['n_kp_enc']}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7)
            else:
                ax.imshow(to_hwc(out[which][key][r]))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"val#{idx[r]}", fontsize=8)
    fig.suptitle(f"seed {args.seed if args.seed is not None else '?'} | {hw['nvidia_smi'][0]['name']} node {hw['node']} | "
                 f"{lab['A']}={os.path.basename(ckpts['A'])} {lab['D']}={os.path.basename(ckpts['D'])} | particle slots (n_kp_enc)="
                 f"{out['A']['n_kp_enc']} | 'used' = obj_on (Beta-posterior mean) > 0.5", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)

    counts_path = os.path.splitext(args.out)[0] + "_particle_counts.json"
    json.dump({"seed": args.seed, "n_kp_enc": out["A"]["n_kp_enc"], "val_indices": idx,
              "checkpoints": {lab["A"]: ckpts["A"], lab["D"]: ckpts["D"]},
              "n_active": {lab["A"]: out["A"]["n_active"], lab["D"]: out["D"]["n_active"]}}, open(counts_path, "w"), indent=1)
    print("wrote", counts_path)


if __name__ == "__main__":
    main()
