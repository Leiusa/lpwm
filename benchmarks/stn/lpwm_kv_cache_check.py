"""
LPWM sampling with and without the key/value cache (DLPDynamics.sample(kv_cache=True)): same outputs? how fast?

    python benchmarks/stn/lpwm_kv_cache_check.py CONFIG.json [--ckpt PATH] [--device cuda] [--bs 4]
        [--cond 1] [--steps 29] [--override '{"pint_dim": 128}'] [--repeats 5]

Without --ckpt the weights are random (seeded): this checks exactness of the cache, not prediction quality.
The input frames are random too; only the shapes matter. Deterministic sampling (deterministic=True) on both
paths, so the two unrolls see identical inputs and any difference is the cache's arithmetic.

Expected: per-timestep difference at rounding level while cond + k <= timestep_horizon (the cache is exact,
only matrix shapes differ); beyond that the cached run recomputes the window as the reference does, so the
difference stays at the level it reached. Reports max |a - b| per predicted timestep for every latent, and the
median wall time of the unroll for each path.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_model import build  # noqa: E402

KEYS = ("z_pos", "z_scale", "z_obj_on", "z_depth", "z_features", "z_bg_features", "z_context")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--cond", type=int, default=None, help="conditioning frames (default: config cond_steps)")
    ap.add_argument("--steps", type=int, default=None, help="predicted frames (default: animation_horizon - cond)")
    ap.add_argument("--override", default="{}", help="JSON applied to the config (e.g. smaller dims on CPU)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cfg.update(json.loads(args.override))
    tmp = os.path.join(os.path.dirname(os.path.abspath(args.config)), f".kvcheck_{os.getpid()}.json")
    json.dump(cfg, open(tmp, "w"))
    torch.manual_seed(args.seed)
    try:
        model, cfg, _ = build(tmp, device=args.device)
    finally:
        os.remove(tmp)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint {args.ckpt}: missing {len(missing)} unexpected {len(unexpected)}")
    else:
        # several output heads are zero-initialized, so at init some latents (e.g. z_pos) never move and a
        # difference in them would go unseen: perturb every weight a little
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.02 * torch.randn_like(p))
    model.eval()

    horizon = cfg["timestep_horizon"]
    cond = args.cond if args.cond is not None else cfg.get("cond_steps", 1)
    steps = args.steps if args.steps is not None else cfg.get("animation_horizon", horizon + 1) - cond
    size = cfg["image_size"]
    ch = cfg.get("ch", cfg.get("cdim", 3))
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.rand(args.bs, cond + steps, ch, size, size, generator=gen).to(args.device)
    print(f"config {args.config}  override {args.override}  device {args.device}  torch {torch.__version__}")
    print(f"block size (timestep_horizon) {horizon}  cond {cond}  steps {steps}  batch {args.bs}  "
          f"n_kp_enc {cfg.get('n_kp_enc')}  pint_dim {cfg.get('pint_dim')}")
    print(f"exact phase: predicted steps 1..{max(horizon - cond, 0)} (sequence length <= block size)")

    def run(kv):
        with torch.no_grad():
            _, z = model.sample_from_x(x, num_steps=steps, deterministic=True, cond_steps=cond, return_z=True,
                                       decode=False, kv_cache=kv)
        return z

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    ref, kvz = run(False), run(True)
    print(f"\nmax |ref - kv| per predicted step k (frame cond + k - 1)")
    print("step " + " ".join(f"{k:>13s}" for k in KEYS))
    worst_exact = 0.0
    for k in range(steps):
        row = []
        for key in KEYS:  # the last `steps` entries are the predicted ones (z_context is one shorter than z)
            a, b = ref[key][:, k - steps], kvz[key][:, k - steps]
            row.append((a - b).abs().max().item())
        if cond + k <= horizon:  # predicted from a sequence that fits in the block size
            worst_exact = max(worst_exact, max(row))
        print(f"{k + 1:4d} " + " ".join(f"{d:13.3e}" for d in row) + ("" if cond + k <= horizon else "  (slid)"))
    # scale: how much each latent moves between consecutive predicted steps (0 = constant, comparison is blind)
    moves = [(ref[key][:, -steps + 1:] - ref[key][:, -steps:-1]).abs().max().item() if steps > 1 else float("nan")
             for key in KEYS]
    print("move " + " ".join(f"{d:13.3e}" for d in moves) + "  (max step-to-step change of the reference)")
    print(f"worst difference in the exact phase: {worst_exact:.3e}")

    times = {}
    for kv in (False, True):
        run(kv)  # warmup
        ts = []
        for _ in range(args.repeats):
            sync()
            t0 = time.perf_counter()
            run(kv)
            sync()
            ts.append(time.perf_counter() - t0)
        times[kv] = sorted(ts)[len(ts) // 2]
    print(f"\nunroll time (encode cond frames + {steps} steps, median of {args.repeats}): "
          f"reference {times[False] * 1e3:.1f} ms, kv cache {times[True] * 1e3:.1f} ms, "
          f"speedup {times[False] / times[True]:.2f}x")


if __name__ == "__main__":
    main()
