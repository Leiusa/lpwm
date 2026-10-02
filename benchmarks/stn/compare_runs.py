#!/usr/bin/env python
"""
Side-by-side per-epoch comparison of train_dlp.py / train_lpwm.py runs, read from their log files.

Per epoch and run: median step time, peak allocated memory, train loss, train PSNR ([step-timing] lines, needs
log_step_timing), the validation ELBO when it improved ("validation loss updated" lines; a stochastic estimate), and
the deterministic video-prediction metrics LPWM logs with eval_im_metrics ("validation: lpips: .., psnr: .., ssim: ..",
from epoch 1 on). The first run is the reference; differences of the others are printed against it. No threshold is
applied.

    python compare_runs.py train_lpwm_q_fp32.log train_lpwm_q_tf32.log [--labels fp32,tf32]
"""
import argparse
import re
import sys

STEP = re.compile(r"\[step-timing\] epoch=(\d+) iters=(\d+) median_ms=([\d.]+).*?peak_alloc_mb=([\d.]+) "
                  r"train_loss=([-\d.eE+]+) train_rec=([-\d.eE+]+) psnr=([-\d.eE+]+) epoch_wall_s=([\d.]+)")
IMM = re.compile(r"validation: lpips: ([\d.]+), psnr: ([\d.]+), ssim: ([\d.]+)")
VAL = re.compile(r"validation loss updated: ([-\d.e+]+) -> ([-\d.e+]+)")
PATH = re.compile(r"path check \(first step\) ok: (.*)")


def parse(path):
    text = open(path, "rb").read().decode("utf-8", errors="replace")
    epochs = {}
    for m in STEP.finditer(text):
        e = int(m.group(1))
        epochs.setdefault(e, {}).update(step_ms=float(m.group(3)), peak_gb=float(m.group(4)) / 1e3,
                                        train_loss=float(m.group(5)), train_psnr=float(m.group(7)),
                                        epoch_wall_s=float(m.group(8)))
    # image metrics are logged after the epoch's [step-timing] line: attach each to the closest preceding epoch
    pos_epochs = [(m.start(), int(m.group(1))) for m in STEP.finditer(text)]
    for pattern, keys in ((IMM, ("val_lpips", "val_psnr", "val_ssim")), (VAL, (None, "val_elbo_best"))):
        for m in pattern.finditer(text):
            prev = [e for p, e in pos_epochs if p < m.start()]
            if not prev:
                continue
            for k, g in zip(keys, m.groups()):
                if k:
                    epochs.setdefault(prev[-1], {})[k] = float(g)
    path_line = PATH.search(text)
    return epochs, path_line.group(1) if path_line else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--labels", default=None)
    args = ap.parse_args()
    labels = args.labels.split(",") if args.labels else [p.rsplit("/", 1)[-1] for p in args.logs]
    runs = [parse(p) for p in args.logs]
    for lab, (_, path_line) in zip(labels, runs):
        print(f"[{lab}] {path_line or 'no path check line found'}")
    cols = ("step_ms", "peak_gb", "train_loss", "train_psnr", "val_elbo_best", "val_lpips", "val_psnr", "val_ssim")
    all_epochs = sorted(set().union(*[set(r[0]) for r in runs]))
    for col in cols:
        if not any(col in r[0].get(e, {}) for r in runs for e in all_epochs):
            continue
        print(f"\n== {col}" + ("   (lower is better)" if col in ("step_ms", "peak_gb", "train_loss", "val_elbo_best", "val_lpips")
                               else "   (higher is better)"))
        print("epoch  " + "  ".join(f"{lab:>14s}" for lab in labels) + "  " +
              "  ".join(f"{lab + ' - ' + labels[0]:>22s}" for lab in labels[1:]))
        for e in all_epochs:
            vals = [r[0].get(e, {}).get(col) for r in runs]
            if all(v is None for v in vals):
                continue
            cells = "  ".join(f"{v:14.4f}" if v is not None else f"{'-':>14s}" for v in vals)
            diffs = []
            for v in vals[1:]:
                if v is None or vals[0] is None:
                    diffs.append(f"{'-':>22s}")
                else:
                    rel = 100 * (v - vals[0]) / abs(vals[0]) if vals[0] else float("nan")
                    diffs.append(f"{v - vals[0]:+12.4f} ({rel:+6.2f}%)")
            print(f"{e:5d}  {cells}  " + "  ".join(diffs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
