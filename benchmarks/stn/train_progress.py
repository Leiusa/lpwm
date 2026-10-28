"""
Progress of a running training from its log (stdout/stderr of train_lpwm.py / train_dlp.py, tqdm bars included):

    python benchmarks/stn/train_progress.py LOG CONFIG.json

Reads the last tqdm bar and the config's num_epochs / max_steps_per_epoch (tqdm's own total is the full epoch even
when max_steps_per_epoch caps it), prints overall progress, s/step, time left for the training steps (evaluations come
on top), and the last [step-timing], validation and LPIPS lines. Works on a resumed run (epochs continue).
"""
import json
import re
import sys

log, cfg = sys.argv[1], json.load(open(sys.argv[2]))
n_ep, cap = cfg["num_epochs"], cfg.get("max_steps_per_epoch")
text = open(log, errors="replace").read().replace("\r", "\n")
bars = re.findall(r"epoch #(\d+).*?(\d+)/(\d+) \[([\d:]+)<[^,]*,\s*([\d.]+)(s/it|it/s)", text)
if not bars:
    print("no progress bar yet (loading / compiling?)")
else:
    ep, step, tot, _, rate, unit = bars[-1]
    ep, step = int(ep), int(step)
    per_ep = min(cap, int(tot)) if cap else int(tot)
    s_step = float(rate) if unit == "s/it" else 1 / float(rate)
    done, total = ep * per_ep + step, n_ep * per_ep
    w = 40
    fill = int(w * done / total)
    print(f"[{'#' * fill}{'.' * (w - fill)}] {100 * done / total:5.1f}%  epoch {ep}/{n_ep - 1}  step {step}/{per_ep}"
          f"  total {done}/{total}  {s_step:.3f} s/step  ~{(total - done) * s_step / 3600:.1f} h of steps left")
for key in ("[step-timing]", "validation loss:", "validation: lpips", "resuming from"):
    lines = [l for l in text.splitlines() if key in l]
    if lines:
        print(f"last {key.strip('[]:')}: {lines[-1].strip()[:200]}")
err = [l for l in text.splitlines() if "Traceback" in l or "Error" in l]
if err:
    print("!! ERROR:", err[-1].strip()[:200])
