"""GPU ledger of the milestone-1 study: sums every job's own start/end record (Slurm accounting is unavailable), including the
smoke test, failed attempts, hardware rejections and retries. A job still marked 'running' is counted up to now.

    python m1_ledger.py LEDGER_DIR [--cap-hours 8] [--json OUT]
"""
import argparse
import glob
import json
import time

ap = argparse.ArgumentParser()
ap.add_argument("ledger_dir")
ap.add_argument("--cap-hours", type=float, default=8.0)
ap.add_argument("--json", default=None)
a = ap.parse_args()
now = int(time.time())
rows, total = [], 0.0
for f in sorted(glob.glob(a.ledger_dir + "/*.json")):
    r = json.load(open(f))
    el = (now - r["start_epoch"]) if r["status"] == "running" else r["elapsed_s"]
    rows.append({**r, "counted_s": el})
    total += el
rows.sort(key=lambda r: r["start_epoch"])
print("%-9s %-10s %-5s %-11s %-22s %-8s %7s" % ("job", "kind", "seed", "node", "status", "exit", "minutes"))
for r in rows:
    print("%-9s %-10s %-5s %-11s %-22s %-8s %7.1f" % (r["job"], r["kind"], r.get("seed", "-"), r["node"], r["status"], r["exit_code"], r["counted_s"] / 60))
h = total / 3600
print(f"TOTAL {h:.3f} GPU-hours over {len(rows)} jobs | cap {a.cap_hours} | remaining {a.cap_hours - h:.3f}")
if a.json:
    json.dump({"jobs": rows, "total_gpu_hours": h, "cap_hours": a.cap_hours, "remaining_hours": a.cap_hours - h, "written_epoch": now}, open(a.json, "w"), indent=1)
