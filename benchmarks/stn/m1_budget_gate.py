"""Admission gate for milestone-1 jobs. Enforces the aggregate GPU-hour cap from inside the jobs.

Committed hours = finished jobs' actual elapsed time + every job that is still 'running' counted at its full time limit
(worst case) + this job's own time limit. A job may proceed only if committed <= cap. Stale 'running' entries (older than
limit + 10 min) count their limit as elapsed. Exit 0 = proceed, 5 = blocked (the job then uses essentially no GPU time).

    python m1_budget_gate.py LEDGER_DIR SELF_JOB LIMIT_SECONDS CAP_HOURS
"""
import glob
import json
import sys
import time

ledger, me, limit_s, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
now = int(time.time())
finished = running = 0.0
n_fin = n_run = 0
for f in glob.glob(ledger + "/*.json"):
    r = json.load(open(f))
    if r["job"] == me:
        continue
    lim = r.get("limit_s", limit_s)
    if r["status"] == "running":
        if now - r["start_epoch"] > lim + 600:
            finished += lim / 3600
            n_fin += 1
        else:
            running += lim / 3600
            n_run += 1
    else:
        finished += r["elapsed_s"] / 3600
        n_fin += 1
committed = finished + running + limit_s / 3600
print(f"budget gate: finished {finished:.3f} h ({n_fin} jobs) + running worst-case {running:.3f} h ({n_run} jobs) + this job worst-case {limit_s / 3600:.3f} h = {committed:.3f} h vs cap {cap} h")
if committed > cap:
    print("BUDGET_BLOCKED")
    sys.exit(5)
print("BUDGET_OK")
