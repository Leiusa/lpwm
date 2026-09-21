"""
Per-seed and pooled summary of the quality-validation runs. Reports differences only: no threshold is applied,
no epoch is selected, and a small mean difference is NOT presented as evidence of quality equivalence.

    python summarize_quality.py OUT.json seed_dir_0 seed_dir_1 seed_dir_2     (each dir has train_report.json)
"""

import json
import math
import os
import statistics as st
import sys

out_json, dirs = sys.argv[1], sys.argv[2:]
PATHS = ("reference", "triton", "fused")
PAIRS = (("triton", "reference"), ("fused", "triton"), ("fused", "reference"))   # first two are the pre-declared comparisons


def load(d):
    f = os.path.join(d, "train_report.json")
    r = json.load(open(f if os.path.exists(f) else os.path.join(d, "train_compare.json")))
    runs = {x["path"]: x for x in r["runs"] if x["tag"] == "_p1"}
    rc = r["reconstruction"]["runs"]
    return r, runs, rc


seeds = {}
checks = []
for d in dirs:
    r, runs, rc = load(d)
    seed = r.get("seed", 0)
    S = {"dir": d, "job": r.get("slurm_job_id"), "host": r.get("host"), "order": r.get("order"), "epochs": r.get("epochs"), "paths": {}}
    for p in PATHS:
        x = runs[p]
        n = len(x["epochs"])
        full = rc[p + "_p1"]["full_validation_set"]
        S["paths"][p] = {"val_loss_curve": x["validation_loss"], "train_loss_curve": [e["train_loss"] for e in x["epochs"]],
                         "train_psnr_curve": [e["psnr"] for e in x["epochs"]],
                         "final_val_loss": x["validation_loss"][-1], "final_train_loss": x["epochs"][-1]["train_loss"],
                         "full_val_mse": full["mse"], "full_val_psnr_db": full["psnr_db"], "full_val_images": full["images"],
                         "fixed_batch_psnr_db": rc[p + "_p1"]["fixed_batch"]["psnr_db"],
                         "step_ms_median": st.median(e["median_ms"] for e in x["epochs"]), "peak_alloc_mb": max(e["peak_alloc_mb"] for e in x["epochs"]),
                         "epoch_wall_s_mean": st.fmean(e["epoch_wall_s"] for e in x["epochs"]), "epochs_run": n,
                         "startup": x.get("startup"), "first_step": x.get("first_step")}
        vals = S["paths"][p]["val_loss_curve"] + S["paths"][p]["train_loss_curve"] + [S["paths"][p]["full_val_mse"]]
        checks.append((seed, p, "finite", all(math.isfinite(v) for v in vals)))
        checks.append((seed, p, "epochs==%s" % r.get("epochs"), n == r.get("epochs") and len(x["validation_loss"]) == r.get("epochs")))
        checks.append((seed, p, "1200 validation images", full["images"] == 1200))
    same = {(S["paths"][p]["startup"]["init_weights_sha256"], S["paths"][p]["first_step"]["first_batch_sha256"]) for p in PATHS}
    checks.append((seed, "all", "identical initial weights and first batch (hashes) across paths", len(same) == 1))
    fl = [S["paths"][p]["first_step"]["first_step_loss"] for p in PATHS]
    S["first_step_loss"] = {"values": dict(zip(PATHS, fl)), "max_relative_spread": (max(fl) - min(fl)) / abs(fl[0])}   # reported, not a criterion
    checks.append((seed, "all", "path check (fused calls in step 0: ref 0, triton 0, fused 1)",
                   [S["paths"][p]["first_step"]["fused_composite_calls"] for p in PATHS] == [0, 0, 1]))
    seeds[seed] = S

print("integrity checks:", "ALL PASSED" if all(c[3] for c in checks) else [c for c in checks if not c[3]])
seed_list = sorted(seeds)
E = seeds[seed_list[0]]["epochs"]
print("first-step loss (reported, not a criterion):", {s: seeds[s]["first_step_loss"] for s in seed_list})
print("seeds:", seed_list, "| epochs per path:", E, "| jobs:", {s: seeds[s]["job"] for s in seed_list}, "| hosts:", {s: seeds[s]["host"] for s in seed_list}, "| run orders:", {s: seeds[s]["order"] for s in seed_list})


def line(vals):
    return "mean %+.4g | sample std %.3g | min %+.4g | max %+.4g" % (st.fmean(vals), st.stdev(vals) if len(vals) > 1 else float("nan"), min(vals), max(vals))


print("\n=== A. Final checkpoint (epoch %s), per seed" % E)
print("%-5s %-10s %-12s %-12s %-12s %-10s %-14s" % ("seed", "path", "val loss", "full-val MSE", "full-val PSNR", "train loss", "fixed-16 PSNR"))
for s in seed_list:
    for p in PATHS:
        x = seeds[s]["paths"][p]
        print("%-5s %-10s %-12.3f %-12.4e %-12.3f %-10.3f %-14.3f" % (s, p, x["final_val_loss"], x["full_val_mse"], x["full_val_psnr_db"], x["final_train_loss"], x["fixed_batch_psnr_db"]))

summary = {"seeds": seed_list, "epochs": E, "per_seed": seeds, "pairs": {}, "training_loss_offset_percent": {}}
print("\n=== B. Per-seed differences (first name minus second) and their spread over seeds")
for a, b in PAIRS:
    tag = "%s - %s" % (a, b) + ("" if (a, b) != ("fused", "reference") else "   (additional, not a pre-declared comparison)")
    print("\n  " + tag)
    rows = {"val_loss": [], "val_loss_pct": [], "full_val_mse": [], "full_val_mse_pct": [], "full_val_psnr_db": []}
    for s in seed_list:
        x, y = seeds[s]["paths"][a], seeds[s]["paths"][b]
        d = {"val_loss": x["final_val_loss"] - y["final_val_loss"], "val_loss_pct": 100 * (x["final_val_loss"] - y["final_val_loss"]) / y["final_val_loss"],
             "full_val_mse": x["full_val_mse"] - y["full_val_mse"], "full_val_mse_pct": 100 * (x["full_val_mse"] - y["full_val_mse"]) / y["full_val_mse"],
             "full_val_psnr_db": x["full_val_psnr_db"] - y["full_val_psnr_db"]}
        for k, v in d.items():
            rows[k].append(v)
        print("    seed %s: val loss %+.3f (%+.2f%%) | full-val MSE %+.3e (%+.2f%%) | full-val PSNR %+.3f dB" % (s, d["val_loss"], d["val_loss_pct"], d["full_val_mse"], d["full_val_mse_pct"], d["full_val_psnr_db"]))
    for k, v in rows.items():
        print("    %-18s %s" % (k, line(v)))
    summary["pairs"]["%s_minus_%s" % (a, b)] = {"per_seed": {k: v for k, v in rows.items()}, "mean": {k: st.fmean(v) for k, v in rows.items()},
                                                 "std": {k: (st.stdev(v) if len(v) > 1 else None) for k, v in rows.items()},
                                                 "min": {k: min(v) for k, v in rows.items()}, "max": {k: max(v) for k, v in rows.items()}}

print("\n=== C. Context: each path's own spread over seeds at the final checkpoint (the scale the paired differences should be read against)")
for p in PATHS:
    for k, lab in (("final_val_loss", "val loss"), ("full_val_mse", "full-val MSE"), ("full_val_psnr_db", "full-val PSNR")):
        v = [seeds[s]["paths"][p][k] for s in seed_list]
        print("  %-10s %-14s mean %.4g | sample std %.3g | range %.4g .. %.4g" % (p, lab, st.fmean(v), st.stdev(v) if len(v) > 1 else float("nan"), min(v), max(v)))

print("\n=== D. Training-loss offset relative to reference, per epoch (percent), kept separate from the quality comparison")
for p in ("triton", "fused"):
    per_epoch = []
    for e in range(E):
        vals = [100 * (seeds[s]["paths"][p]["train_loss_curve"][e] / seeds[s]["paths"]["reference"]["train_loss_curve"][e] - 1) for s in seed_list]
        per_epoch.append(vals)
    summary["training_loss_offset_percent"][p] = per_epoch
    print("  %-7s " % p + "  ".join("e%d %+.2f" % (e + 1, st.fmean(v)) for e, v in enumerate(per_epoch)) + "   (mean over seeds)")
    print("          per-seed final epoch: " + ", ".join("seed %s %+.2f%%" % (s, per_epoch[-1][i]) for i, s in enumerate(seed_list)))

print("\n=== E. Validation loss per epoch (curves; every epoch is shown, none selected)")
for s in seed_list:
    for p in PATHS:
        print("  seed %s %-10s " % (s, p) + " ".join("%7.3f" % v for v in seeds[s]["paths"][p]["val_loss_curve"]))

print("\n=== F. Cost context per path (median over epochs within each job; jobs may sit on different nodes)")
for p in PATHS:
    print("  %-10s step ms %s | peak alloc MB %s | epoch wall s %s" % (p, [round(seeds[s]["paths"][p]["step_ms_median"], 1) for s in seed_list],
          [round(seeds[s]["paths"][p]["peak_alloc_mb"]) for s in seed_list], [round(seeds[s]["paths"][p]["epoch_wall_s_mean"], 1) for s in seed_list]))
json.dump(summary, open(out_json, "w"), indent=2)
print("\nwrote", out_json)
