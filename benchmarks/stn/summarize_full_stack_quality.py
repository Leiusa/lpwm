"""
Paired summary of the full integrated-stack quality validation (A = reference vs D = triton + composite fusion +
particle_dec channels_last). Reports differences only: no threshold is applied, no epoch is selected, and a small
difference is not presented as evidence of quality equivalence. D changes the actual computation precision of two
particle_dec convolutions (fp32 FFT -> TF32), so this is not a same-precision comparison.

    python summarize_full_stack_quality.py OUT.json seed_dir_0 seed_dir_1 seed_dir_2   (each dir has train_report.json)
"""

import json
import math
import os
import statistics as st
import sys

out_json, dirs = sys.argv[1], sys.argv[2:]
A, D = "reference", "fused_cl"

seeds, checks = {}, []
for d in dirs:
    r = json.load(open(os.path.join(d, "train_report.json")))
    runs = {x["path"]: x for x in r["runs"] if x["tag"] == "_p1"}
    rc = r["reconstruction"]["runs"]
    seed = r["seed"]
    S = {"dir": d, "job": r.get("slurm_job_id"), "host": r.get("host"), "order": r.get("order"), "epochs": r.get("epochs"), "paths": {},
         "weight_rel_l2_D_vs_A": r["reconstruction"].get("weight_relative_l2_between_final_models", {}).get("fused_cl_vs_reference"),
         "deployment_check": r.get("deployment_check")}
    for p in (A, D):
        x = runs[p]
        n = len(x["epochs"])
        full = rc[p + "_p1"]["full_validation_set"]
        S["paths"][p] = {"val_loss_curve": x["validation_loss"], "train_loss_curve": [e["train_loss"] for e in x["epochs"]], "train_psnr_curve": [e["psnr"] for e in x["epochs"]],
                         "step_median_ms": [e["median_ms"] for e in x["epochs"]], "epoch_wall_s": [e["epoch_wall_s"] for e in x["epochs"]], "iters": [e["iters"] for e in x["epochs"]],
                         "peak_alloc_mb": [e["peak_alloc_mb"] for e in x["epochs"]], "peak_reserved_mb": [e.get("peak_reserved_mb") for e in x["epochs"]],
                         "final_val_loss": x["validation_loss"][-1], "final_train_loss": x["epochs"][-1]["train_loss"], "full_val_mse": full["mse"], "full_val_psnr_db": full["psnr_db"],
                         "full_val_images": full["images"], "fixed_batch_psnr_db": rc[p + "_p1"]["fixed_batch"]["psnr_db"], "startup": x.get("startup"), "first_step": x.get("first_step")}
        P = S["paths"][p]
        vals = P["val_loss_curve"] + P["train_loss_curve"] + [P["full_val_mse"]]
        checks.append((seed, p, "finite", all(math.isfinite(v) for v in vals)))
        checks.append((seed, p, f"epochs=={r.get('epochs')} and val curve complete", n == r.get("epochs") and len(x["validation_loss"]) == r.get("epochs")))
        checks.append((seed, p, "1200 validation images", full["images"] == 1200))
    a, b = S["paths"][A], S["paths"][D]
    checks.append((seed, "pair", "identical initial weights and first batch (hashes)", (a["startup"]["init_weights_sha256"], a["first_step"]["first_batch_sha256"]) == (b["startup"]["init_weights_sha256"], b["first_step"]["first_batch_sha256"])))
    checks.append((seed, "pair", "path check (fused/cl calls in first step: A 0/0, D 1/1)", (a["first_step"]["fused_composite_calls"], b["first_step"]["fused_composite_calls"]) == (0, 1)))
    checks.append((seed, "pair", "same number of iterations per epoch", a["iters"] == b["iters"]))
    S["first_step_loss"] = {A: a["first_step"]["first_step_loss"], D: b["first_step"]["first_step_loss"]}
    seeds[seed] = S

print("integrity checks:", "ALL PASSED" if all(c[3] for c in checks) else [c for c in checks if not c[3]])
seed_list = sorted(seeds)
E = seeds[seed_list[0]]["epochs"]
print("seeds", seed_list, "| epochs", E, "| jobs", {s: seeds[s]["job"] for s in seed_list}, "| hosts", {s: seeds[s]["host"] for s in seed_list}, "| run orders", {s: seeds[s]["order"] for s in seed_list})
print("first-step loss (reported, not a criterion):", {s: seeds[s]["first_step_loss"] for s in seed_list})
summary = {"seeds": seed_list, "epochs": E, "per_seed": seeds, "speed": {}, "quality": {}}


def pct(a, b):
    return 100 * (a - b) / b


B = None
try:
    B = json.load(open(seeds[seed_list[0]]["dir"] + "/cfg_" + A + "_p1.json")).get("batch_size")
except Exception:
    pass

print("\n=== A. Speed and memory per epoch (step = training step CUDA-event median within the epoch)")
for s in seed_list:
    a, b = seeds[s]["paths"][A], seeds[s]["paths"][D]
    print(f"seed {s} (order {seeds[s]['order']})")
    for name, P in ((A, a), (D, b)):
        print("  %-9s step ms %s | epoch wall s %s | peak allocated MB %s | peak reserved MB %s" % (name, [round(v, 1) for v in P["step_median_ms"]],
              [round(v, 1) for v in P["epoch_wall_s"]], [round(v) for v in P["peak_alloc_mb"]], [round(v) if v else v for v in P["peak_reserved_mb"]]))
    d_step = [pct(y, x) for x, y in zip(a["step_median_ms"], b["step_median_ms"])]
    d_wall = [pct(y, x) for x, y in zip(a["epoch_wall_s"], b["epoch_wall_s"])]
    print("  D vs A: step time %s %% | epoch wall %s %% | peak allocated %s MB | peak reserved %s MB" % ([round(v, 1) for v in d_step], [round(v, 1) for v in d_wall],
          [round(y - x, 1) for x, y in zip(a["peak_alloc_mb"], b["peak_alloc_mb"])], [round(y - x) if x and y else None for x, y in zip(a["peak_reserved_mb"], b["peak_reserved_mb"])]))
    summary["speed"][s] = {"step_pct": d_step, "epoch_wall_pct": d_wall}
allsteps = {n: st.fmean(v for s in seed_list for v in seeds[s]["paths"][n]["step_median_ms"]) for n in (A, D)}
allwalls = {n: st.fmean(v for s in seed_list for v in seeds[s]["paths"][n]["epoch_wall_s"]) for n in (A, D)}
print("pooled (all epochs, all seeds): step %.1f -> %.1f ms (%+.1f%%, %.2fx)" % (allsteps[A], allsteps[D], pct(allsteps[D], allsteps[A]), allsteps[A] / allsteps[D]))
if B:
    print("  throughput: %.1f -> %.1f img/s (B=%d)" % (B * 1000 / allsteps[A], B * 1000 / allsteps[D], B))
print("  epoch wall: %.1f -> %.1f s (%+.1f%%)" % (allwalls[A], allwalls[D], pct(allwalls[D], allwalls[A])))
summary["speed"]["pooled"] = {"step_ms": allsteps, "epoch_wall_s": allwalls, "batch_size": B}

print("\n=== B. Quality at the final checkpoint (epoch %s), both evaluated on the REFERENCE path, whole validation set" % E)
print("%-5s %-9s %-10s %-13s %-13s %-10s %-14s" % ("seed", "path", "val loss", "full-val MSE", "full-val PSNR", "train loss", "fixed-16 PSNR"))
diffs = {"val_loss": [], "val_loss_pct": [], "full_val_mse_pct": [], "full_val_psnr_db": [], "fixed_batch_psnr_db": [], "train_loss_pct": []}
for s in seed_list:
    for name in (A, D):
        x = seeds[s]["paths"][name]
        print("%-5s %-9s %-10.3f %-13.4e %-13.3f %-10.3f %-14.3f" % (s, name, x["final_val_loss"], x["full_val_mse"], x["full_val_psnr_db"], x["final_train_loss"], x["fixed_batch_psnr_db"]))
    a, b = seeds[s]["paths"][A], seeds[s]["paths"][D]
    diffs["val_loss"].append(b["final_val_loss"] - a["final_val_loss"])
    diffs["val_loss_pct"].append(pct(b["final_val_loss"], a["final_val_loss"]))
    diffs["full_val_mse_pct"].append(pct(b["full_val_mse"], a["full_val_mse"]))
    diffs["full_val_psnr_db"].append(b["full_val_psnr_db"] - a["full_val_psnr_db"])
    diffs["fixed_batch_psnr_db"].append(b["fixed_batch_psnr_db"] - a["fixed_batch_psnr_db"])
    diffs["train_loss_pct"].append(pct(b["final_train_loss"], a["final_train_loss"]))
print("\nPer-seed paired differences (D minus A):")
for k, v in diffs.items():
    print("  %-18s %s" % (k, "  ".join("seed %s %+.4g" % (s, x) for s, x in zip(seed_list, v))))
print("\nAcross-seed summary of the paired differences (mean, sample std, range) -- context, not a pass/fail rule:")
for k, v in diffs.items():
    print("  %-18s mean %+.4g | std %s | range %+.4g .. %+.4g" % (k, st.fmean(v), ("%.3g" % st.stdev(v)) if len(v) > 1 else "n/a", min(v), max(v)))
summary["quality"]["paired_final"] = diffs

print("\n=== C. Curves per epoch (every epoch shown, none selected)")
for s in seed_list:
    for name in (A, D):
        P = seeds[s]["paths"][name]
        print("  seed %s %-9s val loss %s | train loss %s | train PSNR %s" % (s, name, [round(v, 3) for v in P["val_loss_curve"]], [round(v, 3) for v in P["train_loss_curve"]], [round(v, 2) for v in P["train_psnr_curve"]]))

print("\n=== D. Deployment cross-check: D-checkpoint via its own real path vs via the reference path")
for s in seed_list:
    dc = seeds[s]["deployment_check"]
    if dc:
        print("  seed %s: %s" % (s, json.dumps(dc["diff_native_minus_reference_path"])))
        print("           fused_composite_calls_during_eval=%s channels_last_calls_during_eval=%s" % (dc.get("fused_composite_calls_during_eval"), dc.get("channels_last_calls_during_eval")))

print("\n=== E. Final weights: relative L2 between A's and D's final models: %s" % {s: seeds[s]["weight_rel_l2_D_vs_A"] for s in seed_list})
json.dump(summary, open(out_json, "w"), indent=2)
print("\nwrote", out_json)
