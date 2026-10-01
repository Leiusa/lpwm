"""
Paired summary of the short training validation of `particle_dec_channels_last` (off = "fused", on = "fused_cl").
Reports differences only: no threshold is applied, no epoch is selected, and a small difference is not presented as
evidence of quality equivalence. The switch changes the actual computation precision of two particle_dec convolutions
(fp32 FFT -> TF32 algorithm), so nothing here is a same-precision comparison.

    python summarize_layout_quality.py OUT.json seed_dir_0 seed_dir_1 ...     (each dir has train_report.json)
"""

import json
import math
import os
import statistics as st
import sys

out_json, dirs = sys.argv[1], sys.argv[2:]
OFF, ON = "fused", "fused_cl"
B = 16

seeds, checks = {}, []
for d in dirs:
    r = json.load(open(os.path.join(d, "train_report.json")))
    runs = {x["path"]: x for x in r["runs"] if x["tag"] == "_p1"}
    rc = r["reconstruction"]["runs"]
    seed = r["seed"]
    S = {"dir": d, "job": r.get("slurm_job_id"), "host": r.get("host"), "order": r.get("order"), "epochs": r.get("epochs"), "paths": {},
         "weight_rel_l2_fused_cl_vs_fused": r["reconstruction"].get("weight_relative_l2_fused_cl_vs_fused")}
    for p in (OFF, ON):
        x = runs[p]
        n = len(x["epochs"])
        full = rc[p + "_p1"]["full_validation_set"]
        S["paths"][p] = {"val_loss_curve": x["validation_loss"], "train_loss_curve": [e["train_loss"] for e in x["epochs"]], "train_psnr_curve": [e["psnr"] for e in x["epochs"]],
                         "step_median_ms": [e["median_ms"] for e in x["epochs"]], "step_mean_ms": [e["mean_ms"] for e in x["epochs"]], "epoch_wall_s": [e["epoch_wall_s"] for e in x["epochs"]],
                         "iters": [e["iters"] for e in x["epochs"]], "peak_alloc_mb": [e["peak_alloc_mb"] for e in x["epochs"]], "peak_reserved_mb": [e.get("peak_reserved_mb") for e in x["epochs"]],
                         "reserved_now_mb": [e.get("reserved_now_mb") for e in x["epochs"]],
                         "final_val_loss": x["validation_loss"][-1], "final_train_loss": x["epochs"][-1]["train_loss"], "full_val_mse": full["mse"], "full_val_psnr_db": full["psnr_db"],
                         "full_val_images": full["images"], "fixed_batch_psnr_db": rc[p + "_p1"]["fixed_batch"]["psnr_db"], "startup": x.get("startup"), "first_step": x.get("first_step"),
                         "layout_startup_flag": x.get("layout_startup_flag"), "layout_first_step_calls": x.get("layout_first_step_calls")}
        P = S["paths"][p]
        vals = P["val_loss_curve"] + P["train_loss_curve"] + [P["full_val_mse"]]
        checks.append((seed, p, "finite", all(math.isfinite(v) for v in vals)))
        checks.append((seed, p, f"epochs=={r.get('epochs')} and val curve complete", n == r.get("epochs") and len(x["validation_loss"]) == r.get("epochs")))
        checks.append((seed, p, "1200 validation images", full["images"] == 1200))
    a, b = S["paths"][OFF], S["paths"][ON]
    checks.append((seed, "pair", "identical initial weights and first batch (hashes)", (a["startup"]["init_weights_sha256"], a["first_step"]["first_batch_sha256"]) == (b["startup"]["init_weights_sha256"], b["first_step"]["first_batch_sha256"])))
    checks.append((seed, "pair", "layout flag at startup: off False, on True", (a["layout_startup_flag"], b["layout_startup_flag"]) == ("False", "True")))
    checks.append((seed, "pair", "channels_last forward calls in first step: off 0, on 1", (a["layout_first_step_calls"], b["layout_first_step_calls"]) == (0, 1)))
    checks.append((seed, "pair", "fused composite called in first step on both", (a["first_step"]["fused_composite_calls"], b["first_step"]["fused_composite_calls"]) == (1, 1)))
    checks.append((seed, "pair", "same number of iterations per epoch", a["iters"] == b["iters"]))
    S["first_step_loss"] = {OFF: a["first_step"]["first_step_loss"], ON: b["first_step"]["first_step_loss"]}
    seeds[seed] = S

print("integrity checks:", "ALL PASSED" if all(c[3] for c in checks) else [c for c in checks if not c[3]])
seed_list = sorted(seeds)
E = seeds[seed_list[0]]["epochs"]
print("seeds", seed_list, "| epochs", E, "| jobs", {s: seeds[s]["job"] for s in seed_list}, "| hosts", {s: seeds[s]["host"] for s in seed_list}, "| run orders", {s: seeds[s]["order"] for s in seed_list})
print("first-step loss (reported, not a criterion):", {s: seeds[s]["first_step_loss"] for s in seed_list})
summary = {"seeds": seed_list, "epochs": E, "per_seed": seeds, "speed": {}, "quality": {}}


def pct(a, b):
    return 100 * (a - b) / b


print("\n=== A. Speed and memory per epoch (step = training step CUDA-event median within the epoch; throughput = images per second)")
for s in seed_list:
    a, b = seeds[s]["paths"][OFF], seeds[s]["paths"][ON]
    print(f"seed {s} (order {seeds[s]['order']})")
    for name, P in ((OFF, a), (ON, b)):
        print("  %-9s step ms %s | step img/s %s | epoch wall s %s | epoch img/s %s" % (name, [round(v, 1) for v in P["step_median_ms"]], [round(B * 1000 / v, 1) for v in P["step_median_ms"]],
              [round(v, 1) for v in P["epoch_wall_s"]], [round(i * B / w, 1) for i, w in zip(P["iters"], P["epoch_wall_s"])]))
        print("  %-9s peak allocated MB %s | peak reserved MB %s | reserved now MB %s" % ("", [round(v) for v in P["peak_alloc_mb"]], [round(v) if v else v for v in P["peak_reserved_mb"]], [round(v) if v else v for v in P["reserved_now_mb"]]))
    d_step = [pct(y, x) for x, y in zip(a["step_median_ms"], b["step_median_ms"])]
    d_wall = [pct(y, x) for x, y in zip(a["epoch_wall_s"], b["epoch_wall_s"])]
    print("  on vs off: step time %s %% | epoch wall %s %% | peak allocated %s MB | peak reserved %s MB" % ([round(v, 1) for v in d_step], [round(v, 1) for v in d_wall],
          [round(y - x, 1) for x, y in zip(a["peak_alloc_mb"], b["peak_alloc_mb"])], [round(y - x) for x, y in zip(a["peak_reserved_mb"], b["peak_reserved_mb"])]))
    summary["speed"][s] = {"step_pct": d_step, "epoch_wall_pct": d_wall}
for name in (OFF, ON):
    steps = [v for s in seed_list for v in seeds[s]["paths"][name]["step_median_ms"]]
    walls = [v for s in seed_list for v in seeds[s]["paths"][name]["epoch_wall_s"]]
    pa = [v for s in seed_list for v in seeds[s]["paths"][name]["peak_alloc_mb"]]
    pr = [v for s in seed_list for v in seeds[s]["paths"][name]["peak_reserved_mb"] if v]
    rn = [v for s in seed_list for v in seeds[s]["paths"][name]["reserved_now_mb"] if v]
    print("all epochs, all seeds %-9s step ms mean %.1f (range %.1f-%.1f) | epoch wall s mean %.1f (range %.1f-%.1f) | peak alloc range %.0f-%.0f | peak reserved range %.0f-%.0f | reserved now range %.0f-%.0f" % (
        name, st.fmean(steps), min(steps), max(steps), st.fmean(walls), min(walls), max(walls), min(pa), max(pa), min(pr), max(pr), min(rn), max(rn)))
    summary["speed"][name] = {"step_ms_mean": st.fmean(steps), "epoch_wall_s_mean": st.fmean(walls), "peak_alloc_range": [min(pa), max(pa)], "peak_reserved_range": [min(pr), max(pr)]}
allsteps = {n: st.fmean(v for s in seed_list for v in seeds[s]["paths"][n]["step_median_ms"]) for n in (OFF, ON)}
allwalls = {n: st.fmean(v for s in seed_list for v in seeds[s]["paths"][n]["epoch_wall_s"]) for n in (OFF, ON)}
print("pooled: step %.1f -> %.1f ms (%+.1f%%) | epoch wall %.1f -> %.1f s (%+.1f%%) | step throughput %.1f -> %.1f img/s" % (allsteps[OFF], allsteps[ON], pct(allsteps[ON], allsteps[OFF]),
      allwalls[OFF], allwalls[ON], pct(allwalls[ON], allwalls[OFF]), B * 1000 / allsteps[OFF], B * 1000 / allsteps[ON]))
summary["speed"]["pooled"] = {"step_ms": allsteps, "epoch_wall_s": allwalls}

print("\n=== B. Quality at the final checkpoint (epoch %s), evaluation identical for both: reference-path STN, NCHW particle decoder, whole validation set" % E)
print("%-5s %-9s %-10s %-13s %-13s %-10s %-14s" % ("seed", "layout", "val loss", "full-val MSE", "full-val PSNR", "train loss", "fixed-16 PSNR"))
diffs = {"val_loss": [], "val_loss_pct": [], "full_val_mse_pct": [], "full_val_psnr_db": [], "fixed_batch_psnr_db": [], "train_loss_pct": []}
for s in seed_list:
    for name in (OFF, ON):
        x = seeds[s]["paths"][name]
        print("%-5s %-9s %-10.3f %-13.4e %-13.3f %-10.3f %-14.3f" % (s, name, x["final_val_loss"], x["full_val_mse"], x["full_val_psnr_db"], x["final_train_loss"], x["fixed_batch_psnr_db"]))
    a, b = seeds[s]["paths"][OFF], seeds[s]["paths"][ON]
    diffs["val_loss"].append(b["final_val_loss"] - a["final_val_loss"])
    diffs["val_loss_pct"].append(pct(b["final_val_loss"], a["final_val_loss"]))
    diffs["full_val_mse_pct"].append(pct(b["full_val_mse"], a["full_val_mse"]))
    diffs["full_val_psnr_db"].append(b["full_val_psnr_db"] - a["full_val_psnr_db"])
    diffs["fixed_batch_psnr_db"].append(b["fixed_batch_psnr_db"] - a["fixed_batch_psnr_db"])
    diffs["train_loss_pct"].append(pct(b["final_train_loss"], a["final_train_loss"]))
print("\nPaired differences (on minus off) per seed:")
for k, v in diffs.items():
    print("  %-20s %s" % (k, "  ".join("seed %s %+.4g" % (s, x) for s, x in zip(seed_list, v))))
summary["quality"]["paired_final"] = diffs
print("\n=== C. Curves per epoch (every epoch shown, none selected)")
for s in seed_list:
    for name in (OFF, ON):
        P = seeds[s]["paths"][name]
        print("  seed %s %-9s val loss %s | train loss %s | train PSNR %s" % (s, name, [round(v, 3) for v in P["val_loss_curve"]], [round(v, 3) for v in P["train_loss_curve"]], [round(v, 2) for v in P["train_psnr_curve"]]))
    a, b = seeds[s]["paths"][OFF], seeds[s]["paths"][ON]
    print("  seed %s on-off: val loss %s | train loss %% %s" % (s, [round(y - x, 3) for x, y in zip(a["val_loss_curve"], b["val_loss_curve"])], [round(pct(y, x), 2) for x, y in zip(a["train_loss_curve"], b["train_loss_curve"])]))
print("\n=== D. Final weights: relative L2 between the two layouts' final models: %s" % {s: seeds[s]["weight_rel_l2_fused_cl_vs_fused"] for s in seed_list})
json.dump(summary, open(out_json, "w"), indent=2)
print("\nwrote", out_json)
