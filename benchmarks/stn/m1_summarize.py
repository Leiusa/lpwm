"""
Analysis of the milestone-1 controlled quality study (A vs D static DLP, six seeds, verified RTX A6000, paired within seed).

PRIMARY: evaluation E2 (D checkpoint, full optimized path) minus E1 (A checkpoint, original path): full-validation pooled PSNR
difference at epoch 8, paired over seeds, mean, sample std, standard error and a two-sided 95% t-interval (df = n-1).
No equivalence margin, no tolerance, no epoch selection. Diagnostics (inference only): E2-E3, E2-E4. Supplementary:
pooled MSE, per-image mean PSNR, fixed 16-image batch, logged (stochastic) validation loss, paired training time / throughput / memory.

    python m1_summarize.py RUNS_DIR OUT.json
      RUNS_DIR contains seed<N>-<job>/ directories with train_report.json and eval_m1.json
"""
import glob
import json
import math
import os
import re
import statistics as st
import sys

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
runs_dir, out_json = sys.argv[1:3]
per = {}
for d in sorted(glob.glob(os.path.join(runs_dir, "seed*-*"))):
    m = re.match(r"seed(\d+)-", os.path.basename(d))
    tr, ev = os.path.join(d, "train_report.json"), os.path.join(d, "eval_m1.json")
    if m and os.path.exists(tr) and os.path.exists(ev):
        per[int(m.group(1))] = (json.load(open(tr)), json.load(open(ev)), d)
seeds = sorted(per)
n = len(seeds)
print("seeds with complete outputs:", seeds)
print("GPU models (expected, per seed):", {s: per[s][0]["hardware"]["expected_gpu"] for s in seeds}, "-> unified:", len({per[s][0]["hardware"]["expected_gpu"] for s in seeds}) == 1)


def ci(vals):
    mean = st.fmean(vals)
    if len(vals) < 2:
        return {"n": len(vals), "mean": mean}
    sd = st.stdev(vals)
    se = sd / math.sqrt(len(vals))
    t = T975[len(vals) - 1]
    return {"n": len(vals), "mean": mean, "sample_std": sd, "se": se, "t_crit": t, "ci95": [mean - t * se, mean + t * se], "min": min(vals), "max": max(vals), "per_seed": vals,
            "n_positive": sum(v > 0 for v in vals), "n_negative": sum(v < 0 for v in vals)}


out = {"seeds": seeds, "integrity": {}, "primary": {}, "diagnostics": {}, "supplementary": {}, "performance": {}}
print("\n== integrity (per seed)")
for s in seeds:
    tr, ev, d = per[s]
    runs = {r["path"]: r for r in tr["runs"]}
    a, dd = runs["reference"], runs["fused_cl"]
    chk = {
        "train_hardware_verified_as_expected": tr["hardware"]["verified"] and tr["hardware"]["nvidia_smi"][0]["name"] == tr["hardware"]["expected_gpu"],
        "eval_hardware_verified_as_expected": ev["hardware"]["verified"] and ev["hardware"]["nvidia_smi"][0]["name"] == ev["hardware"]["expected_gpu"],
        "same_expected_gpu_train_eval": tr["hardware"]["expected_gpu"] == ev["hardware"]["expected_gpu"],
        "same_gpu_uuid_train_eval": tr["hardware"]["nvidia_smi"][0]["uuid"] == ev["hardware"]["nvidia_smi"][0]["uuid"],
        "epochs_8_both": a["epochs_completed"] == 8 and dd["epochs_completed"] == 8,
        "checkpoints_exist_both": a["final_checkpoint_exists"] and dd["final_checkpoint_exists"],
        "init_weights_identical": a["startup"]["init_weights_sha256"] == dd["startup"]["init_weights_sha256"],
        "first_batch_identical": a["first_step"]["first_batch_sha256"] == dd["first_step"]["first_batch_sha256"],
        "training_runtime_evidence_ok": a["runtime_evidence"]["evidence_ok"] and dd["runtime_evidence"]["evidence_ok"],
        "eval_runtime_evidence_ok": ev["all_evidence_ok"],
        "eval_checkpoint_hashes_match_training": ev["checkpoints"]["A"]["sha256"] == a["final_checkpoint_sha256"] and ev["checkpoints"]["D"]["sha256"] == dd["final_checkpoint_sha256"],
        "eval_1200_images": ev["validation_images"] == 1200,
        "code_hashes_equal_train_eval": tr["code_hashes"]["modules/modules.py"] == ev["code_hashes"]["modules/modules.py"],
        "all_evaluations_finite": all(v["finite"] for v in ev["evaluations"].values()),
    }
    out["integrity"][s] = {"checks": chk, "all_ok": all(chk.values()), "order": tr["order"], "gpu_uuid": tr["hardware"]["nvidia_smi"][0]["uuid"], "node": tr["hardware"]["node"],
                           "eval_node": ev["hardware"]["node"], "init_hash": a["startup"]["init_weights_sha256"], "first_batch_hash": a["first_step"]["first_batch_sha256"]}
    print(f"seed {s}: order {tr['order']} | node {tr['hardware']['node']} eval_node {ev['hardware']['node']} | GPU {tr['hardware']['nvidia_smi'][0]['uuid']} | all ok: {all(chk.values())} {[k for k, v in chk.items() if not v]}")


def diff(x, y, key, sub="full_validation"):
    return [per[s][1]["evaluations"][x][sub][key] - per[s][1]["evaluations"][y][sub][key] for s in seeds]


def pct(x, y, key="mse", sub="full_validation"):
    return [100 * (per[s][1]["evaluations"][x][sub][key] / per[s][1]["evaluations"][y][sub][key] - 1) for s in seeds]


print("\n== per-seed evaluation values (full validation set, pooled)")
print("%-5s %-4s %-13s %-13s %-13s %-13s" % ("seed", "eval", "MSE", "pooled PSNR", "per-img PSNR", "fixed-16 PSNR"))
for s in seeds:
    for e in ("E1", "E2", "E3", "E4"):
        v = per[s][1]["evaluations"][e]
        print("%-5s %-4s %-13.6e %-13.4f %-13.4f %-13.4f" % (s, e, v["full_validation"]["mse"], v["full_validation"]["psnr_pooled_db"], v["full_validation"]["psnr_per_image_mean_db"], v["fixed_batch"]["psnr_db"]))
    e1, e2 = per[s][1]["evaluations"]["E1"], per[s][1]["evaluations"]["E2"]
out["primary"] = {"endpoint": "E2 - E1, full-validation pooled PSNR (dB) at epoch 8", "psnr_pooled_db": ci(diff("E2", "E1", "psnr_pooled_db")), "mse_percent": ci(pct("E2", "E1")),
                  "mse_abs": ci(diff("E2", "E1", "mse"))}
print("\n== PRIMARY: E2 minus E1 (D checkpoint on full optimized path minus A checkpoint on original path)")
for k, lab in (("psnr_pooled_db", "pooled PSNR (dB)  [primary endpoint]"), ("mse_percent", "pooled MSE (% change)")):
    c = out["primary"][k]
    print("  %-38s per-seed %s" % (lab, "  ".join("%+.4f" % v for v in c["per_seed"])))
    if n > 1:
        print("  %-38s mean %+.4f | sample std %.4f | SE %.4f | 95%% CI [%+.4f, %+.4f] (t=%.3f, df=%d) | signs +%d/-%d" % ("", c["mean"], c["sample_std"], c["se"], c["ci95"][0], c["ci95"][1], c["t_crit"], n - 1, c["n_positive"], c["n_negative"]))
print("\n== DIAGNOSTICS (inference only, D checkpoint)")
for name, (x, y) in {"E2 - E3 (triton STN + fusion vs reference STN, channels-last on in both)": ("E2", "E3"), "E2 - E4 (channels-last on vs off, triton + fusion in both)": ("E2", "E4"),
                     "E3 - E1 (hybrid)": ("E3", "E1"), "E4 - E1": ("E4", "E1")}.items():
    out["diagnostics"][name] = {"psnr_pooled_db": ci(diff(x, y, "psnr_pooled_db")), "mse_percent": ci(pct(x, y))}
    c = out["diagnostics"][name]["psnr_pooled_db"]
    print("  %-72s PSNR per-seed %s | mean %+.5f" % (name, " ".join("%+.5f" % v for v in c["per_seed"]), c["mean"]) + ((" | 95%% CI [%+.5f, %+.5f]" % tuple(c["ci95"])) if n > 1 else ""))
print("\n== SUPPLEMENTARY (E2 - E1)")
for key, sub, lab in (("psnr_per_image_mean_db", "full_validation", "per-image mean PSNR (dB)"), ("psnr_db", "fixed_batch", "fixed 16-image PSNR (dB)")):
    c = ci(diff("E2", "E1", key, sub))
    out["supplementary"][lab] = c
    print("  %-28s per-seed %s | mean %+.4f" % (lab, " ".join("%+.4f" % v for v in c["per_seed"]), c["mean"]) + ((" | 95%% CI [%+.4f, %+.4f]" % tuple(c["ci95"])) if n > 1 else ""))
vl = []
for s in seeds:
    runs = {r["path"]: r for r in per[s][0]["runs"]}
    vl.append(runs["fused_cl"]["validation_loss"][-1] - runs["reference"]["validation_loss"][-1])
out["supplementary"]["logged_validation_loss_epoch8_D_minus_A (stochastic, each run's own path)"] = ci(vl)
print("  logged validation loss (epoch 8, D-A; STOCHASTIC ELBO of each run's own path, not a deterministic checkpoint metric): per-seed %s" % " ".join("%+.3f" % v for v in vl))
print("\n== PERFORMANCE (paired within seed, same verified A6000 and process)")
rows = []
for s in seeds:
    runs = {r["path"]: r for r in per[s][0]["runs"]}
    r = {}
    for p, lab in (("reference", "A"), ("fused_cl", "D")):
        ep = runs[p]["epochs"]
        step = st.fmean(e["median_ms"] for e in ep)
        r[lab] = {"step_ms": step, "throughput_img_s": 16000 / step, "epoch_wall_s": st.fmean(e["epoch_wall_s"] for e in ep), "total_wall_s": runs[p]["total_wall_s"],
                  "peak_alloc_mb": [min(e["peak_alloc_mb"] for e in ep), max(e["peak_alloc_mb"] for e in ep)], "peak_reserved_mb": [min(e["peak_reserved_mb"] for e in ep), max(e["peak_reserved_mb"] for e in ep)]}
    r["step_time_change_pct"] = 100 * (r["D"]["step_ms"] / r["A"]["step_ms"] - 1)
    r["throughput_change_pct"] = 100 * (r["D"]["throughput_img_s"] / r["A"]["throughput_img_s"] - 1)
    r["total_wall_change_pct"] = 100 * (r["D"]["total_wall_s"] / r["A"]["total_wall_s"] - 1)
    r["gpu_uuid"] = per[s][0]["hardware"]["nvidia_smi"][0]["uuid"]
    rows.append(r)
    print(f"seed {s}: A step {r['A']['step_ms']:.1f} ms ({r['A']['throughput_img_s']:.1f} img/s) | D {r['D']['step_ms']:.1f} ms ({r['D']['throughput_img_s']:.1f} img/s) | step {r['step_time_change_pct']:+.1f}% | run wall A {r['A']['total_wall_s']:.0f}s D {r['D']['total_wall_s']:.0f}s "
          f"({r['total_wall_change_pct']:+.1f}%) | peak alloc A {r['A']['peak_alloc_mb']} D {r['D']['peak_alloc_mb']} | peak reserved A {r['A']['peak_reserved_mb']} D {r['D']['peak_reserved_mb']}")
out["performance"] = {"per_seed": dict(zip(seeds, rows)), "step_time_change_pct": ci([r["step_time_change_pct"] for r in rows]), "throughput_change_pct": ci([r["throughput_change_pct"] for r in rows])}
if n > 1:
    c = out["performance"]["step_time_change_pct"]
    print("paired step-time change D vs A: mean %+.2f%% | sample std %.2f | range %+.2f..%+.2f" % (c["mean"], c["sample_std"], c["min"], c["max"]))
json.dump(out, open(out_json, "w"), indent=2)
print("\nwrote", out_json)
