"""
Corrected summary of the full-stack quality validation. Reads the UNMODIFIED server reports and a separate metadata
record (training/eval hosts, code versions; null = unknown) and prints/writes:

  * MAIN table: (1) A checkpoint, original implementation  vs  (2) D checkpoint, full optimized implementation
  * auxiliary diagnostics: (3) D checkpoint on the hybrid path (reference STN, no fusion, channels-last STILL ON),
    reported as (3)-(1) and (2)-(3); it is not a pure Reference evaluation
  * training speed/memory per seed, labelled with the GPU that seed actually trained on; nothing is pooled across GPUs

No threshold is applied; differences are reported.

    python summarize_full_stack_quality_corrected.py RAW_DIR METADATA.json OUT.json
"""
import json
import os
import statistics as st
import sys

raw, meta_path, out_json = sys.argv[1:4]
meta = json.load(open(meta_path))
SEEDS = (0, 1, 2)
data = {}
for s in SEEDS:
    rep = json.load(open(os.path.join(raw, f"s{s}_recovered_train_report.json")))
    runs = {r["path"]: r for r in json.load(open(os.path.join(raw, f"s{s}_train_runs.json")))}
    rc = rep["reconstruction"]["runs"]
    dc = rep["deployment_check"]
    assert dc.get("failed") is False
    ev = {"1": {"full": rc["reference_p1"]["full_validation_set"], "fixed": rc["reference_p1"]["fixed_batch"]},
          "2": {"full": dc["full_validation_set"], "fixed": dc["fixed_batch"]},
          "3": {"full": rc["fused_cl_p1"]["full_validation_set"], "fixed": rc["fused_cl_p1"]["fixed_batch"]}}
    for k in ev:
        assert ev[k]["full"]["images"] == 1200
    data[s] = {"ev": ev, "runs": runs, "wrel": rep["reconstruction"]["weight_relative_l2_between_final_models"]["fused_cl_vs_reference"]}


def pct(a, b):
    return 100 * (a - b) / b


def stats(v):
    return {"mean": st.fmean(v), "sample_std": st.stdev(v), "min": min(v), "max": max(v), "per_seed": v}


out = {"main_1_vs_2": {}, "aux_3_vs_1": {}, "aux_2_vs_3": {}, "training_perf_per_seed": {}, "val_loss": {}}
print("evaluation definitions:")
for k, v in meta["evaluation_definitions"].items():
    print(f"  {k}: {v}")
print("\nper-seed absolute values (1,200-image validation set; fixed = the 16-image fixed batch)")
print("%-5s %-38s %-13s %-10s %-13s %-10s" % ("seed", "evaluation", "full MSE", "full PSNR", "fixed MSE", "fixed PSNR"))
LAB = {"1": "(1) A ckpt, original impl", "2": "(2) D ckpt, full optimized impl", "3": "(3) D ckpt, HYBRID (ref STN, cl on)"}
for s in SEEDS:
    for k in ("1", "2", "3"):
        f, x = data[s]["ev"][k]["full"], data[s]["ev"][k]["fixed"]
        print("%-5s %-38s %-13.6e %-10.4f %-13.6e %-10.4f" % (s, LAB[k], f["mse"], f["psnr_db"], x["mse"], x["psnr_db"]))


def paired(a, b):
    rows = {"full_mse_pct": [], "full_psnr_db": [], "fixed_mse_pct": [], "fixed_psnr_db": []}
    for s in SEEDS:
        fa, fb = data[s]["ev"][a]["full"], data[s]["ev"][b]["full"]
        xa, xb = data[s]["ev"][a]["fixed"], data[s]["ev"][b]["fixed"]
        rows["full_mse_pct"].append(pct(fa["mse"], fb["mse"]))
        rows["full_psnr_db"].append(fa["psnr_db"] - fb["psnr_db"])
        rows["fixed_mse_pct"].append(pct(xa["mse"], xb["mse"]))
        rows["fixed_psnr_db"].append(xa["psnr_db"] - xb["psnr_db"])
    return {k: stats(v) for k, v in rows.items()}


for name, key, (a, b) in (("MAIN: (2) minus (1)", "main_1_vs_2", ("2", "1")), ("AUX: (3) minus (1), hybrid path", "aux_3_vs_1", ("3", "1")),
                          ("AUX: (2) minus (3), implementation at inference", "aux_2_vs_3", ("2", "3"))):
    out[key] = paired(a, b)
    print("\n== " + name)
    for m, lab in (("full_mse_pct", "full-val MSE %"), ("full_psnr_db", "full-val PSNR dB"), ("fixed_mse_pct", "fixed-16 MSE %"), ("fixed_psnr_db", "fixed-16 PSNR dB")):
        r = out[key][m]
        print("  %-18s %s | mean %+.4f | sample std %.4f | range %+.4f .. %+.4f" % (lab, "  ".join("s%d %+.4f" % (s, x) for s, x in zip(SEEDS, r["per_seed"])), r["mean"], r["sample_std"], r["min"], r["max"]))

print("\n== validation loss at epoch 8 (own-path, stochastic ELBO from the training logs; no hybrid counterpart)")
vl = []
for s in SEEDS:
    a, d = data[s]["runs"]["reference"]["validation_loss"][-1], data[s]["runs"]["fused_cl"]["validation_loss"][-1]
    vl.append(d - a)
    print("  seed %d: A %.3f | D %.3f | D-A %+.3f (%+.2f%%)" % (s, a, d, d - a, pct(d, a)))
out["val_loss"] = stats(vl)
print("  D-A mean %+.4f | sample std %.4f | range %+.4f .. %+.4f" % (st.fmean(vl), st.stdev(vl), min(vl), max(vl)))

print("\n== training speed and memory PER SEED, on the GPU that seed actually trained on (never pooled across GPUs)")
for s in SEEDS:
    m = meta["seeds"][str(s)]
    print("seed %d | training job %s | host %s | GPU %s | order %s" % (s, m["training_job"], m["training_host"], m["training_gpu"], m["order"]))
    rows = {}
    for p, lab in (("reference", "A"), ("fused_cl", "D")):
        ep = data[s]["runs"][p]["epochs"]
        rows[p] = {"step_ms_mean": st.fmean(e["median_ms"] for e in ep), "step_ms_range": [min(e["median_ms"] for e in ep), max(e["median_ms"] for e in ep)],
                   "epoch_wall_s_mean": st.fmean(e["epoch_wall_s"] for e in ep), "peak_alloc_mb_range": [min(e["peak_alloc_mb"] for e in ep), max(e["peak_alloc_mb"] for e in ep)],
                   "peak_reserved_mb_range": [min(e["peak_reserved_mb"] for e in ep), max(e["peak_reserved_mb"] for e in ep)]}
        r = rows[p]
        print("   %s: step %.1f ms (per-epoch %.1f-%.1f) | epoch wall %.1f s | peak alloc %d-%d MB | peak reserved %d-%d MB" % (
            lab, r["step_ms_mean"], *r["step_ms_range"], r["epoch_wall_s_mean"], *r["peak_alloc_mb_range"], *r["peak_reserved_mb_range"]))
    rows["D_vs_A_step_pct"] = pct(rows["fused_cl"]["step_ms_mean"], rows["reference"]["step_ms_mean"])
    rows["D_vs_A_epoch_wall_pct"] = pct(rows["fused_cl"]["epoch_wall_s_mean"], rows["reference"]["epoch_wall_s_mean"])
    print("   D vs A on this GPU: step %+.1f%% | epoch wall %+.1f%%" % (rows["D_vs_A_step_pct"], rows["D_vs_A_epoch_wall_pct"]))
    out["training_perf_per_seed"][s] = {"gpu": m["training_gpu"], "host": m["training_host"], **rows}
print("\nfinal-weight relative L2, D vs A:", {s: data[s]["wrel"] for s in SEEDS})
out["weight_rel_l2_D_vs_A"] = {s: data[s]["wrel"] for s in SEEDS}
json.dump(out, open(out_json, "w"), indent=2)
print("wrote", out_json)
