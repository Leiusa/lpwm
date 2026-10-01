"""
Summary of the integrated-stack performance scan: A=reference, B=triton, C=triton+fusion, D=triton+fusion+channels_last.
Same protocol as summarize_scan.py, extended with per-round (pass) values (not just the mean), the direct cumulative
gain of D over A, and each step's own incremental gain (B-A, C-B, D-C).

    python summarize_full_stack.py DIR
"""
import glob
import json
import os
import statistics as st
import sys

D = sys.argv[1]
R = {}
for f in glob.glob(D + "/fs-b*-p*-*.json"):
    r = json.load(open(f))
    R[(r["batch_size"], os.path.basename(f).split("-")[2], r["path"])] = r
print(open(D + "/env.txt").read().split("hashes")[0].strip()[:900])
first = next(iter(R.values()))
print("\nprocesses:", len(R), "| GPU:", first["gpu"], "(%.0f MB) | host %s | job %s | commit %s clean=%s" % (
    first["gpu_total_mb"], first["host"], first["slurm_job_id"], first["repo"]["commit"][:8], first["repo"]["clean"]))
print("statuses:", {k: r["status"] for k, r in sorted(R.items())})
PATHS = ("reference", "triton", "fused", "fused_cl")
LABEL = {"reference": "A reference", "triton": "B triton", "fused": "C +fusion", "fused_cl": "D +fusion+cl"}
BS = sorted({k[0] for k in R})
summary = {"processes": len(R), "gpu": first["gpu"], "job": first["slurm_job_id"], "commit": first["repo"]["commit"], "by_batch": {}}

for B in BS:
    print("\n" + "=" * 110 + "\nB=%d" % B)
    ok = {p: [R[(B, t, p)] for t in ("p1", "p2") if (B, t, p) in R and R[(B, t, p)]["status"] == "ok"] for p in PATHS}
    for p in PATHS:
        for t in ("p1", "p2"):
            r = R.get((B, t, p))
            if r and r["status"] != "ok":
                print("  %-9s %s: %s at stage '%s' | allocator peak allocated %.0f MB / reserved %.0f MB of %.0f MB | %s" % (
                    p, t, r["status"], r.get("oom_stage"), r.get("allocator_peak_allocated_mb", 0), r.get("allocator_peak_reserved_mb", 0), r["gpu_total_mb"], r.get("oom_message", "")[:140]))
    same = {(r["initial_weights_sha256"][:12], r["first_batch_sha256"][:12], round(r["first_step_loss"], 3)) for v in ok.values() for r in v}
    print("  identical weights / first batch / first-step loss across all 4 paths:", len(same) == 1, sorted(same)[:2])
    print("  %-14s %-20s %-9s %-14s %-9s %-14s %-13s %-13s %s" % ("path", "wall ms p1 | p2", "spread", "CUDA-event ms", "img/s", "peak alloc MB", "peak resv MB", "masks-step MB", "path check"))
    agg = {}
    for p in PATHS:
        v = ok[p]
        if not v:
            print("  %-14s (no completed measurement)" % LABEL[p])
            continue
        w = [r["wall_ms_median"] for r in v]
        agg[p] = {"wall": st.mean(w), "wall_each": w, "img_s": st.mean(r["throughput_images_per_s"] for r in v),
                  "peak": st.mean(r["steady_peak_allocated_mb"] for r in v), "peak_each": [r["steady_peak_allocated_mb"] for r in v],
                  "resv": st.mean(r["steady_peak_reserved_mb"] for r in v), "resv_each": [r["steady_peak_reserved_mb"] for r in v],
                  "masks": st.mean(r["masks_step_peak_allocated_mb"] for r in v), "event": st.mean(r["event_ms_median"] for r in v)}
        print("  %-14s %-20s %-9s %-14.1f %-9.1f %-14.0f %-13.0f %-13.0f %s" % (
            LABEL[p], " | ".join("%.1f" % x for x in w), ("%.2f%%" % (100 * (max(w) - min(w)) / st.mean(w))) if len(w) > 1 else "n/a", agg[p]["event"],
            agg[p]["img_s"], agg[p]["peak"], agg[p]["resv"], agg[p]["masks"], all(r["path_check"]["ok"] for r in v)))
    print("\n  incremental gain (each step vs the step before it):")
    chain = (("triton", "reference", "B vs A (crop/paste)"), ("fused", "triton", "C vs B (+composite fusion)"), ("fused_cl", "fused", "D vs C (+particle_dec channels_last)"))
    for a, b, lab in chain:
        if a in agg and b in agg:
            print("    %-38s wall %.1f -> %.1f ms (%+.1f%%) | throughput %.1f -> %.1f img/s (%+.1f%%) | peak alloc %.0f -> %.0f MB (%+.1f%%, %+.0f MB) | peak reserved %.0f -> %.0f MB (%+.0f MB)" % (
                lab, agg[b]["wall"], agg[a]["wall"], 100 * (agg[a]["wall"] - agg[b]["wall"]) / agg[b]["wall"], agg[b]["img_s"], agg[a]["img_s"],
                100 * (agg[a]["img_s"] - agg[b]["img_s"]) / agg[b]["img_s"], agg[b]["peak"], agg[a]["peak"], 100 * (agg[a]["peak"] - agg[b]["peak"]) / agg[b]["peak"],
                agg[a]["peak"] - agg[b]["peak"], agg[b]["resv"], agg[a]["resv"], agg[a]["resv"] - agg[b]["resv"]))
    if "fused_cl" in agg and "reference" in agg:
        a, b = agg["fused_cl"], agg["reference"]
        print("\n  DIRECT cumulative gain, D (fused+cl) vs A (reference):")
        print("    wall %.1f -> %.1f ms (%+.1f%%, %.2fx) | throughput %.1f -> %.1f img/s (%+.1f%%, %.2fx) | peak alloc %.0f -> %.0f MB (%+.1f%%, %+.0f MB) | peak reserved %.0f -> %.0f MB (%+.0f MB)" % (
                b["wall"], a["wall"], 100 * (a["wall"] - b["wall"]) / b["wall"], b["wall"] / a["wall"], b["img_s"], a["img_s"],
                100 * (a["img_s"] - b["img_s"]) / b["img_s"], a["img_s"] / b["img_s"], b["peak"], a["peak"], 100 * (a["peak"] - b["peak"]) / b["peak"],
                a["peak"] - b["peak"], b["resv"], a["resv"], a["resv"] - b["resv"]))
    summary["by_batch"][B] = {"agg": {p: {k: v for k, v in agg[p].items() if k not in ()} for p in agg},
                              "path_check_all_ok": {p: all(r["path_check"]["ok"] for r in ok[p]) for p in ok if ok[p]}}

print("\n" + "=" * 110 + "\nSCALING of the mean values across the tested batch sizes (measured points only)")
for p in PATHS:
    pts = []
    for B in BS:
        v = [R[(B, t, p)] for t in ("p1", "p2") if (B, t, p) in R and R[(B, t, p)]["status"] == "ok"]
        if v:
            pts.append((B, st.mean(r["wall_ms_median"] for r in v), st.mean(r["throughput_images_per_s"] for r in v), st.mean(r["steady_peak_allocated_mb"] for r in v)))
    print("  %-14s " % LABEL[p] + " | ".join("B=%d: %.0f ms, %.1f img/s, %.0f MB" % x for x in pts))
    for (b0, w0, i0, m0), (b1, w1, i1, m1) in zip(pts, pts[1:]):
        print("                 B=%d->%d: ms/image %.1f -> %.1f, memory per added image %.0f MB, throughput x%.2f" % (b0, b1, w0 / b0, w1 / b1, (m1 - m0) / (b1 - b0), i1 / i0))

json.dump(summary, open(os.path.join(D, "full_stack_summary.json"), "w"), indent=2)
print("\nwrote", os.path.join(D, "full_stack_summary.json"))
