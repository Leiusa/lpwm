import glob
import json
import os
import statistics as st
import sys

D = sys.argv[1]
R = {}
for f in glob.glob(D + "/scan-b*-p*-*.json"):
    r = json.load(open(f))
    R[(r["batch_size"], os.path.basename(f).split("-")[2], r["path"])] = r
print(open(D + "/env.txt").read().split("hashes")[0].strip()[:900])
first = next(iter(R.values()))
print("\nprocesses:", len(R), "| GPU:", first["gpu"], "(%.0f MB) | host %s | job %s | commit %s clean=%s" % (
    first["gpu_total_mb"], first["host"], first["slurm_job_id"], first["repo"]["commit"][:8], first["repo"]["clean"]))
print("statuses:", {k: r["status"] for k, r in sorted(R.items())})
PATHS = ("reference", "triton", "fused")
BS = sorted({k[0] for k in R})

for B in BS:
    print("\n" + "=" * 100 + "\nB=%d" % B)
    ok = {p: [R[(B, t, p)] for t in ("p1", "p2") if (B, t, p) in R and R[(B, t, p)]["status"] == "ok"] for p in PATHS}
    for p in PATHS:
        for t in ("p1", "p2"):
            r = R.get((B, t, p))
            if r and r["status"] != "ok":
                print("  %-9s %s: %s at stage '%s' | allocator peak allocated %.0f MB / reserved %.0f MB of %.0f MB | %s" % (
                    p, t, r["status"], r.get("oom_stage"), r.get("allocator_peak_allocated_mb", 0), r.get("allocator_peak_reserved_mb", 0), r["gpu_total_mb"], r.get("oom_message", "")[:140]))
    same = {(r["initial_weights_sha256"][:12], r["first_batch_sha256"][:12], round(r["first_step_loss"], 3)) for v in ok.values() for r in v}
    print("  identical weights / first batch / first-step loss across the measured paths:", len(same) == 1, sorted(same)[:2])
    print("  %-10s %-16s %-8s %-14s %-11s %-9s %-14s %-14s %-13s %s" % ("path", "wall ms p1|p2", "spread", "CUDA-event ms", "fetch ms", "img/s", "peak alloc MB", "masks-step MB", "peak resv MB", "path check"))
    agg = {}
    for p in PATHS:
        v = ok[p]
        if not v:
            print("  %-10s (no completed measurement)" % p)
            continue
        w = [r["wall_ms_median"] for r in v]
        agg[p] = {"wall": st.mean(w), "img_s": st.mean(r["throughput_images_per_s"] for r in v), "peak": st.mean(r["steady_peak_allocated_mb"] for r in v),
                  "masks": st.mean(r["masks_step_peak_allocated_mb"] for r in v), "event": st.mean(r["event_ms_median"] for r in v)}
        print("  %-10s %-16s %-8s %-14.1f %-11.2f %-9.1f %-14.0f %-14.0f %-13.0f %s" % (
            p, " | ".join("%.1f" % x for x in w), ("%.1f%%" % (100 * (max(w) - min(w)) / st.mean(w))) if len(w) > 1 else "n/a", agg[p]["event"],
            st.mean(r["fetch_ms_median"] for r in v), agg[p]["img_s"], agg[p]["peak"], agg[p]["masks"], st.mean(r["steady_peak_reserved_mb"] for r in v),
            all(r["path_check"]["ok"] for r in v)))
    for a, b in (("triton", "reference"), ("fused", "reference"), ("fused", "triton")):
        if a in agg and b in agg:
            print("  %-18s wall %.1f -> %.1f ms (%+.1f%%) | throughput %.1f -> %.1f img/s (%+.1f%%) | peak %.0f -> %.0f MB (%+.1f%%, %+.0f MB) | plotting-batch peak %.0f -> %.0f MB" % (
                a + " vs " + b, agg[b]["wall"], agg[a]["wall"], 100 * (agg[a]["wall"] - agg[b]["wall"]) / agg[b]["wall"], agg[b]["img_s"], agg[a]["img_s"],
                100 * (agg[a]["img_s"] - agg[b]["img_s"]) / agg[b]["img_s"], agg[b]["peak"], agg[a]["peak"], 100 * (agg[a]["peak"] - agg[b]["peak"]) / agg[b]["peak"],
                agg[a]["peak"] - agg[b]["peak"], agg[b]["masks"], agg[a]["masks"]))

print("\n" + "=" * 100 + "\nSCALING of the mean values across the tested batch sizes (measured points only)")
for p in PATHS:
    pts = []
    for B in BS:
        v = [R[(B, t, p)] for t in ("p1", "p2") if (B, t, p) in R and R[(B, t, p)]["status"] == "ok"]
        if v:
            pts.append((B, st.mean(r["wall_ms_median"] for r in v), st.mean(r["throughput_images_per_s"] for r in v), st.mean(r["steady_peak_allocated_mb"] for r in v)))
    print("  %-10s " % p + " | ".join("B=%d: %.0f ms, %.1f img/s, %.0f MB" % x for x in pts))
    for (b0, w0, i0, m0), (b1, w1, i1, m1) in zip(pts, pts[1:]):
        print("             B=%d->%d: ms/image %.1f -> %.1f, memory per added image %.0f MB, throughput x%.2f" % (b0, b1, w0 / b0, w1 / b1, (m1 - m0) / (b1 - b0), i1 / i0))
