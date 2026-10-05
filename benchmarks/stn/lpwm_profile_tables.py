#!/usr/bin/env python
"""
Print the forward / backward per-module tables from an lpwm_profile.py result (lpwm_profile.json), without re-running.
Forward = CUDA-event time per module (hooks); backward = GPU kernel time attributed to the module that created each
autograd node. Kernel categories of jsons written before the annotation fix (kernel_categories, top_kernels,
forward_kernel_categories_by_module) included the mod:: scope annotations and are not printed here.

    python lpwm_profile_tables.py fixcheck_XXXX/eager_bwd/lpwm_profile.json [--top 30]
"""
import argparse
import json

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("json")
ap.add_argument("--top", type=int, default=30)
a = ap.parse_args()
r = json.load(open(a.json))
s = r["step_split_median_ms"]
print(f"{r['gpu']} | tf32_matmul={r['tf32_matmul']} cudnn_benchmark={r.get('cudnn_benchmark', False)} "
      f"cudnn_deterministic={r.get('cudnn_deterministic', True)} | step {s['step_ms']:.1f} = fwd {s['forward_ms']:.1f} + bwd "
      f"{s['backward_ms']:.1f} + opt {s['optimizer_ms']:.1f} ms (CUDA events, median)")
c = r.get("backward_attribution_coverage", {})
if c:
    print(f"backward kernel time {c['backward_kernel_ms_per_step']:.1f} ms/step: attributed {c['attributed_to_a_module_ms']:.1f}, "
          f"unattributed {c['unattributed_ms']:.1f} ({c['nodes']} nodes/step, {c['nodes_without_forward_match']} unmatched)")
    for k, v in list(c.get("unattributed_top_nodes_ms", {}).items())[:5]:
        print(f"   unattributed {v:7.1f} ms  {k}")
short = {"convolution": "conv", "matmul / linear": "matmul", "elementwise / reduction / other": "elemwise",
         "custom triton (STN / composite)": "triton", "grid_sample (reference STN)": "grid_smp", "attention": "attn"}
mods, bwd = r["modules"], r.get("backward_kernel_categories_by_module", {})
print(f"\n{'module':56s} {'fwd ms':>7s} {'bwd ms':>7s} {'bwd/fwd':>7s} " + " ".join(f"{v:>8s}" for v in short.values()))
rows = sorted(set(mods) | set(bwd), key=lambda n: -sum(bwd.get(n, {}).values()))
for n in rows[:a.top]:
    f = mods.get(n, {}).get("forward_ms_per_step", 0.0)
    b = sum(bwd.get(n, {}).values())
    print(f"{n[:56]:56s} {f:7.1f} {b:7.1f} {(b / f if f else float('nan')):7.2f} " +
          " ".join(f"{bwd.get(n, {}).get(k, 0.0):8.1f}" for k in short))
sp = r.get("dlp_vs_lpwm_split", {})
if "backward_kernel_ms_per_step" in sp:
    print()
    for (k, f), b in zip(sp["forward_ms_per_step"].items(), sp["backward_kernel_ms_per_step"].values()):
        print(f"{k:62s} fwd {f:7.1f} ms   bwd {b:7.1f} ms")
