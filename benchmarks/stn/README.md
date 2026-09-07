# `benchmarks/stn` — end-to-end LPWM benchmark harness

Measures the **whole model** with the STN stage swapped through the `lpwm_stn`
backend registry. It exists to keep isolated-kernel numbers and end-to-end
numbers apart: on LPWM they differ by an order of magnitude, and conflating them
produces false claims.

Nothing here modifies the repo, the model, the fixtures or the tolerances. The
backend is switched with `lpwm_stn.set_backend`; profiling scopes are installed
by patching the bindings in `modules.modules` for the duration of the profiled
step only, so no production file carries profiling labels.

## Files
| file | role |
|---|---|
| `build_model.py` | builds a `DLP` from a `configs/*.json` by signature introspection |
| `attribution.py` | profiler attribution — pure logic + a version-dependent adapter |
| `bench_e2e.py` | latency, throughput, peak memory, STN attribution; one backend per process |
| `test_attribution.py` | CPU-only tests for the attribution rules (no GPU) |
| `test_harness.py` | CPU-only tests for repo-root resolution and report schema |

## Metrics — named so they cannot be confused

| field | meaning |
|---|---|
| `D_clean_wall_ms` | median of the **uninstrumented** timed loop. The **only** basis for end-to-end speedup, and the only valid input to an Amdahl argument. |
| `D_prof_wall_ms` | wall time of the profiled step. Slower by construction; attribution only. |
| `total_device_kernel_ms` | summed self device time of CUDA kernels. **Not** wall time. |
| `stn_device_kernel_ms` | the STN part of the above. |
| `stn_share_of_device_kernel_time_pct` | ratio of the two. A share of **kernel time**, not wall time. |

No wall-time share is reported. Kernels overlap, so a defensible wall share needs
concurrency analysis this harness does not perform. Never derive an Amdahl
ceiling from `D_prof`.

## Attribution rules

Each rule fixes a defect found in the first GPU validation:

* **Device events only.** A CPU operator aggregate (`aten::grid_sampler_2d`) and
  the CUDA kernel it launches are the same work; the first harness summed both
  and reported roughly double. Every device event is now counted exactly once.
* **Explicit Triton allowlist**, never a `triton_` substring — which would
  classify unrelated Triton kernels as STN. Compiler signature suffixes are
  normalized; an arbitrary continuation is not. A crop/paste-looking kernel off
  the list lands in `suspicious_kernels_ms` rather than being counted.
* **Scope beats kernel name.** `create_masks_fast` has no kernel of its own — it
  calls the same `grid_sampler_2d` as crop and paste. Only an enclosing
  `record_function` scope can separate it, so masks are attributed by scope and
  never merged into `sampler`. Nested scopes: innermost wins.
* **Scopes also capture unnamed work**, notably the reference path's
  `affine_grid` build, which emits a generic elementwise kernel that no
  name-based rule could match.
* **Suspicious kernels — one policy.** A crop/paste-looking kernel that is not on
  the allowlist is *always* flagged in `suspicious_kernels_ms`. Whether it counts
  depends on evidence: inside a trusted STN scope it counts in that scope's
  bucket (the scope is direct evidence of what the caller was doing);
  with no trusted scope it is excluded, and the excluded subset is reported
  separately as `suspicious_excluded_ms`.

`attribution.py` splits pure logic from the profiler adapter precisely so the
rules above are unit-tested on CPU. The adapter (`records_from_profiler`) is the
one part CPU tests cannot cover; it reports which strategy it used, and that must
be checked on each GPU run.

## Scope health — how to tell if ancestry was lost

`unscoped_device_ms_informational` is **not** a health signal: convolutions and
GEMMs are legitimately unscoped, so it is large in a healthy run. Use instead:

| field | healthy run |
|---|---|
| `adapter_strategy` | `cpu_op_kernels` (the fallback may lose ancestry) |
| `scope_call_counts` | every op the graph uses is nonzero |
| `scoped_crop_device_event_count` / `..._paste_...` / `..._masks_...` | all > 0 |
| `device_kernel_ms_by_scope["masks"]` | > 0 |
| `known_sampler_forward_events_without_scope` | **0** — the ancestry-loss signal |
| `known_sampler_backward_events_without_scope` | **may be > 0**, see below |

**Forward and backward are not the same signal.** `ScopeProbe` wraps the
Python-level `stn_crop` / `stn_paste` calls, so a scope has already exited by the
time autograd runs its backward. A reference `grid_sampler_2d_backward_kernel` is
therefore *expected* to be unscoped during training and must never be read as
ancestry loss. Per configuration:

| run | forward unscoped | backward unscoped |
|---|---|---|
| reference, inference | must be 0 | must be 0 (no backward exists) |
| reference, training | must be 0 | expected > 0 |
| triton, either | must be 0 | reference sampler normally absent entirely; Triton backward kernels are classified by the exact allowlist |

### Adapter invariants (version-stable)

Adapter diagnostics are reported separately from attribution counters and must
not be conflated. Both strategies emit device records only, so
`skipped_non_device_count` is normally 0 even though CPU aggregates *were*
excluded — it is not the proof. Under `cpu_op_kernels`, require:

* `linked_kernel_count > 0`
* `emitted_device_record_count == linked_kernel_count`
* `emitted_cpu_record_count == 0`
* each linked kernel emitted exactly once

`cpu_events_with_device_aggregate_excluded` and `cpu_operators_with_kernels`
remain useful diagnostics, but the first depends on the PyTorch build exposing
`device_time_total` on CPU rows, so a zero there is **not** a failure when the
invariants above hold.

## Protocol
* one **process per backend** — a backend that OOMs must not poison the other
* identical seed for weight init and input
* `torch.cuda.synchronize()` around every measured region
* warmup excluded, so Triton JIT compilation never enters a steady-state number
* peak memory on a dedicated step with `reset_peak_memory_stats()` first
* `profile_memory=True` on the profiled step

Peak memory and memory events are **impact measurements and supporting
evidence**. Neither proves that a particular named tensor was or was not
allocated; that requires structural evidence about the code path taken.

Memory rows are separated deliberately: `top_self_allocations` ranks **positive
self**-device allocations (aggregate rows include children and double-count
nested operators), `top_self_deallocations` keeps frees apart from allocations,
and `top_aggregate_rows_informational` is labelled as such.

## Usage
```bash
for bk in reference triton; do
  python benchmarks/stn/bench_e2e.py --repo-root /workspace/lpwm \
    --config configs/bair.json --backend $bk --batch-size 1 \
    --out /workspace/experiments/<stamp>/bair_bs1_$bk.json
done
```
`--repo-root` is explicit so a harness copied outside the repository still
resolves imports, configs and git provenance against the real tree. The tests
take the same root, via `--repo-root` or `LPWM_REPO_ROOT`:

```bash
python /workspace/harness_val/<timestamp>/stn/test_attribution.py
python /workspace/harness_val/<timestamp>/stn/test_harness.py --repo-root /workspace/lpwm
```

Each revision is copied to its own timestamped directory on the network volume
rather than overwriting the last, so the code behind every profiler result stays
recoverable. The exact harness path is recorded in `COMMANDS.txt` and in the
result JSON.

Crop glimpse size comes from `cfg["patch_size"]` when present; BAIR sets 8 while
`round(anchor_s * (image_size - 1))` would give 16, so the derivation is only a
documented fallback for configs that omit the key.

## Known limitation
Within a scope, `affine_grid` construction and `grid_sample` cannot be separated:
the reference `affine_grid_sample` is `@torch.jit.script`, so a Python-level
wrapper cannot reach inside it. Splitting them would require editing a production
file. The scope-level total is correct; only the sub-split is unavailable.


## Validated on RTX 4090 (2026-09-07, bair bs=1, pod `lpwm-stn-validation-4`)

Harness validation run 2. Results below are the ones that do **not** depend on
profiler scope attribution, so they stand on their own:

| metric | reference | triton |
|---|---:|---:|
| inference `D_clean_wall_ms` | 87.581 | 81.687 |
| training `D_clean_wall_ms` | 280.384 | 258.442 |
| inference peak alloc | 1667.2 MB | 1666.7 MB |
| training peak alloc | 18493.1 MB | 17119.4 MB |

Iteration spread 0.21–0.90%. These are 10-iteration harness-validation numbers;
they do not replace the pinned full baseline.

Also validated: 47/47 CPU tests with zero stderr; `adapter_strategy =
cpu_op_kernels` on all four profiles with `emitted_device_record_count ==
linked_kernel_count` and `emitted_cpu_record_count == 0`; unscoped **forward**
sampler count 0 everywhere and unscoped **backward** > 0 only in reference
training, as predicted; `suspicious_kernels_ms` empty; grad-x expectations met
for both backends; provenance, `patch_size = 8` from `config:patch_size`, and
schema all correct with no legacy fields.

### Retracted: the "scope leakage" finding

An earlier reading of this run flagged `ampere_sgemm_128x128_tn` and a generic
`direct_copy` elementwise kernel inside the STN scopes as scope over-capture.
**That was wrong, and is retracted.**

`F.affine_grid` is not a single kernel. Profiling it on CPU shows it dispatches
`aten::bmm`, `aten::linspace`, `aten::fill_` and `aten::copy_` before
`grid_sample` runs at all. On CUDA those become cuBLAS `ampere_sgemm_*`,
`linspace_cuda_out`, `FillFunctor` and `direct_copy` — exactly the kernels that
were flagged. A cuBLAS kernel name is shared across unrelated GEMMs, so the same
name appearing outside the scope proves nothing.

The arithmetic confirms it. In the Triton inference profile, the crop and paste
buckets total 0.456642 ms against 0.452130 ms of `_stn_*` kernels — leaving
**4.5 µs** of non-Triton time in those scopes — while the generic kernels total
4.875569 ms against a masks bucket of 4.871057 ms, the same 4.5 µs apart. Every
generic kernel is inside the masks scope, and the masks scope is
`create_masks_fast`, which the Triton backend does not implement: the registry
resolves it to `lpwm_stn.reference`, which runs the full affine-grid chain.

### Masks is now the dominant unoptimized STN operation

With crop and paste accelerated, the Triton inference profile is:

| scope | device kernel time |
|---|---:|
| masks (reference fallback) | **4.871 ms** |
| paste (Triton) | 0.432 ms |
| crop (Triton) | 0.024 ms |

`create_masks_fast` is ~10x the accelerated ops combined, and most of its cost is
`affine_grid` grid construction rather than sampling.

### Optional future diagnostic

A `kernel_ms_by_scope` cross-tab (per-scope, per-kernel time) would have settled
the question above from the run's own output instead of by elimination. It is a
small additive change and deliberately **not** a blocking requirement.
