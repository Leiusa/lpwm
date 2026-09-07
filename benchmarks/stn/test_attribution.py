#!/usr/bin/env python
"""
CPU-only tests for the STN profiler attribution logic.

No GPU, no torch profiler -- synthetic :class:`EventRecord` objects stand in for
real profiler output, which is exactly the point: the classification and
bucketing rules are testable without burning GPU time, and each test pins one of
the defects found in the first validation run.

    python benchmarks/stn/test_attribution.py
    pytest benchmarks/stn/test_attribution.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attribution import (  # noqa: E402
    SCOPE_PREFIX, STN_TRITON_KERNELS, EventRecord, attribute, classify_kernel,
    evaluate_gradx_expectation, normalize_kernel, records_from_profiler,
)

CROP = f"{SCOPE_PREFIX}crop"
PASTE = f"{SCOPE_PREFIX}paste"
MASKS = f"{SCOPE_PREFIX}masks_fast"
SAMPLER_K = "void at::native::(anonymous namespace)::grid_sampler_2d_kernel<float, int>(...)"
SAMPLER_BWD_K = "void at::native::(anonymous namespace)::grid_sampler_2d_backward_kernel<float, int>(...)"


def test_d1_no_cpu_aggregate_double_counting():
    """D1: the aten:: aggregate and its CUDA child must count once, not twice."""
    recs = [
        EventRecord("aten::grid_sampler_2d", is_device=False, self_device_us=1083.0, scopes=(CROP,)),
        EventRecord(SAMPLER_K, is_device=True, self_device_us=1083.0, scopes=(CROP,)),
    ]
    a = attribute(recs)
    assert a.device_event_count == 1, a.device_event_count
    assert a.skipped_non_device_count == 1
    assert abs(a.total_device_kernel_ms - 1.083) < 1e-9, a.total_device_kernel_ms
    assert abs(a.buckets_ms["crop"] - 1.083) < 1e-9
    # the pre-fix harness would have reported 2.166 ms here
    assert a.total_device_kernel_ms < 2.0


def test_d2_exact_triton_allowlist():
    """Every named STN kernel classifies; nothing else Triton-shaped does."""
    for k in STN_TRITON_KERNELS:
        bucket, suspicious = classify_kernel(k)
        assert bucket in ("crop", "paste"), (k, bucket)
        assert not suspicious, k
    assert classify_kernel("_stn_crop_forward_kernel")[0] == "crop"
    assert classify_kernel("_stn_paste_grad_patches_kernel")[0] == "paste"


def test_d2_unrelated_triton_kernels_stay_unmatched():
    """A `triton_` substring must NOT imply STN."""
    for name in ("triton_poi_fused_add_0", "triton_red_fused_softmax_3", "triton_unrelated_kernel"):
        bucket, suspicious = classify_kernel(name)
        assert bucket is None, (name, bucket)
        assert not suspicious, name
    a = attribute([EventRecord("triton_poi_fused_add_0", True, 5000.0)])
    assert a.stn_device_kernel_ms == 0.0
    assert "triton_poi_fused_add_0" in a.unmatched_kernels_ms


def test_d2_suspicious_kernels_reported_not_counted():
    """A crop/paste-looking kernel off the allowlist is surfaced, never bucketed."""
    a = attribute([EventRecord("_stn_crop_experimental_kernel", True, 1000.0)])
    assert a.stn_device_kernel_ms == 0.0
    assert "_stn_crop_experimental_kernel" in a.suspicious_kernels_ms
    assert "_stn_crop_experimental_kernel" in a.unmatched_kernels_ms


def test_compiler_suffix_normalizes_but_not_arbitrary_names():
    assert normalize_kernel("_stn_crop_forward_kernel_0d1d2d3de") == "_stn_crop_forward_kernel"
    assert normalize_kernel("_stn_paste_forward_kernel.kd") == "_stn_paste_forward_kernel"
    # an arbitrary continuation must NOT collapse onto an allowlisted name
    assert normalize_kernel("_stn_crop_forward_kernel_v2_experimental") != "_stn_crop_forward_kernel"
    assert classify_kernel("_stn_crop_forward_kernel_v2_experimental")[1] is True


def test_d3_masks_attributed_to_masks_not_sampler():
    """The decisive D3 case: masks run the SAME kernel as crop; only scope separates them."""
    recs = [
        EventRecord(SAMPLER_K, True, 4920.0, scopes=(MASKS,)),
        EventRecord(SAMPLER_K, True, 1580.0, scopes=(CROP,)),
    ]
    a = attribute(recs)
    assert abs(a.buckets_ms["masks"] - 4.92) < 1e-9, a.buckets_ms
    assert abs(a.buckets_ms["crop"] - 1.58) < 1e-9, a.buckets_ms
    assert "sampler" not in a.buckets_ms, a.buckets_ms


def test_d4_affine_grid_elementwise_captured_by_scope():
    """A generic elementwise kernel is unattributable by name but caught by scope."""
    generic = "void at::native::elementwise_kernel<128, 2, ...>(int, ...)"
    assert classify_kernel(generic)[0] is None
    a = attribute([EventRecord(generic, True, 800.0, scopes=(CROP,))])
    assert abs(a.buckets_ms["crop"] - 0.8) < 1e-9
    assert not a.unmatched_kernels_ms


def test_nested_scope_precedence_innermost_wins():
    a = attribute([EventRecord(SAMPLER_K, True, 1000.0, scopes=(CROP, MASKS))])
    assert a.buckets_ms == {"masks": 1.0}, a.buckets_ms
    b = attribute([EventRecord(SAMPLER_K, True, 1000.0, scopes=(MASKS, CROP))])
    assert b.buckets_ms == {"crop": 1.0}, b.buckets_ms


def test_scope_overrides_kernel_name():
    """A paste kernel inside a masks scope is masks work; scope beats kernel name."""
    a = attribute([EventRecord("_stn_paste_forward_kernel", True, 500.0, scopes=(MASKS,))])
    assert a.buckets_ms == {"masks": 0.5}, a.buckets_ms


def test_unscoped_device_events_reported():
    recs = [
        EventRecord("aten_kernel_x", True, 1000.0),
        EventRecord(SAMPLER_K, True, 500.0, scopes=(CROP,)),
    ]
    a = attribute(recs)
    assert abs(a.unscoped_device_ms - 1.0) < 1e-9
    assert abs(a.total_device_kernel_ms - 1.5) < 1e-9


def test_denominator_and_share():
    recs = [
        EventRecord(SAMPLER_K, True, 2000.0, scopes=(CROP,)),
        EventRecord("aten::mm_kernel", True, 8000.0),
    ]
    a = attribute(recs)
    assert abs(a.total_device_kernel_ms - 10.0) < 1e-9
    assert abs(a.stn_device_kernel_ms - 2.0) < 1e-9
    assert abs(a.stn_share_of_device_kernel_time_pct - 20.0) < 1e-9


def test_every_device_event_counted_exactly_once():
    recs = [EventRecord(f"k{i}", True, 100.0, scopes=(CROP,) if i % 2 else ()) for i in range(10)]
    a = attribute(recs)
    assert a.device_event_count == 10
    assert abs(a.total_device_kernel_ms - 1.0) < 1e-9
    assert abs(sum(a.buckets_ms.values()) + sum(a.unmatched_kernels_ms.values())
               - a.total_device_kernel_ms) < 1e-9


def test_json_schema_field_names():
    """Field names are part of the contract downstream analysis reads."""
    d = attribute([EventRecord(SAMPLER_K, True, 1000.0, scopes=(CROP,))]).as_dict()
    for key in ("total_device_kernel_ms", "stn_device_kernel_ms",
                "stn_share_of_device_kernel_time_pct", "buckets_ms", "matched_kernels_ms",
                "unmatched_kernels_ms", "suspicious_kernels_ms", "suspicious_excluded_ms",
                "unscoped_device_ms_informational", "device_event_count",
                "skipped_non_device_count", "device_event_count_by_scope",
                "device_kernel_ms_by_scope", "scoped_crop_device_event_count",
                "scoped_paste_device_event_count", "scoped_masks_device_event_count",
                "known_sampler_forward_events_without_scope",
                "known_sampler_backward_events_without_scope"):
        assert key in d, key
    # the discredited names must be gone
    for gone in ("total_cuda_ms", "stn_ms", "stn_share_of_cuda_pct", "stn_share_of_wall_pct",
                 "top_unmatched_kernels_ms"):
        assert gone not in d, gone


def test_backward_kernels_classify():
    assert classify_kernel(SAMPLER_BWD_K)[0] == "sampler"
    assert classify_kernel("_stn_crop_grad_x_kernel")[0] == "crop"
    assert classify_kernel("_stn_paste_grad_params_kernel")[0] == "paste"


# --------------------------------------------------------------------------- #
# B1: adapter diagnostics must not be conflated with attribution counters
# --------------------------------------------------------------------------- #
class _FakeKernel:
    def __init__(self, name, duration):
        self.name, self.duration = name, duration


class _FakeEvent:
    """Minimal stand-in for a profiler FunctionEvent."""

    def __init__(self, name, device_type, kernels=(), device_total=0.0, parent=None):
        self.name = self.key = name
        self.device_type = device_type
        self.kernels = list(kernels)
        self.device_time_total = device_total
        self.cpu_parent = parent
        self.self_device_time_total = device_total


class _FakeProf:
    def __init__(self, events):
        self._events = events

    def events(self):
        return self._events


def _cuda_device_type():
    from torch.autograd import DeviceType
    return DeviceType.CUDA, DeviceType.CPU


def test_b1_adapter_diagnostics_prove_cpu_aggregates_excluded():
    """skipped_non_device_count is ZERO here even though CPU aggregates were
    excluded -- which is exactly why it cannot be the acceptance signal."""
    cuda, cpu = _cuda_device_type()
    scope = _FakeEvent(f"{SCOPE_PREFIX}crop", cpu)
    op = _FakeEvent("aten::grid_sampler_2d", cpu,
                    kernels=[_FakeKernel(SAMPLER_K, 1083.0)], device_total=1083.0, parent=scope)
    recs, diag = records_from_profiler(_FakeProf([scope, op]))

    assert diag["strategy"] == "cpu_op_kernels"
    assert diag["raw_event_count"] == 2
    assert diag["cpu_event_count_seen"] == 2
    assert diag["cpu_events_with_device_aggregate_excluded"] == 1     # the real proof
    assert diag["linked_kernel_count"] == 1
    assert diag["emitted_device_record_count"] == 1

    a = attribute(recs)
    assert a.device_event_count == 1
    assert a.skipped_non_device_count == 0, "adapter emits device records only"
    assert abs(a.total_device_kernel_ms - 1.083) < 1e-9
    assert a.buckets_ms == {"crop": 1.083}, a.buckets_ms


def test_b1_diagnostics_and_attribution_counters_are_distinct_fields():
    cuda, cpu = _cuda_device_type()
    op = _FakeEvent("aten::mm", cpu, kernels=[_FakeKernel("gemm", 500.0)], device_total=500.0)
    recs, diag = records_from_profiler(_FakeProf([op]))
    a = attribute(recs).as_dict()
    assert set(diag) & set(a) == set(), f"diagnostic/attribution field collision: {set(diag) & set(a)}"


def test_b1_scope_ancestry_recovered_from_parent_chain():
    cuda, cpu = _cuda_device_type()
    outer = _FakeEvent(f"{SCOPE_PREFIX}masks_fast", cpu)
    op = _FakeEvent("aten::grid_sampler_2d", cpu,
                    kernels=[_FakeKernel(SAMPLER_K, 4920.0)], device_total=4920.0, parent=outer)
    recs, _ = records_from_profiler(_FakeProf([outer, op]))
    assert recs[0].scopes == (f"{SCOPE_PREFIX}masks_fast",)
    assert attribute(recs).buckets_ms == {"masks": 4.92}


# --------------------------------------------------------------------------- #
# C4: version-stable adapter invariants
# --------------------------------------------------------------------------- #
def test_c4_linked_kernel_invariants_hold_without_the_version_dependent_property():
    """The acceptance signal must survive a PyTorch build that never exposes
    `device_time_total` on CPU rows -- then
    cpu_events_with_device_aggregate_excluded is 0 while behaviour is correct."""
    cuda, cpu = _cuda_device_type()
    scope = _FakeEvent(f"{SCOPE_PREFIX}crop", cpu)
    op = _FakeEvent("aten::grid_sampler_2d", cpu,
                    kernels=[_FakeKernel(SAMPLER_K, 1083.0)],
                    device_total=0.0,           # property absent on this build
                    parent=scope)
    recs, diag = records_from_profiler(_FakeProf([scope, op]))

    assert diag["cpu_events_with_device_aggregate_excluded"] == 0, "version-dependent, may be 0"
    # the stable invariants still prove correctness:
    assert diag["linked_kernel_count"] == 1
    assert diag["emitted_device_record_count"] == diag["linked_kernel_count"]
    assert diag["emitted_cpu_record_count"] == 0
    assert diag["cpu_operators_with_kernels"] == 1
    assert all(r.is_device for r in recs)


def test_c4_each_linked_kernel_emitted_exactly_once():
    cuda, cpu = _cuda_device_type()
    ops = [
        _FakeEvent("aten::a", cpu, kernels=[_FakeKernel("k1", 10.0), _FakeKernel("k2", 20.0)]),
        _FakeEvent("aten::b", cpu, kernels=[_FakeKernel("k3", 30.0)]),
    ]
    recs, diag = records_from_profiler(_FakeProf(ops))
    assert diag["linked_kernel_count"] == 3
    assert diag["emitted_device_record_count"] == 3
    assert diag["cpu_operators_with_kernels"] == 2
    assert sorted(r.name for r in recs) == ["k1", "k2", "k3"]
    a = attribute(recs)
    assert abs(a.total_device_kernel_ms - 0.060) < 1e-9


def test_c4_cpu_operators_without_kernels_are_not_counted():
    cuda, cpu = _cuda_device_type()
    ops = [_FakeEvent("aten::empty", cpu), _FakeEvent("aten::mm", cpu,
                                                      kernels=[_FakeKernel("gemm", 5.0)])]
    _, diag = records_from_profiler(_FakeProf(ops))
    assert diag["cpu_event_count_seen"] == 2
    assert diag["cpu_operators_with_kernels"] == 1
    assert diag["emitted_device_record_count"] == 1


# --------------------------------------------------------------------------- #
# B5: scope-health diagnostics
# --------------------------------------------------------------------------- #
def test_b5_scope_health_counters():
    recs = [
        EventRecord(SAMPLER_K, True, 1000.0, scopes=(MASKS,)),
        EventRecord("_stn_crop_forward_kernel", True, 300.0, scopes=(CROP,)),
        EventRecord("_stn_paste_forward_kernel", True, 400.0, scopes=(PASTE,)),
        EventRecord("ampere_sgemm_128x128_nt", True, 50000.0),
    ]
    d = attribute(recs).as_dict()
    assert d["scoped_masks_device_event_count"] == 1
    assert d["scoped_crop_device_event_count"] == 1
    assert d["scoped_paste_device_event_count"] == 1
    assert d["device_event_count_by_scope"]["<unscoped>"] == 1
    assert d["known_sampler_forward_events_without_scope"] == 0
    assert d["known_sampler_backward_events_without_scope"] == 0
    # bulk unscoped GEMM time is expected and must NOT read as ancestry loss
    assert d["unscoped_device_ms_informational"] == 50.0


def test_b5_lost_ancestry_is_caught_by_unscoped_FORWARD_sampler():
    """If the adapter loses ancestry, FORWARD sampler kernels appear unscoped."""
    recs = [EventRecord(SAMPLER_K, True, 1000.0), EventRecord(SAMPLER_K, True, 900.0)]
    d = attribute(recs).as_dict()
    assert d["known_sampler_forward_events_without_scope"] == 2
    assert d["known_sampler_backward_events_without_scope"] == 0
    assert d["scoped_masks_device_event_count"] == 0
    assert d["scoped_crop_device_event_count"] == 0


# --------------------------------------------------------------------------- #
# C1: unscoped BACKWARD sampler is expected, not an ancestry failure
# --------------------------------------------------------------------------- #
def test_c1_reference_training_unscoped_backward_is_expected():
    """ScopeProbe wraps the Python forward call; its scope has exited by the time
    autograd runs, so a reference backward sampler is legitimately unscoped."""
    recs = [
        EventRecord(SAMPLER_K, True, 1580.0, scopes=(CROP,)),       # forward, scoped
        EventRecord(SAMPLER_BWD_K, True, 975.0),                    # backward, unscoped
    ]
    d = attribute(recs).as_dict()
    assert d["known_sampler_forward_events_without_scope"] == 0, "forward must stay scoped"
    assert d["known_sampler_backward_events_without_scope"] == 1, "backward unscoped is expected"
    assert d["scoped_crop_device_event_count"] == 1


def test_c1_reference_inference_has_no_backward_sampler():
    recs = [EventRecord(SAMPLER_K, True, 1092.0, scopes=(CROP,))]
    d = attribute(recs).as_dict()
    assert d["known_sampler_forward_events_without_scope"] == 0
    assert d["known_sampler_backward_events_without_scope"] == 0


def test_c1_triton_run_has_no_reference_sampler_events():
    recs = [
        EventRecord("_stn_crop_forward_kernel", True, 22.0, scopes=(CROP,)),
        EventRecord("_stn_paste_grad_patches_kernel", True, 641.0),
    ]
    d = attribute(recs).as_dict()
    assert d["known_sampler_forward_events_without_scope"] == 0
    assert d["known_sampler_backward_events_without_scope"] == 0
    assert d["buckets_ms"]["crop"] == 0.022
    assert d["buckets_ms"]["paste"] == 0.641, "allowlist classifies backward Triton kernels"


def test_c1_sampler_direction_backward_checked_before_forward():
    from attribution import sampler_direction
    assert sampler_direction(SAMPLER_BWD_K) == "backward", "backward name also contains the forward substring"
    assert sampler_direction(SAMPLER_K) == "forward"
    assert sampler_direction("_stn_crop_forward_kernel") is None
    assert sampler_direction("ampere_sgemm_128x128_nt") is None


# --------------------------------------------------------------------------- #
# B6: suspicious-kernel policy, both halves
# --------------------------------------------------------------------------- #
def test_b6_suspicious_inside_trusted_scope_is_counted_and_flagged():
    a = attribute([EventRecord("_stn_crop_experimental_kernel", True, 1000.0, scopes=(CROP,))])
    assert a.buckets_ms == {"crop": 1.0}, a.buckets_ms
    assert a.stn_device_kernel_ms == 1.0
    assert "_stn_crop_experimental_kernel" in a.suspicious_kernels_ms
    assert "_stn_crop_experimental_kernel" not in a.suspicious_excluded_ms


def test_b6_suspicious_without_scope_is_flagged_and_excluded():
    a = attribute([EventRecord("_stn_crop_experimental_kernel", True, 1000.0)])
    assert a.stn_device_kernel_ms == 0.0
    assert "_stn_crop_experimental_kernel" in a.suspicious_kernels_ms
    assert "_stn_crop_experimental_kernel" in a.suspicious_excluded_ms
    assert "_stn_crop_experimental_kernel" in a.unmatched_kernels_ms


# --------------------------------------------------------------------------- #
# B2: backend-specific grad-x expectations
# --------------------------------------------------------------------------- #
def test_b2_triton_gradx_expectations():
    ok_true = evaluate_gradx_expectation("triton", True, ["_stn_crop_forward_kernel", "_stn_crop_grad_x_kernel"])
    assert ok_true["expectation_met"] is True
    assert ok_true["expected_grad_x_kernel_family"] == "triton:_stn_crop_grad_x_kernel"

    bad_true = evaluate_gradx_expectation("triton", True, ["_stn_crop_forward_kernel"])
    assert bad_true["expectation_met"] is False

    ok_false = evaluate_gradx_expectation("triton", False, ["_stn_crop_forward_kernel",
                                                            "_stn_crop_grad_params_kernel"])
    assert ok_false["expectation_met"] is True, "absent grad-x is CORRECT when grad not requested"


def test_b2_reference_gradx_expectations():
    ok = evaluate_gradx_expectation("reference", True, [SAMPLER_K, SAMPLER_BWD_K])
    assert ok["expectation_met"] is True
    assert ok["expected_grad_x_kernel_family"] == "reference:grid_sampler_2d_backward_kernel"

    leaked = evaluate_gradx_expectation("reference", True, [SAMPLER_BWD_K, "_stn_crop_grad_x_kernel"])
    assert leaked["expectation_met"] is False, "a Triton kernel in the reference backend is a leak"
    assert "leaked" in leaked["reason"]

    missing = evaluate_gradx_expectation("reference", True, [SAMPLER_K])
    assert missing["expectation_met"] is False


def test_b2_reference_never_requires_the_triton_grad_x_kernel():
    """The discredited claim: grad_x_kernel_present must NOT be demanded of reference."""
    r = evaluate_gradx_expectation("reference", False, [SAMPLER_K])
    assert r["expectation_met"] is True


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                                   # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL {name}: {exc}")
        else:
            print(f"  ok   {name}")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} of {len(tests)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
