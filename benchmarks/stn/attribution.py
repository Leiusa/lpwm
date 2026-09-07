"""
Profiler attribution for the LPWM STN stage.

Split deliberately in two:

* :class:`EventRecord` + :func:`attribute` -- pure logic over normalized records.
  No torch profiler types, so it is unit-testable on CPU (see test_attribution.py).
* :func:`records_from_profiler` -- the version-dependent adapter that turns a
  ``torch.profiler`` result into those records. This is the part that can only be
  validated on a GPU run.

Rules this module enforces, each one a defect found in the first validation run:

D1  Only true device events are summed. A CPU operator aggregate
    (``aten::grid_sampler_2d``) and the CUDA kernel it launches
    (``...grid_sampler_2d_kernel...``) are the same work; counting both
    double-counts it. Device events only, every event exactly once.
D2  Triton kernels are matched against an explicit allowlist, never a
    ``triton_`` substring, which would swallow unrelated Triton kernels.
    A crop/paste-looking kernel that is not on the list is reported as
    suspicious rather than silently bucketed.
D3  ``create_masks_fast`` has no uniquely named kernel -- it calls the same
    ``grid_sampler_2d`` as everything else. It is therefore attributed by
    enclosing record_function scope, never by kernel name, and never merged
    into the generic sampler bucket.
D4  The reference path's ``affine_grid`` build emits a generic elementwise
    kernel. Scope attribution captures it; kernel-name matching never could.

Suspicious-kernel policy (B6) -- one rule, stated once:

    inside a trusted STN scope   -> counted in that scope's bucket AND flagged
                                    suspicious. The scope is direct evidence of
                                    what the caller was doing, which outranks an
                                    unrecognized kernel name.
    with no trusted STN scope    -> flagged suspicious AND excluded from STN
                                    totals. Nothing but the name vouches for it.

So ``suspicious_kernels_ms`` is a review queue, not a synonym for "excluded".
``suspicious_excluded_ms`` reports the excluded subset explicitly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "EventRecord", "Attribution", "attribute",
    "STN_TRITON_KERNELS", "SAMPLER_KERNEL_MARKERS", "SCOPE_BUCKETS", "SCOPE_PREFIX",
    "normalize_kernel", "classify_kernel", "records_from_profiler",
]

#: prefix for every record_function scope this harness installs
SCOPE_PREFIX = "lpwm_stn::"

#: scope name -> bucket. Scope attribution takes precedence over kernel names,
#: because a scope knows what the caller was doing and a kernel name does not.
SCOPE_BUCKETS = {
    f"{SCOPE_PREFIX}crop": "crop",
    f"{SCOPE_PREFIX}paste": "paste",
    f"{SCOPE_PREFIX}masks_fast": "masks",
    f"{SCOPE_PREFIX}masks_scale": "masks",
}

#: exact Triton kernel names owned by the LPWM STN backend. Allowlist, not a pattern.
STN_TRITON_KERNELS = frozenset({
    "_stn_crop_forward_kernel",
    "_stn_crop_grad_x_kernel",
    "_stn_crop_grad_params_kernel",
    "_stn_paste_forward_kernel",
    "_stn_paste_grad_patches_kernel",
    "_stn_paste_grad_params_kernel",
    "_stn_paste_reduce_params_kernel",
})

#: substrings identifying the reference sampler's own CUDA kernels. These are
#: unambiguous device-kernel names, unlike the `aten::` aggregates above them.
SAMPLER_KERNEL_MARKERS = ("grid_sampler_2d_kernel", "grid_sampler_2d_backward_kernel")

#: backward marker checked FIRST -- "grid_sampler_2d_backward_kernel" also
#: contains "grid_sampler_2d", so order matters
SAMPLER_BACKWARD_MARKER = "grid_sampler_2d_backward"


def sampler_direction(name: str) -> Optional[str]:
    """``"backward"``, ``"forward"``, or None if this is not a sampler kernel."""
    low = name.lower()
    if SAMPLER_BACKWARD_MARKER in low:
        return "backward"
    if any(m.lower() in low for m in SAMPLER_KERNEL_MARKERS):
        return "forward"
    return None

#: a kernel that looks like it belongs to the STN but is not on the allowlist
#: must be surfaced, never silently counted
_SUSPICIOUS_MARKERS = ("stn_", "_crop_", "_paste_")


@dataclass(frozen=True)
class EventRecord:
    """One profiler event, normalized.

    ``is_device`` is the D1 gate: True only for actual on-device execution.
    ``scopes`` lists enclosing record_function scopes, outermost first.
    """

    name: str
    is_device: bool
    self_device_us: float = 0.0
    scopes: Tuple[str, ...] = ()


@dataclass
class Attribution:
    total_device_kernel_ms: float = 0.0
    stn_device_kernel_ms: float = 0.0
    buckets_ms: Dict[str, float] = field(default_factory=dict)
    matched_kernels_ms: Dict[str, float] = field(default_factory=dict)
    unmatched_kernels_ms: Dict[str, float] = field(default_factory=dict)
    suspicious_kernels_ms: Dict[str, float] = field(default_factory=dict)
    #: the subset of suspicious time actually excluded from STN totals (no trusted scope)
    suspicious_excluded_ms: Dict[str, float] = field(default_factory=dict)
    #: informational only -- NOT a scope-health signal. Convolutions and GEMMs are
    #: legitimately unscoped, so this is large in a healthy run (B5).
    unscoped_device_ms: float = 0.0
    device_event_count: int = 0
    #: only meaningful if the adapter emits non-device records; today it does not,
    #: so this is a pure-attribution field and never a GPU acceptance signal (B1)
    skipped_non_device_count: int = 0
    #: scope-health diagnostics (B5)
    device_event_count_by_scope: Dict[str, int] = field(default_factory=dict)
    device_kernel_ms_by_scope: Dict[str, float] = field(default_factory=dict)
    #: C1 -- ScopeProbe wraps the Python-level forward calls, so their scopes have
    #: already exited by the time autograd runs. A reference backward sampler
    #: kernel is therefore EXPECTED to be unscoped during training and is not an
    #: ancestry failure. Only an unscoped FORWARD sampler indicates lost ancestry.
    known_sampler_forward_events_without_scope: int = 0
    known_sampler_backward_events_without_scope: int = 0

    @property
    def stn_share_of_device_kernel_time_pct(self) -> float:
        if not self.total_device_kernel_ms:
            return 0.0
        return 100.0 * self.stn_device_kernel_ms / self.total_device_kernel_ms

    def as_dict(self) -> dict:
        by_scope = dict(sorted(self.device_event_count_by_scope.items()))
        return {
            "total_device_kernel_ms": self.total_device_kernel_ms,
            "stn_device_kernel_ms": self.stn_device_kernel_ms,
            "stn_share_of_device_kernel_time_pct": self.stn_share_of_device_kernel_time_pct,
            "buckets_ms": dict(sorted(self.buckets_ms.items())),
            "matched_kernels_ms": dict(sorted(self.matched_kernels_ms.items(), key=lambda kv: -kv[1])),
            "unmatched_kernels_ms": dict(sorted(self.unmatched_kernels_ms.items(), key=lambda kv: -kv[1])[:20]),
            "suspicious_kernels_ms": dict(sorted(self.suspicious_kernels_ms.items(), key=lambda kv: -kv[1])),
            "suspicious_excluded_ms": dict(sorted(self.suspicious_excluded_ms.items(), key=lambda kv: -kv[1])),
            "unscoped_device_ms_informational": self.unscoped_device_ms,
            "device_event_count": self.device_event_count,
            "skipped_non_device_count": self.skipped_non_device_count,
            "device_event_count_by_scope": by_scope,
            "device_kernel_ms_by_scope": dict(sorted(self.device_kernel_ms_by_scope.items())),
            "scoped_crop_device_event_count": by_scope.get("crop", 0),
            "scoped_paste_device_event_count": by_scope.get("paste", 0),
            "scoped_masks_device_event_count": by_scope.get("masks", 0),
            "known_sampler_forward_events_without_scope":
                self.known_sampler_forward_events_without_scope,
            "known_sampler_backward_events_without_scope":
                self.known_sampler_backward_events_without_scope,
        }


def normalize_kernel(name: str) -> str:
    """Strip the compiler-added suffix from a Triton kernel name.

    Triton appends a signature suffix (``_0d1d2de``, ``.kd``, ...). An allowlist
    entry matches when it is a prefix and the remainder is only such a suffix --
    never an arbitrary continuation, so ``_stn_crop_forward_kernel_v2_experimental``
    does NOT normalize onto ``_stn_crop_forward_kernel``.
    """
    base = name.split("(")[0].strip()
    base = base.rsplit("::", 1)[-1]
    for known in STN_TRITON_KERNELS:
        if base == known:
            return known
        if base.startswith(known):
            suffix = base[len(known):]
            # only a compiler signature suffix: separators, digits, and the few
            # letters Triton emits ('d' divisible, 'c' constant, 'e' equal-1, '.kd')
            if suffix and all(ch in "._0123456789dcek" for ch in suffix):
                return known
    return base


def classify_kernel(name: str) -> Tuple[Optional[str], bool]:
    """Return ``(bucket_or_None, suspicious)`` for a device-kernel name."""
    norm = normalize_kernel(name)
    if norm in STN_TRITON_KERNELS:
        return ("crop" if "_crop_" in norm else "paste"), False
    low = name.lower()
    if any(m.lower() in low for m in SAMPLER_KERNEL_MARKERS):
        return "sampler", False
    if any(m in low for m in _SUSPICIOUS_MARKERS):
        return None, True
    return None, False


def _scope_bucket(scopes: Sequence[str]) -> Optional[str]:
    """Innermost enclosing STN scope wins (most specific description of the work)."""
    for scope in reversed(tuple(scopes)):
        if scope in SCOPE_BUCKETS:
            return SCOPE_BUCKETS[scope]
    return None


def attribute(records: Sequence[EventRecord]) -> Attribution:
    """Bucket device events. Every device event is counted exactly once.

    Precedence: an enclosing trusted STN scope decides the bucket; only when
    there is none does the kernel name decide. See the module docstring for the
    suspicious-kernel policy (B6).
    """
    out = Attribution()
    for rec in records:
        if not rec.is_device:
            out.skipped_non_device_count += 1
            continue
        out.device_event_count += 1
        ms = rec.self_device_us / 1e3
        out.total_device_kernel_ms += ms

        scope_bucket = _scope_bucket(rec.scopes)     # trusted-scope evidence
        name_bucket, suspicious = classify_kernel(rec.name)
        bucket = scope_bucket if scope_bucket is not None else name_bucket

        # scope-health accounting (B5): key by trusted scope, not by kernel name
        key = scope_bucket or "<unscoped>"
        out.device_event_count_by_scope[key] = out.device_event_count_by_scope.get(key, 0) + 1
        out.device_kernel_ms_by_scope[key] = out.device_kernel_ms_by_scope.get(key, 0.0) + ms

        if scope_bucket is None:
            out.unscoped_device_ms += ms
            direction = sampler_direction(rec.name)
            if direction == "forward":
                # an unscoped FORWARD sampler is the specific ancestry-loss signal
                out.known_sampler_forward_events_without_scope += 1
            elif direction == "backward":
                # expected during training: the forward scope has already exited
                out.known_sampler_backward_events_without_scope += 1

        if suspicious:
            out.suspicious_kernels_ms[rec.name] = out.suspicious_kernels_ms.get(rec.name, 0.0) + ms
            if scope_bucket is None:
                out.suspicious_excluded_ms[rec.name] = out.suspicious_excluded_ms.get(rec.name, 0.0) + ms

        if bucket is None:
            out.unmatched_kernels_ms[rec.name] = out.unmatched_kernels_ms.get(rec.name, 0.0) + ms
        else:
            out.buckets_ms[bucket] = out.buckets_ms.get(bucket, 0.0) + ms
            out.stn_device_kernel_ms += ms
            out.matched_kernels_ms[rec.name] = out.matched_kernels_ms.get(rec.name, 0.0) + ms
    return out


# --------------------------------------------------------------------------- #
# profiler adapter -- version dependent, validated only on a GPU run
# --------------------------------------------------------------------------- #
def _scope_chain(event) -> Tuple[str, ...]:
    """record_function scopes enclosing ``event``, outermost first."""
    chain: List[str] = []
    node = getattr(event, "cpu_parent", None)
    guard = 0
    while node is not None and guard < 64:
        name = getattr(node, "name", "") or getattr(node, "key", "")
        if isinstance(name, str) and name.startswith(SCOPE_PREFIX):
            chain.append(name)
        node = getattr(node, "cpu_parent", None)
        guard += 1
    chain.reverse()
    return tuple(chain)


def records_from_profiler(prof) -> Tuple[List[EventRecord], dict]:
    """Normalize a ``torch.profiler`` result into :class:`EventRecord` objects.

    Returns ``(records, diagnostics)``.

    Both strategies emit **device records only**, so
    ``Attribution.skipped_non_device_count`` is normally zero and must never be
    used as evidence that CPU aggregates were excluded (B1). The proof lives in
    these diagnostics instead:

    ``raw_event_count``
        events returned by ``prof.events()``.
    ``cpu_event_count_seen``
        events that are not device events, i.e. CPU-side operator rows.
    ``cpu_events_with_device_aggregate_excluded``
        CPU operator rows carrying a nonzero device-time aggregate that were
        deliberately NOT emitted as records -- the double-counting fix made
        visible. **Version dependent**: it relies on ``device_time_total`` /
        ``cuda_time_total`` being exposed on CPU rows, so a zero here is not by
        itself a failure. Use the linked-kernel invariants below instead (C4).
    ``cpu_operators_with_kernels``
        CPU operator rows with a non-empty ``.kernels`` list. Stable across
        versions, and the denominator that makes ``linked_kernel_count``
        interpretable.
    ``linked_kernel_count``
        kernels reached through their launching CPU operator (strategy
        ``cpu_op_kernels``); each kernel belongs to exactly one operator.
    ``standalone_device_event_count``
        device events found directly (strategy ``device_events``).
    ``emitted_device_record_count``
        records handed to :func:`attribute`; equals ``linked_kernel_count`` or
        ``standalone_device_event_count`` depending on the strategy used.
    ``strategy``
        which path produced the records.
    """
    events = list(prof.events())
    try:
        from torch.autograd import DeviceType
        cuda_type = DeviceType.CUDA
    except Exception:                                    # pragma: no cover
        cuda_type = None

    def _is_device(ev):
        dt = getattr(ev, "device_type", None)
        return (dt == cuda_type) if cuda_type is not None else False

    def _device_aggregate_us(ev):
        return float(getattr(ev, "device_time_total", None)
                     or getattr(ev, "cuda_time_total", None) or 0.0)

    diag = {
        "raw_event_count": len(events),
        "cpu_event_count_seen": 0,
        "cpu_events_with_device_aggregate_excluded": 0,
        "cpu_operators_with_kernels": 0,
        "linked_kernel_count": 0,
        "standalone_device_event_count": 0,
        "emitted_device_record_count": 0,
        "emitted_cpu_record_count": 0,
        "strategy": None,
    }

    records: List[EventRecord] = []
    for ev in events:
        if _is_device(ev):
            diag["standalone_device_event_count"] += 1
            continue
        diag["cpu_event_count_seen"] += 1
        kernels = getattr(ev, "kernels", None) or []
        if _device_aggregate_us(ev) > 0:
            # this CPU row carries a device-time aggregate; emitting it alongside
            # its kernels is exactly the D1 double count, so it is excluded
            diag["cpu_events_with_device_aggregate_excluded"] += 1
        if not kernels:
            continue
        diag["cpu_operators_with_kernels"] += 1
        scopes = _scope_chain(ev)
        own = getattr(ev, "name", "") or getattr(ev, "key", "")
        if isinstance(own, str) and own.startswith(SCOPE_PREFIX):
            scopes = scopes + (own,)
        for k in kernels:
            diag["linked_kernel_count"] += 1
            records.append(EventRecord(
                name=getattr(k, "name", "<kernel>"),
                is_device=True,
                self_device_us=float(getattr(k, "duration", 0.0)),
                scopes=scopes,
            ))

    if records:
        diag["strategy"] = "cpu_op_kernels"
        diag["emitted_device_record_count"] = sum(1 for r in records if r.is_device)
        diag["emitted_cpu_record_count"] = sum(1 for r in records if not r.is_device)
        return records, diag

    # fallback: standalone device events; scope ancestry may be unavailable
    for ev in events:
        if not _is_device(ev):
            continue
        self_us = (getattr(ev, "self_device_time_total", None)
                   or getattr(ev, "self_cuda_time_total", None) or 0.0)
        records.append(EventRecord(name=getattr(ev, "key", "<event>"), is_device=True,
                                   self_device_us=float(self_us), scopes=_scope_chain(ev)))
    diag["strategy"] = "device_events"
    diag["emitted_device_record_count"] = sum(1 for r in records if r.is_device)
    diag["emitted_cpu_record_count"] = sum(1 for r in records if not r.is_device)
    return records, diag


# --------------------------------------------------------------------------- #
# grad-x expectation logic (B2) -- pure, so it is CPU-testable
# --------------------------------------------------------------------------- #
#: what a correct backward looks like per backend when grad w.r.t. x IS requested
GRAD_X_FAMILY = {
    "triton": "triton:_stn_crop_grad_x_kernel",
    "reference": "reference:grid_sampler_2d_backward_kernel",
}


def evaluate_gradx_expectation(backend: str, requires_grad_x: bool,
                               observed_names: Sequence[str]) -> dict:
    """Decide whether a crop-backward microprofile met its backend's expectation.

    Triton
        ``x.requires_grad=True``  -> ``_stn_crop_grad_x_kernel`` must appear.
        ``x.requires_grad=False`` -> it must NOT appear. ``_StnCrop.backward``
        launches it only under ``ctx.needs_input_grad[0]``, so absence there is
        correct behaviour, not a fallback.
    Reference
        ``True`` -> the grid-sampler backward kernel must appear.
        Either way, no ``_stn_*`` Triton kernel may appear at all: seeing one
        would mean the registry leaked the accelerated path into the reference
        backend.
    """
    names = list(observed_names)
    low = [n.lower() for n in names]
    has_triton_gradx = any("_stn_crop_grad_x_kernel" in n for n in names)
    has_any_stn = any("_stn_" in n for n in names)
    has_sampler_bwd = any("grid_sampler_2d_backward" in n for n in low)

    if backend == "triton":
        met = has_triton_gradx if requires_grad_x else not has_triton_gradx
        reason = ("grad-x kernel present as required" if requires_grad_x and met else
                  "grad-x kernel correctly absent (grad not requested)" if met else
                  "grad-x kernel missing while grad WAS requested" if requires_grad_x else
                  "grad-x kernel present although grad was NOT requested")
    elif backend == "reference":
        met = (has_sampler_bwd if requires_grad_x else True) and not has_any_stn
        reason = ("reference sampler backward present, no Triton kernels" if met else
                  "Triton kernel leaked into the reference backend" if has_any_stn else
                  "reference sampler backward missing while grad WAS requested")
    else:                                                        # pragma: no cover
        return {"expected_grad_x_kernel_family": None, "observed_grad_x_kernel_names": names,
                "expectation_met": None, "reason": f"unknown backend {backend!r}"}

    return {
        "backend": backend,
        "x_requires_grad": requires_grad_x,
        "expected_grad_x_kernel_family": GRAD_X_FAMILY.get(backend),
        "observed_grad_x_kernel_names": names,
        "expectation_met": met,
        "reason": reason,
    }
