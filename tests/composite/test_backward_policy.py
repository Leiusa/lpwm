#!/usr/bin/env python
"""
Frozen backward-policy foundation: fixture integrity + held-out re-validation.

No kernel is involved. This checks that the compact dense envelope survives
serialization intact, that its invariants hold, and that the held-out particle
orders -- never used to build it -- still fall inside when the envelope is read
back from the committed file rather than from memory.

    python tests/composite/test_backward_policy.py
"""

import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from composite_cases import CASES, make_inputs                      # noqa: E402
from gen_backward_envelope import (                                 # noqa: E402
    ATOL, ENVELOPED, HELDOUT_SPEC, L2_TOL, RTOL, SCENARIOS, TINY,
    base_cots, combined_report, envelope_report, make_order, run_grads, selftest_reporting)

FIXTURE = os.path.join(_HERE, "fixtures", "backward_envelope.pt")
POLICY = os.path.join(_HERE, "fixtures", "frozen_backward_policy.json")

#: 7808 bounded elements, lower+upper, float32
EXPECTED_NUMEL = 15616
EXPECTED_LOGICAL_BYTES = 62464
FILE_SIZE_LIMIT = 1_000_000


def load():
    d = torch.load(FIXTURE, weights_only=False)
    p = json.load(open(POLICY))
    return d, p


def test_fixture_contains_only_bounds():
    d, _ = load()
    assert set(d) == {"version", "env"}, set(d)
    for key, rec in d["env"].items():
        assert set(rec) == {"lower", "upper"}, f"{key}: {set(rec)}"
        parts = key.split("|")
        assert len(parts) == 3 and parts[2] in ENVELOPED, key


def test_fixture_size_and_layout():
    d, _ = load()
    numel = logical = 0
    for key, rec in d["env"].items():
        for side, t in rec.items():
            assert t.dtype == torch.float32, f"{key}/{side}: {t.dtype}"
            assert t.device.type == "cpu"
            assert t.is_contiguous(), f"{key}/{side} not contiguous"
            assert t.storage_offset() == 0, f"{key}/{side} storage_offset={t.storage_offset()}"
            sb = t.untyped_storage().nbytes()
            lb = t.numel() * t.element_size()
            assert sb == lb, f"{key}/{side}: storage {sb} != logical {lb} (retained backing store)"
            numel += t.numel(); logical += lb
    assert numel == EXPECTED_NUMEL, f"{numel} != {EXPECTED_NUMEL}"
    assert logical == EXPECTED_LOGICAL_BYTES, f"{logical} != {EXPECTED_LOGICAL_BYTES}"
    size = os.path.getsize(FIXTURE)
    assert size < FILE_SIZE_LIMIT, f"fixture {size} bytes exceeds {FILE_SIZE_LIMIT}"


def test_lower_le_upper():
    d, _ = load()
    for key, rec in d["env"].items():
        assert bool((rec["lower"] <= rec["upper"]).all()), f"{key}: lower > upper somewhere"


def test_roundtrip_bit_identical():
    d, _ = load()
    tmp = os.path.join(_HERE, "fixtures", ".roundtrip_check.pt")
    torch.save(d, tmp)
    try:
        e = torch.load(tmp, weights_only=False)
        assert set(e["env"]) == set(d["env"])
        for key in d["env"]:
            for side in ("lower", "upper"):
                assert torch.equal(d["env"][key][side], e["env"][key][side]), f"{key}/{side}"
    finally:
        os.remove(tmp)


def test_policy_metadata_frozen():
    _, p = load()
    assert p["frozen_before_backward_kernel_exists"] is True
    assert p["atol"]["dec_objects"] == 2e-5
    for g in ("z_kp", "z_scale", "obj_on", "z_depth"):
        assert p["atol"][g] == 1e-5, (g, p["atol"][g])
    assert p["rtol"] == 1e-5 and p["l2_tol"] == 1e-5
    assert p["validation"]["heldout_failures"] == 0
    assert [n for n, _ in p["development_orders"]]
    assert [n for n, _ in p["heldout_orders"]]


def test_reporting_index_regression():
    assert selftest_reporting()


def test_heldout_orders_pass_from_reloaded_fixture():
    """The validation that matters: re-run held-out orders against the file."""
    d, _ = load()
    worst_ratio = 0.0
    worst_l2 = {}
    failures = []
    for cname, kw in CASES:
        K = kw["n_kp"]
        ins = make_inputs(device="cuda", **kw)
        for sc in SCENARIOS:
            cb = base_cots(ins, sc, cname)
            if not cb:
                continue
            canonical = run_grads(ins, cb)
            for oname, spec in HELDOUT_SPEC:
                o = make_order(oname, spec, K, ins["z_kp"].device)
                g = run_grads(ins, cb, order=o)
                for e in ENVELOPED:
                    key = f"{cname}|{sc}|{e}"
                    if g.get(e) is None or key not in d["env"]:
                        continue
                    lo = d["env"][key]["lower"].to(g[e].device)
                    hi = d["env"][key]["upper"].to(g[e].device)
                    r = envelope_report(lo, hi, g[e])
                    worst_ratio = max(worst_ratio, r["worst_ratio"])
                    if not r["ok"]:
                        failures.append((cname, sc, oname, e, r))
                for og in ("dec_objects", "obj_on", "z_depth"):
                    if g.get(og) is None:
                        continue
                    r = combined_report(canonical[og], g[og], og)
                    worst_l2[og] = max(worst_l2.get(og, 0.0), r["rel_l2"])
                    if not r["ok"]:
                        failures.append((cname, sc, oname, og, r))
        del ins
        torch.cuda.empty_cache()
    assert worst_ratio == 0.0, f"worst held-out distance-to-envelope ratio {worst_ratio}"
    assert not failures, failures[:5]
    for og, v in worst_l2.items():
        assert v <= L2_TOL, f"{og} relative L2 {v}"


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                                    # noqa: BLE001
            fails.append((name, exc)); print(f"  FAIL {name}: {exc}")
        else:
            print(f"  ok   {name}")
    print(f"\n{'FAILED' if fails else 'PASSED'}: {len(fails)} of {len(tests)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
