#!/usr/bin/env python
"""
Policy v2 fixture integrity, provenance guard, and held-out re-validation.

v2 widens the v1 reference envelope with the spread of the already-accepted
Triton paste. The guard below is the important part: it proves by static analysis
that the generator never imported the fused backward candidate, so the contract
cannot have been shaped by the thing it judges.
"""

import ast
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

V1_PT = os.path.join(_HERE, "fixtures", "backward_envelope.pt")
V2_PT = os.path.join(_HERE, "fixtures", "backward_envelope_v2.pt")
V2_JSON = os.path.join(_HERE, "fixtures", "frozen_backward_policy_v2.json")
GENERATOR = os.path.join(_HERE, "gen_told_envelope.py")

#: modules that would mean the contract saw the candidate it judges
FORBIDDEN = ("composite_backward", "composite_triton", "sampler_vjp", "fused")
#: the generic dispatcher resolves to whichever backend is active, which would make
#: the recorded provenance meaningless; both sides must be named explicitly
DISPATCHER_CALLS = ("lpwm_stn.stn_paste(", "stn_paste(", )
EXPLICIT_CALLS = ("reference.stn_paste", "triton_backend.stn_paste")
TOLD_BLOB_SHA = "bd18b02c5e01c72a0715c12e1d5924cec21bafe4"
FILE_SIZE_LIMIT = 1_000_000


def _load():
    return torch.load(V2_PT, weights_only=False), json.load(open(V2_JSON))


def test_generator_imports_only_reference_and_accepted_paste():
    """Static provenance guard: no fused-backward import anywhere in the generator."""
    src = open(GENERATOR).read()
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    bad = [m for m in mods if any(f in m for f in FORBIDDEN)]
    assert not bad, f"generator imports the candidate it judges: {bad}"
    assert "lpwm_stn" in mods, mods
    # and no dynamic escape hatch
    for pat in ("__import__", "importlib", "exec(", "eval("):
        assert pat not in src, f"generator contains a dynamic import path: {pat}"


def test_generator_names_both_pastes_explicitly():
    """The generic dispatcher would resolve to whichever backend is active, making
    the recorded T_old provenance meaningless."""
    src = open(GENERATOR).read()
    for call in EXPLICIT_CALLS:
        assert call in src, f"generator never calls {call}"
    assert "lpwm_stn.stn_paste(" not in src, "generator uses the generic dispatcher"
    # bare `stn_paste(` may only appear qualified
    import re
    for m in re.finditer(r"(?<![\w.])stn_paste\s*\(", src):
        ctx = src[max(0, m.start() - 24):m.start()]
        assert ctx.rstrip().endswith(("reference.", "triton_backend.")), (
            f"unqualified stn_paste call near: ...{ctx[-40:]!r}")


def test_told_provenance_pinned():
    _, p = _load()
    assert p["told_blob_sha"] == TOLD_BLOB_SHA, (p["told_blob_sha"], TOLD_BLOB_SHA)
    assert p["told_source"] == "lpwm_stn/triton_backend.py"
    assert p["reference_bounds_reused_from_v1"] == ["z_kp", "z_scale"]
    assert p["reference_bounds_regenerated"] == ["z_depth"]
    assert p["validation"]["reference_heldout_violations"] == 0


def test_v2_metadata_records_provenance():
    _, p = _load()
    assert p["version"] == 2
    assert p["never_derived_from"] == "the fused composite backward candidate"
    assert "cross-implementation" in p["why_v1_was_incomplete"]
    assert p["validation"]["told_heldout_envelope_violations"] == 0
    assert p["validation"]["dec_objects_obj_on_heldout_violations"] == 0
    assert p["z_depth_l2_selected_from"] == "T_old development data only"
    assert p["l2_tol"]["z_depth"] in p["l2_ladder"]
    for g in ("dec_objects", "z_kp", "z_scale", "obj_on"):
        assert p["l2_tol"][g] == 1e-5, (g, p["l2_tol"][g])


def test_v2_contains_v1_envelope():
    """v2 may only widen: every v1 bound must be inside its v2 counterpart."""
    v1 = torch.load(V1_PT, weights_only=False)
    v2, _ = _load()
    for key, rec in v1["env"].items():
        assert key in v2["env"], f"v2 lost {key}"
        lo1, hi1 = rec["lower"], rec["upper"]
        lo2, hi2 = v2["env"][key]["lower"], v2["env"][key]["upper"]
        assert bool((lo2 <= lo1).all()), f"{key}: v2 lower is tighter than v1"
        assert bool((hi2 >= hi1).all()), f"{key}: v2 upper is tighter than v1"


def test_v2_fixture_layout():
    v2, _ = _load()
    assert set(v2) == {"version", "env"}
    for key, rec in v2["env"].items():
        assert set(rec) == {"lower", "upper"}, key
        for side, t in rec.items():
            assert t.dtype == torch.float32 and t.device.type == "cpu"
            assert t.is_contiguous() and t.storage_offset() == 0, f"{key}/{side}"
            assert t.untyped_storage().nbytes() == t.numel() * t.element_size(), (
                f"{key}/{side}: retained backing storage")
        assert bool((rec["lower"] <= rec["upper"]).all()), f"{key}: lower > upper"
    assert os.path.getsize(V2_PT) < FILE_SIZE_LIMIT


def test_v2_covers_z_depth():
    v2, p = _load()
    assert "z_depth" in p["enveloped_gradients"]
    assert any(k.endswith("|z_depth") for k in v2["env"]), "no z_depth envelope in v2"


def test_roundtrip_bit_identical():
    v2, _ = _load()
    tmp = V2_PT + ".roundtrip"
    torch.save(v2, tmp)
    try:
        e = torch.load(tmp, weights_only=False)
        for key in v2["env"]:
            for side in ("lower", "upper"):
                assert torch.equal(v2["env"][key][side], e["env"][key][side]), f"{key}/{side}"
    finally:
        os.remove(tmp)


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
