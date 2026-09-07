#!/usr/bin/env python
"""
Standalone fused Triton composite BACKWARD against frozen policy v2.

This exercises the module as it lives in the repository -- ``lpwm_stn``, not a
scratch copy -- over the same 47 case/scenario combinations the policy was
frozen against. The provenance assertions below exist because the candidate was
developed outside the tree: if this file ever passes while importing something
other than the repository module, the result means nothing.

The policy values are re-asserted from the frozen JSON so that a later edit to
the fixture is a visible failure here rather than a silent loosening.

    python tests/composite/test_composite_triton_backward.py
"""

import ast
import json
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

V2_PT = os.path.join(_HERE, "fixtures", "backward_envelope_v2.pt")
V2_JSON = os.path.join(_HERE, "fixtures", "frozen_backward_policy_v2.json")

#: the frozen contract, restated so that editing the fixture fails this test
EXPECTED_L2 = {"dec_objects": 1e-05, "z_kp": 1e-05, "z_scale": 1e-05,
               "obj_on": 1e-05, "z_depth": 2e-05}
EXPECTED_ENVELOPED = ("z_depth", "z_kp", "z_scale")
GRADS = ("dec_objects", "z_kp", "z_scale", "obj_on", "z_depth")
EXPECTED_COMBINATIONS = 47

#: names that would mean a particle-count special case had been introduced
_PARTICLE_NAMES = ("K", "N_KP", "n_kp", "n_particles")


def _module():
    import lpwm_stn.composite_backward as mod
    return mod


def _source():
    return open(os.path.join(_ROOT, "lpwm_stn", "composite_backward.py")).read()


# --------------------------------------------------------------------------
# provenance: the thing under test must be the repository module
# --------------------------------------------------------------------------

def test_module_resolves_inside_the_repository():
    mod = _module()
    got = os.path.abspath(mod.__file__)
    want = os.path.join(_ROOT, "lpwm_stn", "composite_backward.py")
    assert got == want, f"imported {got}, expected {want}"


def test_support_modules_resolve_inside_the_repository():
    import composite_cases
    import gen_backward_envelope
    import lpwm_stn
    for mod, root in ((composite_cases, _HERE), (gen_backward_envelope, _HERE),
                      (lpwm_stn, os.path.join(_ROOT, "lpwm_stn"))):
        got = os.path.abspath(mod.__file__)
        assert got.startswith(root), f"{mod.__name__} resolved to {got}, outside {root}"


def test_no_scratch_directory_on_the_path():
    bad = [p for p in sys.path if "bench" in os.path.basename(p.rstrip("/")).lower()]
    assert not bad, f"scratch directories leaked onto sys.path: {bad}"


# --------------------------------------------------------------------------
# static invariants
# --------------------------------------------------------------------------

def test_no_float64():
    src = _source()
    hits = [ln for ln in src.splitlines() if "float64" in ln or "double" in ln]
    assert not hits, f"float64/double present: {hits}"


def test_no_particle_count_special_case():
    """No `K == 1` style branch, textually or in the parsed tree."""
    src = _source()
    pat = re.compile(r"\b(?:%s)\s*[=!]=\s*1\b" % "|".join(_PARTICLE_NAMES))
    textual = [ln.strip() for ln in src.splitlines() if pat.search(ln)]
    assert not textual, f"particle-count special case: {textual}"

    found = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Compare):
            continue
        if not isinstance(node.left, ast.Name) or node.left.id not in _PARTICLE_NAMES:
            continue
        for op, cmp_ in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)) and \
                    isinstance(cmp_, ast.Constant) and cmp_.value == 1:
                found.append(ast.dump(node))
    assert not found, f"particle-count special case in AST: {found}"


def test_policy_values_are_the_frozen_ones():
    pol = json.load(open(V2_JSON))
    assert pol["l2_tol"] == EXPECTED_L2, pol["l2_tol"]
    assert tuple(sorted(pol["enveloped_gradients"])) == tuple(sorted(EXPECTED_ENVELOPED))


# --------------------------------------------------------------------------
# the gate itself
# --------------------------------------------------------------------------

def _evaluate():
    """Run all combinations once; return (results, coverage)."""
    from composite_cases import CASES, make_inputs
    from gen_backward_envelope import (SCENARIOS, TINY, base_cots, run_grads,
                                       combined_report, envelope_report)
    from lpwm_stn.composite_backward import composite_backward_triton

    if not torch.cuda.is_available():
        raise RuntimeError("this gate requires CUDA")

    env = torch.load(V2_PT, weights_only=False)
    pol = json.load(open(V2_JSON))
    l2_tol, enveloped = pol["l2_tol"], set(pol["enveloped_gradients"])

    worst = {g: 0.0 for g in GRADS}
    wl2 = {g: 0.0 for g in GRADS}
    fails = []
    cov = {"n": 0, "scenarios": set(), "z_scale_none": 0, "masks_returned": set()}

    for cname, kw in CASES:
        ins = make_inputs(device="cuda", **kw)
        for sc in SCENARIOS:
            cb = base_cots(ins, sc, cname)
            if not cb:
                continue
            cots = dict(cb)
            ref = run_grads(ins, cb)
            cov["n"] += 1
            cov["scenarios"].add(sc)
            cov["z_scale_none"] += int(ins["z_scale"] is None)
            cov["masks_returned"].add("alpha_masks" in cots)

            gp, gkp, gsc, gon, gdep = composite_backward_triton(
                ins["dec_objects"], ins["z_kp"], ins["z_scale"], ins["obj_on"],
                ins["z_depth"], ins["img_size"],
                grad_masks=cots.get("alpha_masks"), grad_bg=cots.get("bg_mask"),
                grad_rgb=cots.get("dec_objects_trans"))
            cand = {"dec_objects": gp, "z_kp": gkp, "z_scale": gsc,
                    "obj_on": gon, "z_depth": gdep}

            for g in GRADS:
                if ref.get(g) is None:
                    if g == "z_scale" and cand.get(g) is not None:
                        fails.append((cname, sc, g, "gradient where reference has none"))
                    continue
                if cand.get(g) is None:
                    fails.append((cname, sc, g, "candidate returned None"))
                    continue
                l2 = float(torch.linalg.vector_norm(cand[g] - ref[g])
                           / max(float(torch.linalg.vector_norm(ref[g])), TINY))
                wl2[g] = max(wl2[g], l2)
                ok_l2 = l2 <= l2_tol[g]
                if g in enveloped:
                    key = f"{cname}|{sc}|{g}"
                    lo = env["env"][key]["lower"].to("cuda")
                    hi = env["env"][key]["upper"].to("cuda")
                    r = envelope_report(lo, hi, cand[g])
                    worst[g] = max(worst[g], r["worst_ratio"])
                    if not r["ok"] or not ok_l2:
                        fails.append((cname, sc, g,
                                      f"n_out={r['n_out']}/{r['numel']} "
                                      f"ratio={r['worst_ratio']:.3f} l2={l2:.3e}"))
                else:
                    r = combined_report(ref[g], cand[g], g)
                    worst[g] = max(worst[g], r["ratio"])
                    if not (r["n_viol"] == 0 and ok_l2):
                        fails.append((cname, sc, g,
                                      f"n_viol={r['n_viol']} ratio={r['ratio']:.3f} "
                                      f"l2={l2:.3e}"))
            del gp, gkp, gsc, gon, gdep
        del ins
        torch.cuda.empty_cache()
    return worst, wl2, fails, cov


_CACHE = {}


def _results():
    if "r" not in _CACHE:
        _CACHE["r"] = _evaluate()
    return _CACHE["r"]


def test_all_combinations_pass_policy_v2():
    worst, wl2, fails, _ = _results()
    print(f"\n    {'grad':<13}{'worst ratio':>13}{'worst rel-L2':>15}{'L2 cap':>10}")
    for g in GRADS:
        print(f"    {g:<13}{worst[g]:>13.4f}{wl2[g]:>15.3e}{EXPECTED_L2[g]:>10.0e}")
    for f in fails[:20]:
        print(f"    FAIL {f[0]}/{f[1]} grad[{f[2]}]: {f[3]}")
    assert not fails, f"{len(fails)} policy-v2 violation(s)"


def test_combination_count_is_complete():
    _, _, _, cov = _results()
    assert cov["n"] == EXPECTED_COMBINATIONS, cov["n"]


def test_all_four_cotangent_scenarios_covered():
    from gen_backward_envelope import SCENARIOS
    _, _, _, cov = _results()
    assert cov["scenarios"] == set(SCENARIOS), cov["scenarios"]
    assert len(cov["scenarios"]) == 4, cov["scenarios"]


def test_z_scale_none_covered():
    _, _, _, cov = _results()
    assert cov["z_scale_none"] > 0, "no combination exercised z_scale=None"


def test_both_mask_settings_covered():
    _, _, _, cov = _results()
    assert cov["masks_returned"] == {True, False}, cov["masks_returned"]


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
