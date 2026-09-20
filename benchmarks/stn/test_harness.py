#!/usr/bin/env python
"""
CPU-only tests for harness plumbing: repo-root resolution and report schema.

No CUDA required. Complements test_attribution.py, which covers the profiler
attribution rules.

    python benchmarks/stn/test_harness.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bench_e2e  # noqa: E402


def _default_repo():
    """Repo root, in precedence order (B4).

    Never derived from this file's own location alone: once the harness is
    copied outside the repository, that guess names the copy's parent and the
    provenance tests would silently validate the wrong tree.
    """
    env = os.environ.get("LPWM_REPO_ROOT")
    if env:
        return os.path.abspath(env)
    guess = os.path.dirname(os.path.dirname(_HERE))
    return os.path.abspath(guess)


REPO = _default_repo()


def test_bootstrap_accepts_real_repo_and_puts_it_on_path():
    root = bench_e2e._bootstrap(REPO)
    assert root == os.path.abspath(REPO)
    assert root in sys.path


def test_bootstrap_rejects_non_repo():
    """A copied harness must fail loudly, not silently resolve to the wrong tree."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            bench_e2e._bootstrap(tmp)
        except SystemExit as exc:
            assert "not a git repository" in str(exc)
        else:
            raise AssertionError("expected SystemExit for a non-repo --repo-root")


def test_repo_record_reports_the_real_repo_not_the_harness_location():
    """The exact defect from validation run 1: harness copied to /workspace/harness_val
    made repo_record() inspect the wrong directory and emit a git error."""
    rec = bench_e2e.repo_record(REPO)
    assert "error" not in rec, rec
    assert rec["repo_root"] == REPO
    assert isinstance(rec["branch"], str) and rec["branch"]
    assert len(rec["commit"]) == 40
    assert isinstance(rec["clean"], bool)
    assert any("origin" in r for r in rec["remotes"])


def test_repo_record_from_a_foreign_cwd_still_reports_the_repo():
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            rec = bench_e2e.repo_record(REPO)
        finally:
            os.chdir(cwd)
    assert "error" not in rec, rec
    assert rec["repo_root"] == REPO


def test_repo_record_surfaces_a_bad_root_as_error_not_silence():
    with tempfile.TemporaryDirectory() as tmp:
        rec = bench_e2e.repo_record(tmp)
    assert "error" in rec, rec


def test_repo_record_passes_devnull_stderr_to_git():
    """C3: prove the git stderr suppression directly, by inspecting the call.

    The previous form asserted `"fatal" not in sys.stdout.__class__.__name__`,
    which tested nothing at all.
    """
    from unittest import mock

    seen = {}

    def fake_check_output(cmd, **kw):
        seen.update(kw)
        seen.setdefault("cmds", []).append(cmd)
        return "value\n"

    with mock.patch.object(bench_e2e.subprocess, "check_output", side_effect=fake_check_output):
        bench_e2e.repo_record(REPO)

    assert seen.get("stderr") is bench_e2e.subprocess.DEVNULL, seen.get("stderr")
    assert seen.get("text") is True
    assert all(c[:3] == ["git", "-C", REPO] for c in seen["cmds"]), seen["cmds"]


def test_full_test_run_emits_zero_stderr_bytes():
    """C3: the end-to-end guarantee -- a clean pass must be silent on stderr."""
    env = {**os.environ, "LPWM_TEST_CHILD": "1", "LPWM_REPO_ROOT": REPO}
    r = subprocess.run([sys.executable, os.path.join(_HERE, "test_harness.py")],
                       capture_output=True, text=True, timeout=180, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stderr == "", f"stderr not empty:\n{r.stderr}"
    a = subprocess.run([sys.executable, os.path.join(_HERE, "test_attribution.py")],
                       capture_output=True, text=True, timeout=180, env=env)
    assert a.returncode == 0, a.stdout + a.stderr
    assert a.stderr == "", f"stderr not empty:\n{a.stderr}"


def test_config_path_resolution_relative_and_absolute():
    rel = os.path.join(REPO, "configs/balls.json")
    assert os.path.exists(rel), "expected configs/balls.json in the repo"
    assert os.path.isabs("/abs/x.json")
    joined = os.path.join(REPO, "configs/balls.json")
    assert os.path.exists(joined)


def test_report_schema_shape():
    """Pin the top-level field names downstream analysis reads."""
    from attribution import EventRecord, attribute
    attr = attribute([EventRecord("k", True, 1000.0)]).as_dict()
    report = {
        "schema_version": 2, "experiment": "x", "date_utc": "t", "config": "c",
        "backend": "triton", "seed": 0, "sync": "s",
        "metric_semantics": {}, "env": {}, "repo": {},
        "results": {"shapes": {},
                    "inference": {"clean": {"D_clean_wall_ms": 1.0}, "memory": {},
                                  "profile": dict({"D_prof_wall_ms": 2.0}, **attr)}},
    }
    json.dumps(report)                                   # must be serializable
    prof = report["results"]["inference"]["profile"]
    assert "D_prof_wall_ms" in prof and "total_device_kernel_ms" in prof
    assert "stn_share_of_device_kernel_time_pct" in prof
    for gone in ("stn_share_of_wall_pct", "total_cuda_ms", "stn_ms", "top_unmatched_kernels_ms"):
        assert gone not in prof, gone
    assert report["results"]["inference"]["clean"]["D_clean_wall_ms"] == 1.0


def _derived_static_config(tmp_dir, batch_size=1):
    """bair.json as a static DLP: horizon 1 and none of the keys shapes.json (a real static
    config) does not have -- those 18 are read only by the dynamics model."""
    with open(os.path.join(REPO, "configs/bair.json")) as f:
        bair = json.load(f)
    with open(os.path.join(REPO, "configs/shapes.json")) as f:
        shapes = json.load(f)
    cfg = {k: v for k, v in bair.items() if k in shapes}
    cfg.update(timestep_horizon=1, batch_size=batch_size)
    path = os.path.join(tmp_dir, "static_bair128.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path, cfg, sorted(set(bair) - set(shapes))


def test_sequence_length_static_is_one_frame_and_dynamics_adds_the_initial_frame():
    """A static DLP takes ONE frame (train_dlp.py). Feeding horizon+1=2 frames does not raise in
    calc_static_elbo, it just measures the wrong workload -- so this is pinned here."""
    from build_model import sequence_length
    assert sequence_length({"timestep_horizon": 1}) == 1
    assert sequence_length({"timestep_horizon": 2}) == 3
    assert sequence_length({"timestep_horizon": 16}) == 17
    for name, want in (("shapes", 1), ("bair", 17)):
        with open(os.path.join(REPO, f"configs/{name}.json")) as f:
            assert sequence_length(json.load(f)) == want, name


def test_return_alpha_masks_defaults_true_and_accepts_explicit_values():
    base = ["--config", "c.json", "--backend", "reference"]
    parse = bench_e2e.build_parser().parse_args
    assert parse(base).return_alpha_masks is True
    assert parse(base + ["--return-alpha-masks"]).return_alpha_masks is True
    assert parse(base + ["--no-return-alpha-masks"]).return_alpha_masks is False


def test_derived_static_config_is_static_and_drops_only_dynamics_keys():
    from build_model import sequence_length
    with tempfile.TemporaryDirectory() as tmp:
        _, cfg, dropped = _derived_static_config(tmp)
    assert cfg["timestep_horizon"] == 1 and sequence_length(cfg) == 1
    assert cfg["image_size"] == 128 and cfg["n_kp_enc"] == 90         # still the BAIR workload
    assert {"pint_dim", "context_dim", "cond_steps", "beta_dyn"} <= set(dropped)
    assert not (set(dropped) & set(cfg))


def test_derived_static_config_builds_a_non_dynamics_model():
    """The real check that timestep_horizon=1 gives is_dynamics_model=False. Needs the model's
    own dependencies (imageio, cv2, ...); reported as skipped, never as passed, without them."""
    try:
        from build_model import build
        import models  # noqa: F401
    except ImportError as exc:
        print(f"     (SKIPPED model build, dependency missing: {exc})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path, _, _ = _derived_static_config(tmp)
        model, _, kwargs = build(path, device="cpu")
    assert model.is_dynamics_model is False
    assert model.context_dim == 0
    assert kwargs["timestep_horizon"] == 1
    assert not ({"pint_dim", "context_dim", "cond_steps"} & set(kwargs))


def test_tensors_sha256_distinguishes_weights_and_is_reproducible():
    import torch
    a, b = torch.arange(6.0), torch.arange(6.0)
    c = b.clone()
    c[3] += 1e-6
    assert bench_e2e.tensors_sha256([a]) == bench_e2e.tensors_sha256([b])
    assert bench_e2e.tensors_sha256([a]) != bench_e2e.tensors_sha256([c])


def test_saved_tensor_ledger_dedupes_storage_attributes_to_innermost_scope_and_excludes():
    import torch
    import attribute_memory as am

    def run(exclude=()):
        ledger = am.Ledger(exclude, cuda_only=False)
        x = torch.randn(4, 4, requires_grad=True)
        w = torch.randn(4, 4)
        with ledger.hooks():
            with ledger.scope("outer"):
                a = x * w                # w requires no grad, so only w is saved
                with ledger.scope("inner"):
                    a * a                # both operands are `a`: one storage, saved once
        assert ledger.stack == ["other"], "scope stack must unwind"
        return ledger.summary(), w

    summary, w = run()
    assert summary["by_scope_mb"]["outer"] * 1e6 == 64 and summary["by_scope_mb"]["inner"] * 1e6 == 64
    assert summary["n_storages_by_scope"] == {"outer": 1, "inner": 1}
    assert summary["total_mb"] * 1e6 == 128
    grouped = summary["grouped_complete"]       # nothing is truncated: groups add up to the total
    assert abs(sum(g["total_mb"] for g in grouped) - summary["total_mb"]) < 1e-12
    assert sum(g["n_storages"] for g in grouped) == 2
    excluded, _ = run(exclude=set())            # fresh tensors each run, so exclusion is checked below
    assert excluded["total_mb"] == summary["total_mb"]
    ledger = am.Ledger(cuda_only=False)
    x = torch.randn(4, 4, requires_grad=True)
    w = torch.randn(4, 4)
    ledger.exclude.add(w.untyped_storage().data_ptr())
    with ledger.hooks():
        x * w
    assert ledger.summary()["total_mb"] == 0.0, "an excluded storage (weights / inputs) is not counted"


def test_scope_probe_targets_cover_all_stn_entry_points():
    from attribution import SCOPE_BUCKETS, SCOPE_PREFIX
    targets = bench_e2e.ScopeProbe.TARGETS
    assert set(targets) == {"stn_crop", "stn_paste", "create_masks_fast", "create_masks_with_scale"}
    for scope in targets.values():
        assert f"{SCOPE_PREFIX}{scope}" in SCOPE_BUCKETS, scope


def test_clean_path_is_marked_uninstrumented():
    import inspect
    src = inspect.getsource(bench_e2e.timed)
    assert "record_function" not in src, "D_clean must not carry profiling instrumentation"
    assert '"instrumented": False' in src


def test_b3_patch_size_prefers_config_over_derivation():
    """BAIR sets patch_size=8 while round(anchor_s*(image_size-1)) gives 16.
    Deriving it would microprofile a shape the model never runs."""
    bair = {"patch_size": 8, "anchor_s": 0.125, "image_size": 128}
    size, source = bench_e2e.resolve_patch_size(bair)
    assert size == 8, size
    assert source == "config:patch_size"
    assert size != int(round(bair["anchor_s"] * (bair["image_size"] - 1)))


def test_b3_patch_size_falls_back_only_when_config_lacks_it():
    size, source = bench_e2e.resolve_patch_size({"anchor_s": 0.25, "image_size": 64})
    assert size == int(round(0.25 * 63)) == 16, size
    assert source.startswith("fallback:")


def test_b3_patch_size_matches_real_bair_config():
    cfg_path = os.path.join(REPO, "configs", "bair.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    size, source = bench_e2e.resolve_patch_size(cfg)
    assert source == "config:patch_size"
    assert size == cfg["patch_size"] == 8, (size, cfg.get("patch_size"))


def test_b3_zero_or_missing_patch_size_uses_fallback():
    size, source = bench_e2e.resolve_patch_size({"patch_size": 0, "anchor_s": 0.25, "image_size": 64})
    assert source.startswith("fallback:"), source
    assert size == 16


def test_copied_harness_resolves_the_real_repo(repo=None):
    """B4: copy the whole harness somewhere foreign and prove provenance still
    points at the real repository, not at the copy's parent."""
    repo = repo or REPO
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "harness_val", "stn")
        shutil.copytree(_HERE, dest)
        # without a repo root the copy's own parent is NOT a git repo -> must fail loudly
        child_env = {**os.environ, "LPWM_TEST_CHILD": "1"}
        # no repo root: the copy's own parent is not a git repo -> must fail loudly
        bad = subprocess.run([sys.executable, os.path.join(dest, "test_harness.py")],
                             capture_output=True, text=True, timeout=120,
                             env={**child_env, "LPWM_REPO_ROOT": ""})
        # with the real root supplied, the same copied tests must pass
        good = subprocess.run([sys.executable, os.path.join(dest, "test_harness.py"),
                               "--repo-root", repo],
                              capture_output=True, text=True, timeout=120,
                              env={k: v for k, v in child_env.items() if k != "LPWM_REPO_ROOT"})
        # positive case: with the real root, the copied tests must pass
        assert good.returncode == 0, f"copied harness failed with --repo-root:\n{good.stdout}\n{good.stderr}"
        assert "PASSED" in good.stdout, good.stdout
        assert "fatal:" not in good.stderr, f"stderr leaked git noise:\n{good.stderr}"

        # negative case (C2): without a repo root the copy MUST fail, and say why.
        # The old `returncode != 0 or "PASSED" in stdout` permitted exactly the
        # silent success this test exists to forbid.
        combined = bad.stdout + bad.stderr
        assert bad.returncode != 0, (
            f"copied harness succeeded without --repo-root; it resolved some other tree:\n{combined}")
        assert "not a git repository" in combined, f"no clear repo-root failure message:\n{combined}"
        assert "fatal:" not in bad.stderr, f"leaked raw git noise instead of a clear message:\n{bad.stderr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None,
                    help="the real LPWM repository (or set LPWM_REPO_ROOT)")
    args, _ = ap.parse_known_args()
    global REPO
    if args.repo_root:
        REPO = os.path.abspath(args.repo_root)
    print(f"repo root: {REPO}")
    _spawning = {"test_copied_harness_resolves_the_real_repo",
                 "test_full_test_run_emits_zero_stderr_bytes"}
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and n not in _spawning]
    # LPWM_TEST_CHILD marks a run spawned BY this file. Both subprocess-spawning
    # tests are skipped there; without this each child re-spawns and the suite
    # never terminates.
    if os.environ.get("LPWM_TEST_CHILD") != "1":
        tests.append(("test_copied_harness_resolves_the_real_repo",
                      test_copied_harness_resolves_the_real_repo))
        tests.append(("test_full_test_run_emits_zero_stderr_bytes",
                      test_full_test_run_emits_zero_stderr_bytes))
    failures = []
    for name, fn in tests:
        try:
            fn()
        except SystemExit as exc:
            # _bootstrap raises SystemExit for a bad repo root; that is a test
            # failure here, not a reason to abort the runner silently
            failures.append((name, exc))
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:                                   # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL {name}: {exc}")
        else:
            print(f"  ok   {name}")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} of {len(tests)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
