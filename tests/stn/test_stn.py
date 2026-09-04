#!/usr/bin/env python
"""
Reference tests for the isolated STN stage.

What is pinned
--------------
forward  -- every case's output, bit for bit, against the frozen baseline
backward -- every case's input gradients (dL/dimage, dL/dkp, dL/dz_scale ...)
            for a fixed upstream cotangent, bit for bit

Bit-exact, not ``allclose``: stage 1 is an isolation refactor and is required
to change nothing.  A later kernel that legitimately reassociates arithmetic
should be run with ``--tol`` and the loosening recorded deliberately, rather
than by quietly relaxing the default.

What is checked
---------------
1. ``lpwm_stn.reference``      -- the isolated code
2. ``lpwm_stn``                -- the dispatching front-end on the active backend
3. model call sites            -- STN ops are imported from the isolated module,
                                  no raw grid_sample survives in the model, and
                                  ``DLPDecoder.translate_patches`` still matches
4. determinism                 -- two runs of the same case agree byte for byte
5. ``--gradcheck``  (opt-in)   -- float64 ``autograd.gradcheck``, an oracle-free
                                  check of a backend's backward

Usage
-----
    python tests/stn/test_stn.py                       # cpu, small shapes
    python tests/stn/test_stn.py --scope all           # include full config shapes
    python tests/stn/test_stn.py --backend triton      # check a custom kernel
    python tests/stn/test_stn.py --gradcheck           # verify a backward from scratch
    pytest tests/stn/test_stn.py                       # if pytest is installed
"""

import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lpwm_stn
from lpwm_stn import reference
from tests.stn import golden
from tests.stn.gen_golden import fixture_path
from tests.stn.stn_cases import BENCH_CASES, CORRECTNESS_CASES

DEFAULT_DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# fixture loading
# --------------------------------------------------------------------------- #
def load_fixture(device=DEFAULT_DEVICE, scope="all"):
    for candidate in (fixture_path(device, scope), fixture_path(device, "all")):
        if os.path.exists(candidate):
            payload = torch.load(candidate, weights_only=False)
            if payload.get("version") != golden.FIXTURE_VERSION:
                raise RuntimeError(f"{candidate}: fixture version {payload.get('version')} "
                                   f"!= expected {golden.FIXTURE_VERSION}")
            return payload, candidate
    raise FileNotFoundError(
        f"no golden fixture for device '{device}'. Generate one from the frozen baseline:\n"
        f"    python tests/stn/gen_golden.py --device {device}")


def _cases_for(scope):
    return {"all": CORRECTNESS_CASES + BENCH_CASES,
            "correctness": CORRECTNESS_CASES,
            "bench": BENCH_CASES}[scope]


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #
def check_case(case, impl, expected, device, tol=0.0, grad_tol=None):
    """Forward + gradients of one case against its pinned record."""
    if grad_tol is None:
        grad_tol = tol
    ins = case.inputs(device=device)
    out = case.run(impl, ins)

    if tol == 0.0:
        golden.compare(f"{case.name}:forward", expected["out"], out)
    else:
        _compare_tol(f"{case.name}:forward", expected["out"], out, tol)

    if not case.needs_grad:
        return

    cot = case.cotangent(out)
    if golden.describe(cot)["sha256"] != expected["cotangent_sha256"]:
        raise golden.TensorMismatch(
            f"{case.name}: the upstream cotangent itself differs from the one the fixture was "
            f"recorded with -- the gradient comparison would be meaningless")
    wrt = [ins[k] for k in case.grad_wrt]
    grads = torch.autograd.grad(out, wrt, grad_outputs=cot, allow_unused=False)
    for key, g in zip(case.grad_wrt, grads):
        label = f"{case.name}:grad[{key}]"
        if grad_tol == 0.0:
            golden.compare(label, expected["grads"][key], g)
        else:
            _compare_tol(label, expected["grads"][key], g, grad_tol)


def _compare_tol(label, expected, actual, tol):
    """Loosened comparison, only reachable via an explicit ``--tol``."""
    if tuple(actual.shape) != tuple(expected["shape"]):
        raise golden.TensorMismatch(f"{label}: shape {tuple(actual.shape)} != {tuple(expected['shape'])}")
    if "full" not in expected:
        # no stored reference to diff against; fall back to the pinned reductions
        got_sum = float(actual.detach().reshape(-1).double().sum())
        if abs(got_sum - expected["sum_f64"]) > tol * max(1.0, abs(expected["sum_f64"])) * expected["numel"] ** 0.5:
            raise golden.TensorMismatch(f"{label}: sum {got_sum:.9g} outside tol of {expected['sum_f64']:.9g}")
        return
    ref = expected["full"].to(actual.device).float()
    diff = (actual.detach().float() - ref).abs()
    worst = float(diff.max())
    if worst > tol:
        raise golden.TensorMismatch(f"{label}: max|diff|={worst:.3e} > tol={tol:.3e}")


def check_determinism(case, impl, device, gradients=True):
    """The same inputs must produce the same bytes twice -- a kernel with a
    non-deterministic reduction (atomics into dL/dimage) fails here."""
    ins = case.inputs(device=device)
    first = case.run(impl, ins)
    rec = golden.describe(first)
    if case.needs_grad and gradients:
        cot = case.cotangent(first)
        g1 = torch.autograd.grad(first, [ins[k] for k in case.grad_wrt], grad_outputs=cot)
        grec = [golden.describe(g)["sha256"] for g in g1]

    ins2 = case.inputs(device=device)
    second = case.run(impl, ins2)
    golden.compare(f"{case.name}:forward(rerun)", rec, second)
    if case.needs_grad and gradients:
        cot2 = case.cotangent(second)
        g2 = torch.autograd.grad(second, [ins2[k] for k in case.grad_wrt], grad_outputs=cot2)
        for key, ref_sha, g in zip(case.grad_wrt, grec, g2):
            if golden.describe(g)["sha256"] != ref_sha:
                raise golden.TensorMismatch(f"{case.name}:grad[{key}] is not deterministic across runs")


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: files that make up the model; none of them may sample a grid on their own
_MODEL_SOURCES = ("modules/modules.py", "modules/vision_modules.py", "models.py", "utils/util_func.py")

#: which module must import which STN ops from ``lpwm_stn``
_REQUIRED_IMPORTS = {
    "utils/util_func.py": ("affine_grid_sample", "spatial_transform", "create_masks_fast",
                           "create_masks_with_scale"),
    "modules/modules.py": ("stn_crop", "stn_paste", "create_masks_fast", "create_masks_with_scale"),
}


def check_module_call_sites():
    """The model must reach the STN only through ``lpwm_stn``.

    A static, import-free check, so it holds even in a bare kernel-dev
    environment.  Two halves:

    * no raw ``affine_grid`` / ``grid_sample`` anywhere in the model -- if one
      is re-inlined, swapping in a kernel silently stops covering that path;
    * the modules that own the STN call sites still import them from
      ``lpwm_stn``, so the isolation is not quietly undone by a local copy.
    """
    import ast
    import re

    problems = []
    for rel in _MODEL_SOURCES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        source = open(path).read()
        for lineno, line in enumerate(source.splitlines(), 1):
            if re.search(r"F\.(affine_grid|grid_sample)\s*\(", line):
                problems.append(f"raw sampler at {rel}:{lineno}: {line.strip()}")

        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "lpwm_stn":
                imported.update(alias.name for alias in node.names)
        for op in _REQUIRED_IMPORTS.get(rel, ()):
            if op not in imported:
                problems.append(f"{rel} no longer imports '{op}' from lpwm_stn")

    if problems:
        raise AssertionError("the STN stage is no longer isolated:\n  " + "\n  ".join(problems))


class Skipped(Exception):
    """Raised by a check that cannot run here (e.g. a missing optional dependency)."""


def _stub_optional_deps():
    """Let ``modules.modules`` import in a bare kernel-dev environment.

    ``utils.util_func`` drags in plotting/video packages the STN path never
    touches; stubbing the missing ones keeps the call-site check runnable
    without installing the full training environment.
    """
    import tempfile
    import types

    cache_root = os.path.join(tempfile.gettempdir(), "lpwm-test-cache")
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(cache_root, "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_root)

    for name in ("imageio", "cv2"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)


def check_decoder_call_site(device):
    """``DLPDecoder.translate_patches`` must still produce the baseline's bytes.

    The op-level cases pin ``stn_paste``; this pins the *call site*, which is
    where a rewiring mistake actually lands -- a swapped argument, a dropped
    keyword, the wrong canvas size.  ``translate_patches`` reads only
    ``self.feature_map_size``, so no decoder needs to be built.

    The encoder side needs a full CNN to invoke, so it is covered by the static
    import check plus the ``stn_crop`` cases instead.
    """
    _stub_optional_deps()
    try:
        from modules.modules import DLPDecoder
    except ImportError as exc:
        raise Skipped(f"cannot import modules.modules here ({exc})") from exc
    from lpwm_stn import _baseline

    class _Fake:
        feature_map_size = 64

    g = torch.Generator(device=device)
    g.manual_seed(0)
    kp = 2 * torch.rand(3, 7, 2, generator=g, device=device) - 1
    patches = torch.rand(3, 7, 4, 16, 16, generator=g, device=device)
    scale = 4 * torch.rand(3, 7, 2, generator=g, device=device) - 2

    for kwargs in ({"scale": scale},
                   {"scale": None},
                   {"scale": torch.sigmoid(scale), "scale_normalized": True}):
        got = DLPDecoder.translate_patches(_Fake(), kp, patches, **kwargs)
        want = _baseline.stn_paste(kp, patches, _Fake.feature_map_size, **kwargs)
        if not torch.equal(got, want):
            raise golden.TensorMismatch(
                f"DLPDecoder.translate_patches({', '.join(kwargs)}) no longer matches the baseline "
                f"(max|diff|={float((got - want).abs().max()):.3e})")


def check_gradcheck(impl, device):
    """Oracle-free backward check for the caller-typed affine primitive.

    The golden fixtures prove a backend reproduces the *baseline's* gradients.
    This proves the gradients are actually right -- the check a hand-written
    backward kernel needs while it is being written. The fused STN paths are
    deliberately excluded because their baseline theta construction is fp32.
    """
    generator = torch.Generator(device=device).manual_seed(41)
    x = torch.rand(1, 1, 3, 3, generator=generator, device=device,
                   dtype=torch.float64, requires_grad=True)
    theta = torch.tensor([[[0.8, 0.0, 0.1], [0.0, 0.7, -0.2]]], device=device,
                         dtype=torch.float64, requires_grad=True)

    def fn(image, affine):
        return impl.affine_grid_sample(image, affine, (1, 1, 2, 2), "bilinear")

    torch.autograd.gradcheck(fn, (x, theta), eps=1e-6, atol=1e-6, rtol=1e-4,
                             nondet_tol=0.0, raise_exception=True)


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run(device=DEFAULT_DEVICE, scope="correctness", backend=None, tol=0.0, grad_tol=None,
        gradcheck=False, verbose=False, gradient_determinism=True):
    payload, path = load_fixture(device, scope)
    expected_cases = payload["cases"]
    cases = [c for c in _cases_for(scope) if c.name in expected_cases]
    missing = sorted({c.name for c in _cases_for(scope)} - set(expected_cases))

    impls = [("lpwm_stn.reference", reference), ("lpwm_stn (dispatch)", lpwm_stn)]

    print(f"fixture: {os.path.relpath(path)}  (torch {payload['meta']['torch']}, "
          f"baseline {(payload['meta'].get('git_commit') or '?')[:8]})")
    effective_grad_tol = tol if grad_tol is None else grad_tol
    print(f"running torch {torch.__version__} on {device}, backend='{backend or lpwm_stn.get_backend_name()}', "
          f"{len(cases)} cases x {len(impls)} impls, tol={tol}, grad_tol={effective_grad_tol}")
    if missing:
        print(f"note: {len(missing)} case(s) not in the fixture, skipped: {', '.join(missing)}")

    failures = []
    skipped = []

    def attempt(label, fn):
        try:
            fn()
        except Skipped as exc:
            skipped.append((label, str(exc)))
            print(f"  SKIP {label}: {exc}")
        except Exception as exc:  # noqa: BLE001 - the runner reports, it does not handle
            # Do not retain the exception object: its traceback keeps this
            # function's CUDA tensors alive and a sweep of expected failures
            # can otherwise consume the whole GPU before the next case runs.
            message = str(exc)
            failures.append((label, message, traceback.format_exc()))
            print(f"  FAIL {label}: {message}")
        else:
            if verbose:
                print(f"  ok   {label}")

    ctx = lpwm_stn.use_backend(backend) if backend else _null_context()
    with ctx:
        for impl_name, impl in impls:
            print(f"\n[{impl_name}] vs baseline")
            for case in cases:
                attempt(f"{impl_name} {case.name}",
                        lambda c=case, i=impl: check_case(
                            c, i, expected_cases[c.name], device, tol, effective_grad_tol))

        suffix = "" if gradient_determinism else " (gradient rerun skipped explicitly)"
        print(f"\n[determinism] same inputs twice{suffix}")
        for case in cases:
            attempt(f"determinism {case.name}", lambda c=case: check_determinism(
                c, lpwm_stn, device, gradients=gradient_determinism))

        if gradcheck:
            print("\n[gradcheck] float64 affine primitive, microscopic shape")
            attempt("gradcheck affine_grid_sample", lambda: check_gradcheck(lpwm_stn, device))

    print("\n[integration] the model still routes through lpwm_stn")
    attempt("isolation: imports + no raw grid_sample", check_module_call_sites)
    attempt("DLPDecoder.translate_patches call site", lambda: check_decoder_call_site(device))

    total = len(failures)
    tail = f", {len(skipped)} skipped" if skipped else ""
    print(f"\n{'FAILED' if total else 'PASSED'}: {total} failure(s){tail}")
    return 1 if total else 0


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# pytest entry points (collected only when pytest is installed)
# --------------------------------------------------------------------------- #
def _pytest_params():
    try:
        payload, _ = load_fixture(DEFAULT_DEVICE, "correctness")
    except FileNotFoundError:
        return []
    known = payload["cases"]
    return [c for c in CORRECTNESS_CASES if c.name in known]


def test_reference_matches_baseline():
    payload, _ = load_fixture(DEFAULT_DEVICE, "correctness")
    for case in _pytest_params():
        check_case(case, reference, payload["cases"][case.name], DEFAULT_DEVICE)


def test_dispatch_matches_baseline():
    payload, _ = load_fixture(DEFAULT_DEVICE, "correctness")
    for case in _pytest_params():
        check_case(case, lpwm_stn, payload["cases"][case.name], DEFAULT_DEVICE)


def test_deterministic():
    for case in _pytest_params():
        check_determinism(case, lpwm_stn, DEFAULT_DEVICE)


def test_model_routes_through_lpwm_stn():
    check_module_call_sites()
    try:
        check_decoder_call_site(DEFAULT_DEVICE)
    except Skipped:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--scope", default="correctness", choices=("all", "correctness", "bench"))
    ap.add_argument("--backend", default=None, help="STN backend to exercise (default: the active one)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max abs deviation allowed; 0 (default) demands bit-exactness")
    ap.add_argument("--grad-tol", type=float, default=None,
                    help="gradient-only max abs deviation (default: use --tol)")
    ap.add_argument("--gradcheck", action="store_true",
                    help="also run float64 autograd.gradcheck on tiny shapes")
    ap.add_argument("--skip-gradient-determinism", action="store_true",
                    help="keep forward rerun checks but skip CUDA gradient byte checks")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    return run(device=args.device, scope=args.scope, backend=args.backend, tol=args.tol,
               grad_tol=args.grad_tol, gradcheck=args.gradcheck, verbose=args.verbose,
               gradient_determinism=not args.skip_gradient_determinism)


if __name__ == "__main__":
    raise SystemExit(main())
