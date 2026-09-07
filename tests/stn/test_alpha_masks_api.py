#!/usr/bin/env python
"""
Tests for the ``return_alpha_masks`` API (commit 2).

The flag makes the per-particle ``[bs, n_kp, 1, H, W]`` alpha stack optional.
It is an output artifact only -- bound but never read in either ELBO path -- so
disabling it must change *nothing* else. These tests hold it to that: bit-exact,
not ``allclose``, because there is no reassociation to tolerate. Any drift is a
bug, never a reason to loosen a tolerance.

Covered:
  * True reproduces the current tensor; False returns None in the same position
  * every other returned tensor, the scalar loss and every loss_dict entry are
    bit-identical, in BOTH the dynamic and the static ELBO configuration
  * input gradients and ALL parameter gradients are bit-identical, including the
    None / non-None structure
  * the disabled branch structurally does not execute the final materialization
    -- proved by counting dispatched ops, not by a memory delta

    python tests/stn/test_alpha_masks_api.py
    pytest tests/stn/test_alpha_masks_api.py
"""

import inspect
import json
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

for _dep in ("imageio", "cv2"):        # plotting/IO deps the STN path never touches
    try:
        __import__(_dep)
    except ImportError:
        sys.modules[_dep] = types.ModuleType(_dep)

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from models import DLP  # noqa: E402
from modules.modules import DLPDecoder  # noqa: E402

#: config keys whose name differs from the DLP constructor parameter
_RENAMES = {"cdim": "ch", "n_static_frames": "num_static_frames", "ctx_dist": "context_dist"}

#: small enough to build and run twice per test on CPU in well under a second,
#: while still exercising the real decode -> composite -> ELBO chain
_TINY = dict(image_size=32, n_kp_enc=4, n_kp_prior=16, patch_size=8, anchor_s=0.25,
             learned_feature_dim=3, pint_dim=32, pint_dyn_layers=1, pint_dyn_heads=2,
             pint_ctx_layers=1, pint_ctx_heads=2, pint_enc_layers=1, pint_enc_heads=1,
             obj_base_ch=8, obj_final_cnn_ch=8, bg_base_ch=8, bg_final_cnn_ch=8,
             obj_res_from_fc=4, bg_res_from_fc=4, num_res_blocks=1, mlp_hidden_dim=32)

SEED = 0


def _build(timestep_horizon):
    """Identical weights every call: the seed is set immediately before construction."""
    cfg = json.load(open(os.path.join(_ROOT, "configs", "balls.json")))
    cfg.update(_TINY)
    cfg["timestep_horizon"] = timestep_horizon
    params = [p for p in inspect.signature(DLP.__init__).parameters if p != "self"]
    kw = {}
    for p in params:
        if p in cfg:
            kw[p] = cfg[p]
        elif p in _RENAMES and _RENAMES[p] in cfg:
            kw[p] = cfg[_RENAMES[p]]
    for k in ("obj_ch_mult", "obj_ch_mult_prior", "bg_ch_mult"):
        if isinstance(kw.get(k), list):
            kw[k] = tuple(kw[k])
    torch.manual_seed(SEED)
    return DLP(**kw), cfg


def _make_input(cfg, timestep_horizon):
    # dynamic models consume an initial frame plus `timestep_horizon` steps;
    # the static model (timestep_horizon == 1) consumes a single frame
    seq = timestep_horizon + 1 if timestep_horizon > 1 else 1
    g = torch.Generator()
    g.manual_seed(SEED + 101)
    return torch.rand(1, seq, cfg["ch"], cfg["image_size"], cfg["image_size"], generator=g)


def _run(timestep_horizon, return_alpha_masks, backward=False):
    """One full forward (and optional backward) at a given flag value."""
    model, cfg = _build(timestep_horizon)
    model.train() if backward else model.eval()
    x = _make_input(cfg, timestep_horizon)
    x.requires_grad_(backward)
    torch.manual_seed(SEED + 7)          # identical sampling noise for both flag values
    out = model(x, deterministic=True, with_loss=True, return_alpha_masks=return_alpha_masks)
    loss = out["loss_dict"]["loss"]
    grads = {}
    if backward:
        loss.backward()
        grads = {n: (None if p.grad is None else p.grad.detach().clone())
                 for n, p in model.named_parameters()}
        grads["__input__"] = None if x.grad is None else x.grad.detach().clone()
    return out, loss.detach().clone(), grads


def _assert_bit_identical(a, b, label):
    if a is None or b is None:
        assert a is None and b is None, f"{label}: None/tensor mismatch ({a is None} vs {b is None})"
        return
    assert a.shape == b.shape, f"{label}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    assert a.dtype == b.dtype, f"{label}: dtype {a.dtype} != {b.dtype}"
    assert torch.equal(a, b), (
        f"{label}: NOT bit-identical, max|diff|={float((a - b).abs().max()):.3e}")


# --------------------------------------------------------------------------- #
# structural proof: the disabled branch does not execute the materialization
# --------------------------------------------------------------------------- #
class _OpCounter(TorchDispatchMode):
    """Counts dispatched aten ops. A code-path assertion, not a memory heuristic."""

    def __init__(self):
        self.counts = {}

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        name = str(func)
        self.counts[name] = self.counts.get(name, 0) + 1
        return func(*args, **(kwargs or {}))

    def total(self, needle):
        return sum(v for k, v in self.counts.items() if needle in k)


def _composite_inputs(bs=2, n_kp=3, size=8):
    g = torch.Generator()
    g.manual_seed(SEED)
    return dict(
        a_obj=torch.rand(bs, n_kp, 1, size, size, generator=g),
        rgb_obj=torch.rand(bs, n_kp, 3, size, size, generator=g),
        obj_on=torch.rand(bs, n_kp, generator=g),
        z_depth=torch.rand(bs, n_kp, 1, generator=g),
    )


class _FakeDecoder:
    """`get_objects_alpha_rgb_with_depth` reads no attribute of self."""


def test_structural_disabled_branch_skips_one_multiply():
    """The ONLY difference must be a single elementwise multiply.

    `alpha_mask` forms its own `importance_map * a_obj` product and reduces it,
    so the skipped line is exactly the retained per-particle stack and nothing else.
    """
    ins = _composite_inputs()
    counts = {}
    for flag in (True, False):
        counter = _OpCounter()
        with counter:
            DLPDecoder.get_objects_alpha_rgb_with_depth(
                _FakeDecoder(), **{k: v.clone() for k, v in ins.items()},
                return_alpha_masks=flag)
        counts[flag] = counter

    mul_true, mul_false = counts[True].total("mul"), counts[False].total("mul")
    assert mul_true - mul_false == 1, (
        f"expected exactly one fewer multiply when disabled, got {mul_true} vs {mul_false}")
    for needle in ("sum", "sigmoid", "div", "add"):
        assert counts[True].total(needle) == counts[False].total(needle), (
            f"'{needle}' op count changed: {counts[True].total(needle)} vs {counts[False].total(needle)}")


def test_structural_source_guard_wraps_only_the_materialization():
    """Static check that the guard encloses the assignment and nothing else."""
    src = inspect.getsource(DLPDecoder.get_objects_alpha_rgb_with_depth)
    assert "if return_alpha_masks:" in src
    assert "a_obj = importance_map * a_obj" in src
    # the reduced outputs must be computed OUTSIDE the guard
    guard_at = src.index("if return_alpha_masks:")
    before = src[:guard_at]
    for essential in ("rgba_obj = a_obj * rgb_obj",
                      "importance_map = a_obj * torch.sigmoid",
                      "dec_objects_trans = (rgba_obj * importance_map).sum(dim=1)",
                      "alpha_mask = 1.0 - (importance_map * a_obj).sum(dim=1)"):
        assert essential in before, f"{essential!r} must be computed before the guard"


def test_composite_outputs_bit_identical_except_alpha_masks():
    ins = _composite_inputs()
    a_t, mask_t, trans_t = DLPDecoder.get_objects_alpha_rgb_with_depth(
        _FakeDecoder(), **{k: v.clone() for k, v in ins.items()}, return_alpha_masks=True)
    a_f, mask_f, trans_f = DLPDecoder.get_objects_alpha_rgb_with_depth(
        _FakeDecoder(), **{k: v.clone() for k, v in ins.items()}, return_alpha_masks=False)
    assert a_t is not None and a_f is None
    _assert_bit_identical(mask_t, mask_f, "composite bg_mask")
    _assert_bit_identical(trans_t, trans_f, "composite dec_objects_trans")


def test_default_is_true_across_the_whole_chain():
    """Default True must preserve the current API exactly."""
    for fn in (DLPDecoder.get_objects_alpha_rgb_with_depth, DLPDecoder.decode_objects,
               DLPDecoder.decode_all, DLPDecoder.forward, DLP.decode_all, DLP.forward):
        p = inspect.signature(fn).parameters.get("return_alpha_masks")
        assert p is not None, f"{fn.__qualname__} does not accept return_alpha_masks"
        assert p.default is True, f"{fn.__qualname__} default is {p.default!r}, must be True"


# --------------------------------------------------------------------------- #
# model-level equivalence, in BOTH ELBO configurations
# --------------------------------------------------------------------------- #
#: timestep_horizon -> which ELBO branch DLP.calc_elbo takes
#: (`is_dynamics_model = timestep_horizon > 1`)
_ELBO_MODES = ((2, "dynamic"), (1, "static"))


def _check_forward_equivalence(timestep_horizon, label):
    out_t, loss_t, _ = _run(timestep_horizon, True)
    out_f, loss_f, _ = _run(timestep_horizon, False)

    assert out_t["alpha_masks"] is not None, f"{label}: True must return the tensor"
    assert out_f["alpha_masks"] is None, f"{label}: False must return None in that position"
    assert "alpha_masks" in out_f, f"{label}: the key must remain present"

    _assert_bit_identical(loss_t, loss_f, f"{label} scalar loss")

    ld_t, ld_f = out_t["loss_dict"], out_f["loss_dict"]
    assert set(ld_t) == set(ld_f), f"{label}: loss_dict keys differ"
    for k in sorted(ld_t):
        if torch.is_tensor(ld_t[k]):
            _assert_bit_identical(ld_t[k], ld_f[k], f"{label} loss_dict[{k}]")
        else:
            assert ld_t[k] == ld_f[k], f"{label} loss_dict[{k}]: {ld_t[k]} != {ld_f[k]}"

    assert set(out_t) == set(out_f), f"{label}: output keys differ"
    compared = 0
    for k in sorted(out_t):
        if k in ("alpha_masks", "loss_dict"):
            continue
        if torch.is_tensor(out_t[k]):
            _assert_bit_identical(out_t[k], out_f[k], f"{label} output[{k}]")
            compared += 1
        else:
            assert type(out_t[k]) is type(out_f[k]), f"{label} output[{k}] type differs"
    assert compared > 20, f"{label}: only {compared} tensors compared, expected the full output dict"


def test_forward_equivalence_dynamic_elbo():
    _check_forward_equivalence(2, "dynamic")


def test_forward_equivalence_static_elbo():
    _check_forward_equivalence(1, "static")


def _check_gradient_equivalence(timestep_horizon, label):
    _, loss_t, g_t = _run(timestep_horizon, True, backward=True)
    _, loss_f, g_f = _run(timestep_horizon, False, backward=True)

    _assert_bit_identical(loss_t, loss_f, f"{label} loss (backward run)")
    assert set(g_t) == set(g_f), f"{label}: parameter sets differ"

    none_t = {n for n, g in g_t.items() if g is None}
    none_f = {n for n, g in g_f.items() if g is None}
    assert none_t == none_f, (
        f"{label}: None/non-None gradient structure differs; "
        f"only-True={sorted(none_t - none_f)} only-False={sorted(none_f - none_t)}")

    nonzero = 0
    for name in sorted(g_t):
        _assert_bit_identical(g_t[name], g_f[name], f"{label} grad[{name}]")
        if g_t[name] is not None and g_t[name].abs().sum() > 0:
            nonzero += 1
    assert nonzero > 10, f"{label}: only {nonzero} non-trivial gradients, backward may not have run"
    assert g_t["__input__"] is not None, f"{label}: input gradient missing"


def test_gradient_equivalence_dynamic_elbo():
    _check_gradient_equivalence(2, "dynamic")


def test_gradient_equivalence_static_elbo():
    _check_gradient_equivalence(1, "static")


def test_disabled_run_never_builds_the_per_particle_stack_in_the_model():
    """Model-level counterpart of the structural test: one fewer multiply of the
    per-particle shape reaches the dispatcher across a whole forward."""
    shapes = {}
    for flag in (True, False):
        model, cfg = _build(2)
        model.eval()
        x = _make_input(cfg, 2)
        seen = []

        class _ShapeWatcher(TorchDispatchMode):
            def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if "mul" in str(func) and torch.is_tensor(out) and out.dim() == 5:
                    seen.append(tuple(out.shape))
                return out

        torch.manual_seed(SEED + 7)
        with torch.no_grad(), _ShapeWatcher():
            model(x, deterministic=True, with_loss=False, return_alpha_masks=flag)
        shapes[flag] = seen

    n_kp = 4
    per_particle = [s for s in shapes[True] if len(s) == 5 and s[1] == n_kp and s[2] == 1]
    per_particle_f = [s for s in shapes[False] if len(s) == 5 and s[1] == n_kp and s[2] == 1]
    assert len(per_particle) - len(per_particle_f) == 1, (
        f"expected exactly one fewer per-particle multiply when disabled, "
        f"got {len(per_particle)} vs {len(per_particle_f)}")


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                                    # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL {name}: {exc}")
        else:
            print(f"  ok   {name}")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} of {len(tests)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
