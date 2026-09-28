#!/usr/bin/env python
"""
Decoder / model integration of the fused composite (`fused_composite` switch, default off).

What is asserted here is wiring, not numerical acceptance:
  * off (default): signature default False, state_dict identical to on, strict cross-loading works,
    the fusion is never called and the original paste (`stn_paste`) is;
  * on: the fusion is used exactly once per decode, the original paste is not, the returned tuple has
    the original order and shapes, dec_objects is unchanged, alpha_masks True/False behave, every input
    and decoder-parameter gradient exists and is finite, and outputs / gradients agree with the off path
    at a coarse level that only catches gross wiring errors (relative L2 <= GROSS_GUARD, fixed in advance,
    ~3 orders above the 1e-6..1e-5 differences already reported for these kernels);
  * unsupported input raises ValueError with a reason (never a silent fallback), and the same input still
    runs with the switch off where the original path supports it.

Fine-grained numerical differences are measured by benchmarks/stn/fused_model_compare.py and only reported.

    python tests/composite/test_decoder_fused_integration.py
"""

import json
import os
import sys
import tempfile
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "stn"))

GROSS_GUARD = 1e-3


class Skip(Exception):
    pass


def _env():
    if not torch.cuda.is_available():
        raise Skip("CUDA is not available")
    try:
        import triton  # noqa: F401
        import build_model  # noqa: F401
        import modules.modules  # noqa: F401
    except ImportError as exc:
        raise Skip(f"dependency missing: {exc}")


def _build(fused, seed=0, config="shapes.json"):
    from build_model import build
    with open(os.path.join(_ROOT, "configs", config)) as f:
        cfg = json.load(f)
    cfg["fused_composite"] = fused
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        torch.manual_seed(seed)
        model, cfg, kwargs = build(path, device="cuda")
    return model, cfg, kwargs


def _pair(seed=0):
    """Two models with identical weights: switch off and on."""
    off, cfg, _ = _build(False, seed)
    on, _, _ = _build(True, seed)
    on.load_state_dict(off.state_dict(), strict=True)
    return off, on, cfg


def _decoder_inputs(dec, bs=3, with_scale=True, seed=1):
    """Same layout decode_all passes to decode_objects: z_features is [bs, n_kp, feat]."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = dec.n_kp_enc

    def r(*shape):
        return torch.rand(*shape, generator=g, device="cuda")

    return {"z_kp": 2 * r(bs, k, 2) - 1,
            "z_features": torch.randn(bs, k, dec.learned_feature_dim, device="cuda", generator=g),
            "obj_on": r(bs, k),
            "z_scale": (4 * r(bs, k, 2) - 2) if with_scale else None,
            "z_depth": 2 * (2 * r(bs, k, 1) - 1)}


def _call(dec, ins, alpha=True, **over):
    a = dict(ins)
    a.update(over)
    return dec.decode_objects(a["z_kp"], a["z_features"], a["obj_on"], z_scale=a["z_scale"],
                              translation=a.get("translation"), z_depth=a["z_depth"], return_alpha_masks=alpha)


def _rel(a, b):
    nb = float(torch.linalg.vector_norm(b.double()))
    return float(torch.linalg.vector_norm(a.double() - b.double())) / nb if nb > 0 else float(torch.linalg.vector_norm(a.double() - b.double()))


def _leaves(ins):
    return {k: (v.detach().clone().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ins.items()}


# --------------------------------------------------------------------------------------------
def test_default_off_signature_and_checkpoint_compatibility():
    _env()
    import inspect
    from models import DLP
    from modules.modules import DLPDecoder
    for cls in (DLP, DLPDecoder):
        p = inspect.signature(cls.__init__).parameters["fused_composite"]
        assert p.default is False, f"{cls.__name__}.fused_composite default is {p.default!r}"
    off, on, _ = _pair()
    assert off.decoder_module.fused_composite is False and on.decoder_module.fused_composite is True
    sd_off, sd_on = off.state_dict(), on.state_dict()
    assert list(sd_off) == list(sd_on) and all(sd_off[k].shape == sd_on[k].shape for k in sd_off)
    assert off.decoder_module.fused_composite_calls == 0 and on.decoder_module.fused_composite_calls == 0
    assert not any("fused" in k for k in sd_off), "the switch must not add parameters or buffers"


def test_off_path_uses_original_paste_and_never_the_fusion():
    _env()
    import modules.modules as M
    off, _, _ = _pair()
    dec = off.decoder_module
    ins = _decoder_inputs(dec)
    with mock.patch.object(M, "composite_fused", side_effect=AssertionError("fusion called with the switch off")), \
            mock.patch.object(M, "stn_paste", wraps=M.stn_paste) as paste:
        _call(dec, ins)
    assert paste.call_count == 1 and dec.fused_composite_calls == 0


def test_on_path_uses_fusion_once_and_skips_the_original_paste():
    _env()
    import modules.modules as M
    _, on, _ = _pair()
    dec = on.decoder_module
    ins = _decoder_inputs(dec)
    for alpha in (True, False):
        before = dec.fused_composite_calls
        with mock.patch.object(M, "stn_paste", side_effect=AssertionError("original paste called with the switch on")):
            out = _call(dec, ins, alpha=alpha)
        assert dec.fused_composite_calls == before + 1
        dec_objects, trans, masks, bg = out
        b, k = ins["z_kp"].shape[:2]
        h = dec.feature_map_size
        assert tuple(dec_objects.shape[:3]) == (b, k, 4) and tuple(trans.shape) == (b, 3, h, h) and tuple(bg.shape) == (b, 1, h, h)
        assert (masks is None) == (not alpha) and (alpha is False or tuple(masks.shape) == (b, k, 1, h, h))


def test_outputs_and_gradients_match_off_path_at_gross_level():
    _env()
    off, on, _ = _pair()
    for with_scale in (True, False):
        for alpha in (True, False):
            ins = _decoder_inputs(off.decoder_module, with_scale=with_scale)
            res = {}
            for name, model in (("off", off), ("on", on)):
                model.zero_grad(set_to_none=True)
                leaves = _leaves(ins)
                dec_objects, trans, masks, bg = _call(model.decoder_module, leaves, alpha=alpha)
                g = torch.Generator(device="cuda").manual_seed(5)
                loss = (torch.rand(bg.shape, generator=g, device="cuda") * bg).sum() + \
                       (torch.rand(trans.shape, generator=g, device="cuda") * trans).sum()
                loss.backward()
                names = [k for k, v in leaves.items() if torch.is_tensor(v) and v.requires_grad]
                res[name] = {"out": (dec_objects, trans, bg) + ((masks,) if alpha else ()),
                             "in_grads": {k: leaves[k].grad for k in names},
                             "params": {n: p.grad for n, p in model.decoder_module.named_parameters()}}
            tag = f"scale={with_scale}/alpha={alpha}"
            assert torch.equal(res["off"]["out"][0], res["on"]["out"][0]), f"{tag}: dec_objects (particle_dec output) must be unchanged"
            for i, (a, b) in enumerate(zip(res["on"]["out"], res["off"]["out"])):
                assert bool(torch.isfinite(a).all()) and _rel(a, b) <= GROSS_GUARD, f"{tag}: output {i} rel-L2 {_rel(a, b):.2e}"
            for k, g_on in res["on"]["in_grads"].items():
                g_off = res["off"]["in_grads"][k]
                assert g_on is not None and g_off is not None, f"{tag}: gradient for input {k} missing"
                assert bool(torch.isfinite(g_on).all()) and _rel(g_on, g_off) <= GROSS_GUARD, f"{tag}: input grad {k} rel-L2 {_rel(g_on, g_off):.2e}"
            n_checked = 0
            for n, g_on in res["on"]["params"].items():
                g_off = res["off"]["params"][n]
                assert (g_on is None) == (g_off is None), f"{tag}: parameter {n} gradient presence differs"
                if g_on is not None:
                    assert bool(torch.isfinite(g_on).all()) and _rel(g_on, g_off) <= GROSS_GUARD, f"{tag}: param grad {n} rel-L2 {_rel(g_on, g_off):.2e}"
                    n_checked += 1
            assert n_checked > 0, "no decoder parameter received a gradient"


def test_unsupported_input_raises_with_reason_and_off_path_still_runs():
    _env()
    off, on, _ = _pair()
    d_on, d_off = on.decoder_module, off.decoder_module
    ins = _decoder_inputs(d_on)
    cases = {
        "translation": (dict(translation=torch.zeros(3, d_on.n_kp_enc, 2, device="cuda")), "translation", True),
        "dtype": (dict(z_depth=ins["z_depth"].double()), "dtype", False),
        "cpu": (dict(obj_on=ins["obj_on"].cpu()), "cpu", False),
    }
    before = d_on.fused_composite_calls
    for name, (over, word, off_ok) in cases.items():
        try:
            _call(d_on, ins, **over)
        except ValueError as exc:
            assert word in str(exc), f"{name}: reason does not mention {word!r}: {exc}"
        else:
            raise AssertionError(f"{name}: unsupported input must raise, not fall back")
        if off_ok:
            _call(d_off, ins, **over)              # the original path supports it
    assert d_on.fused_composite_calls == before, "a rejected call must not count as a fused call"


def test_non_contiguous_z_kp_obj_on_z_depth_z_scale_are_normalized_not_rejected():
    """`_decode_objects_fused` calls .contiguous() on z_kp/obj_on/z_depth/z_scale before the support check (found via
    deterministic=True aliasing mu_* straight from a non-contiguous torch.chunk() view -- see docs/dlp_full_stack_report.md).
    A non-contiguous but otherwise valid input must therefore be ACCEPTED and match the contiguous-input result, not raise."""
    _env()
    _, on, _ = _pair()
    dec = on.decoder_module
    ins = _decoder_inputs(dec)
    noncontig = dict(ins)
    noncontig["z_kp"] = ins["z_kp"].transpose(0, 1).contiguous().transpose(0, 1)
    noncontig["obj_on"] = ins["obj_on"].transpose(0, 1).contiguous().transpose(0, 1)
    noncontig["z_depth"] = ins["z_depth"].transpose(0, 1).contiguous().transpose(0, 1)
    noncontig["z_scale"] = ins["z_scale"].transpose(0, 1).contiguous().transpose(0, 1)
    for k in ("z_kp", "obj_on", "z_depth", "z_scale"):
        assert not noncontig[k].is_contiguous(), f"{k}: test setup did not actually produce a non-contiguous tensor"
    before = dec.fused_composite_calls
    out_noncontig = _call(dec, noncontig)
    assert dec.fused_composite_calls == before + 1, "a normalized (now-contiguous) call must still count as a fused call"
    out_contig = _call(dec, ins)
    for a, b in zip(out_noncontig, out_contig):
        assert torch.equal(a, b), "normalizing a non-contiguous input must not change the result"


def test_full_model_step_uses_fusion_once_and_all_parameters_get_finite_gradients():
    _env()
    off, on, cfg = _pair()
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.rand(2, 1, cfg["ch"], cfg["image_size"], cfg["image_size"], generator=g, device="cuda")
    kw = dict(with_loss=True, return_alpha_masks=False, beta_kl=cfg.get("beta_kl", 0.1), beta_rec=1.0)
    res = {}
    for name, model in (("off", off), ("on", on)):
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        out = model(x, **kw)                       # model_output stays alive until after backward, like train_dlp.py
        loss = out["loss_dict"]["loss"]
        loss.backward()
        assert out["alpha_masks"] is None
        res[name] = (float(loss), {n: p.grad for n, p in model.named_parameters()})
    assert on.decoder_module.fused_composite_calls == 1 and off.decoder_module.fused_composite_calls == 0
    l_off, g_off = res["off"]
    l_on, g_on = res["on"]
    assert abs(l_on - l_off) <= GROSS_GUARD * abs(l_off), (l_on, l_off)
    for n in g_off:
        assert (g_on[n] is None) == (g_off[n] is None), f"gradient presence differs for {n}"
        if g_on[n] is not None:
            assert bool(torch.isfinite(g_on[n]).all()), f"non-finite gradient for {n}"


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures, skipped = [], []
    for name, fn in tests:
        try:
            fn()
        except Skip as exc:
            skipped.append(name)
            print(f"  SKIP {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, str(exc)))
            print(f"  FAIL {name}: {exc}")
        else:
            print(f"  ok   {name}")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} failed, {len(skipped)} skipped, {len(tests)} total")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
