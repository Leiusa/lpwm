#!/usr/bin/env python
"""
Wiring tests of the opt-in `particle_dec_channels_last` switch (default off; only the particle decoder CNN is affected).

Asserted here (wiring, not numerical acceptance):
  * default False in DLP, DLPDecoder and ObjectDecoderCNN; off keeps standard NCHW conv weights and never takes the path;
  * the switch adds no parameter or buffer: identical state_dict keys/shapes/initial values, strict cross-loading both ways,
    and a loaded model keeps the layout its switch asks for;
  * on: every conv weight is channels_last, each forward takes the path exactly once, the module output is standard
    contiguous NCHW fp32, and outputs / gradients agree with the off path at a coarse level fixed in advance
    (GROSS_GUARD) that only catches wiring errors (two of the ten convs change cuDNN algorithm, so results are not bit-identical);
  * composite_fused still receives standard contiguous input in a full model step and its support check is unchanged
    (a channels_last-strided dec_objects is still rejected);
  * an unsupported combination (context-conditioned FILM particle decoder) raises ValueError.

    python tests/composite/test_particle_dec_channels_last.py
"""

import inspect
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

GROSS_GUARD = 1e-2


class Skip(Exception):
    pass


def _env(cuda=False):
    if cuda and not torch.cuda.is_available():
        raise Skip("CUDA is not available")
    try:
        import build_model  # noqa: F401
        import modules.modules  # noqa: F401
        if cuda:
            import triton  # noqa: F401
    except ImportError as exc:
        raise Skip(f"dependency missing: {exc}")


def _build(channels_last, fused=False, seed=0, device="cpu", config="shapes.json"):
    from build_model import build
    with open(os.path.join(_ROOT, "configs", config)) as f:
        cfg = json.load(f)
    cfg["particle_dec_channels_last"] = channels_last
    cfg["fused_composite"] = fused
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        torch.manual_seed(seed)
        model, cfg, kwargs = build(path, device=device)
    return model, cfg


def _pair(fused=False, device="cpu"):
    off, cfg = _build(False, fused, 0, device)
    on, _ = _build(True, fused, 0, device)
    return off, on, cfg


def _conv_weights(pd):
    return [m.weight for m in pd.modules() if isinstance(m, torch.nn.Conv2d)]


def _rel(a, b):
    nb = float(torch.linalg.vector_norm(b.double()))
    d = float(torch.linalg.vector_norm(a.double() - b.double()))
    return d / nb if nb > 0 else d


def test_default_off_signature():
    _env()
    from models import DLP
    from modules.modules import DLPDecoder, ObjectDecoderCNN
    for cls in (DLP, DLPDecoder):
        assert inspect.signature(cls.__init__).parameters["particle_dec_channels_last"].default is False
    assert inspect.signature(ObjectDecoderCNN.__init__).parameters["channels_last"].default is False
    model, _ = _build(False)
    assert model.decoder_module.particle_dec_channels_last is False and model.decoder_module.particle_dec.channels_last is False


def test_off_keeps_nchw_weights_and_on_converts_every_conv():
    _env()
    off, on, _ = _pair()
    w_off, w_on = _conv_weights(off.decoder_module.particle_dec), _conv_weights(on.decoder_module.particle_dec)
    assert len(w_off) == len(w_on) > 0
    assert all(w.is_contiguous() for w in w_off), "off must keep standard contiguous conv weights"
    assert all(w.is_contiguous(memory_format=torch.channels_last) for w in w_on), "on: every conv weight must be channels_last"
    assert any(not w.is_contiguous() for w in w_on), "on: the layout change must be real (a 3x3 weight is not NCHW-contiguous)"
    # the rest of the model is untouched
    assert all(w.is_contiguous() for m in (on.encoder_module, on.decoder_module.bg_dec) for w in [p for p in m.parameters() if p.dim() == 4])


def test_no_new_parameters_same_initial_values_and_strict_cross_loading():
    _env()
    off, on, _ = _pair()
    sd_off, sd_on = off.state_dict(), on.state_dict()
    assert list(sd_off) == list(sd_on) and all(sd_off[k].shape == sd_on[k].shape for k in sd_off)
    assert all(torch.equal(sd_off[k].contiguous(), sd_on[k].contiguous()) for k in sd_off), "same seed must give identical initial values"
    off2, on2, _ = _pair()
    on2.load_state_dict(off.state_dict(), strict=True)
    off2.load_state_dict(on.state_dict(), strict=True)
    assert all(w.is_contiguous(memory_format=torch.channels_last) for w in _conv_weights(on2.decoder_module.particle_dec))
    assert all(w.is_contiguous() for w in _conv_weights(off2.decoder_module.particle_dec))
    assert all(torch.equal(a.contiguous(), b.contiguous()) for a, b in zip(on2.state_dict().values(), off.state_dict().values()))


def test_module_path_output_contract_and_gross_agreement():
    _env()
    off, on, cfg = _pair()
    pd_off, pd_on = off.decoder_module.particle_dec, on.decoder_module.particle_dec
    pd_on.load_state_dict(pd_off.state_dict())
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 5, pd_off.features_dim, generator=g)
    res = {}
    for name, pd in (("off", pd_off), ("on", pd_on)):
        pd.zero_grad(set_to_none=True)
        xin = x.clone().requires_grad_(True)
        out = pd(xin)
        assert out.dtype == torch.float32 and out.is_contiguous(), f"{name}: output must be standard contiguous fp32"
        (out * torch.linspace(0, 1, out.numel()).view_as(out)).sum().backward()
        res[name] = (out.detach(), xin.grad, {n: p.grad for n, p in pd.named_parameters()})
    assert pd_off.channels_last_calls == 0 and pd_on.channels_last_calls == 1
    assert _rel(res["on"][0], res["off"][0]) <= GROSS_GUARD
    assert _rel(res["on"][1], res["off"][1]) <= GROSS_GUARD
    for n, g_on in res["on"][2].items():
        g_off = res["off"][2][n]
        assert (g_on is None) == (g_off is None), n
        if g_on is not None:
            assert bool(torch.isfinite(g_on).all()) and _rel(g_on, g_off) <= GROSS_GUARD, f"{n}: {_rel(g_on, g_off):.2e}"
            assert g_on.shape == g_off.shape


def test_composite_fused_still_gets_standard_contiguous_input_and_check_is_unchanged():
    _env(cuda=True)
    import modules.modules as M
    from lpwm_stn.composite_autograd import fused_composite_supported
    off, on, cfg = _pair(fused=True, device="cuda")
    on.load_state_dict(off.state_dict(), strict=True)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.rand(2, 1, cfg["ch"], cfg["image_size"], cfg["image_size"], generator=g, device="cuda")
    kw = dict(with_loss=True, return_alpha_masks=False, beta_kl=cfg.get("beta_kl", 0.1), beta_rec=1.0)
    seen = []
    real = M.composite_fused

    def spy(dec_objects, *a, **k):
        seen.append((dec_objects.is_contiguous(), dec_objects.dtype, dec_objects.is_contiguous(memory_format=torch.channels_last) and not dec_objects.is_contiguous()))
        return real(dec_objects, *a, **k)

    res = {}
    with mock.patch.object(M, "composite_fused", side_effect=spy):
        for name, model in (("off", off), ("on", on)):
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(11)
            torch.cuda.manual_seed_all(11)
            out = model(x, **kw)
            loss = out["loss_dict"]["loss"]
            loss.backward()
            res[name] = (float(loss), {n: p.grad for n, p in model.named_parameters()})
    assert len(seen) == 2 and all(c and d == torch.float32 and not cl for c, d, cl in seen), seen
    assert on.decoder_module.fused_composite_calls == 1 and on.decoder_module.particle_dec.channels_last_calls == 1
    l_off, g_off = res["off"]
    l_on, g_on = res["on"]
    assert abs(l_on - l_off) <= GROSS_GUARD * abs(l_off), (l_on, l_off)
    for n in g_off:
        assert (g_on[n] is None) == (g_off[n] is None), f"gradient presence differs for {n}"
        if g_on[n] is not None:
            assert bool(torch.isfinite(g_on[n]).all()), f"non-finite gradient for {n}"
    # the support check itself is unchanged: a channels_last-strided dec_objects is still rejected, not silently accepted
    k = on.decoder_module.n_kp_enc
    d = torch.rand(2, k, 4, 8, 8, device="cuda")
    d_cl = d.permute(0, 1, 3, 4, 2).contiguous().permute(0, 1, 4, 2, 3)   # same shape, channels-last strides
    z_kp, z_scale = torch.zeros(2, k, 2, device="cuda"), torch.ones(2, k, 2, device="cuda")
    obj_on, z_depth = torch.ones(2, k, device="cuda"), torch.zeros(2, k, 1, device="cuda")
    ok, _ = fused_composite_supported(d, z_kp, z_scale, obj_on, z_depth, on.decoder_module.feature_map_size)
    ok_cl, reason = fused_composite_supported(d_cl, z_kp, z_scale, obj_on, z_depth, on.decoder_module.feature_map_size)
    assert ok and not ok_cl and "contiguous" in reason, (ok, ok_cl, reason)


def test_unsupported_combination_raises():
    _env()
    from modules.modules import DLPDecoder
    try:
        DLPDecoder(cdim=3, image_size=64, context_dim=4, decode_with_ctx=True, particle_dec_channels_last=True)
    except ValueError as exc:
        assert "ObjectDecoderCNN" in str(exc)
    else:
        raise AssertionError("particle_dec_channels_last with the FILM particle decoder must raise ValueError")


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
            import traceback
            failures.append((name, str(exc)))
            print(f"  FAIL {name}: {exc!r}")
            traceback.print_exc()
        else:
            print(f"  ok   {name}")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} failed, {len(skipped)} skipped, {len(tests)} total")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
