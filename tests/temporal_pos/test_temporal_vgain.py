#!/usr/bin/env python
"""
Tests of the opt-in `temporal_vgain` (per-offset, per-dimension value gain in the temporal attention of the
context/dynamics transformers; modules.TemporalValueGain), used with temporal_pos 'rel_bias' / 'none'.

Asserted here (CPU, random weights):
  1. strict checks: temporal_pos 'abs_adaln', non-bool, non-causal transformer, together with temporal_rope ->
     ValueError;
  2. off by default: no key == explicit False (same parameters, same outputs, bitwise); on adds only the gain
     parameters (one zero `delta` per temporal attention);
  3. zero init is the identity, bitwise: a 'rel_bias' / 'none' model and the same weights with temporal_vgain give
     the same training forward and the same rollouts (recompute, kv cache, sliding cache);
  4. the gradient of the loss reaches delta at zero init;
  5. the correction equals a naive double loop over query/key timesteps (y_i = sum_j a_ij (1 + delta[i-j]) v_j);
  6. with non-zero delta: outputs change, and depend only on time offsets (a kv cache whose positions start at 37
     gives the same outputs as one starting at 0); kv cache == full forward within the block size;
  7. one layer: sliding cache == recomputed window past the block size (transformer and whole model).

    python tests/temporal_pos/test_temporal_vgain.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from test_temporal_pos import build, frames, perturb  # noqa: E402

EXACT_TOL = 1e-5
SENS = 1e-4
LATENTS = ("z_pos", "z_scale", "z_obj_on", "z_depth", "z_features", "z_bg_features", "z_context")


def vgain_params(m):
    return {k: p for k, p in m.named_parameters() if ".vgain." in k}


def set_vgain(m, seed=3, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in vgain_params(m).values():
            p.copy_(scale * torch.randn(p.shape, generator=g))


def rollout(m, x, cond, steps, kv):
    with torch.no_grad():
        _, z = m.sample_from_x(x, num_steps=steps, deterministic=True, cond_steps=cond, return_z=True,
                               decode=False, kv_cache=kv)
    return z


def pair(mode, extra=None):
    base, cfg = build(mode, extra=extra)
    perturb(base)
    vg, _ = build(mode, extra={**(extra or {}), "temporal_vgain": True})
    missing, unexpected = vg.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert missing and all(k.endswith(".vgain.delta") for k in missing), missing
    assert all((p == 0).all() for p in vgain_params(vg).values())
    return base, vg, cfg


def train_out(m, x, seed=0):
    m.train()
    torch.manual_seed(seed)
    out = m(x)
    m.eval()
    return out


def test_strict_checks():
    from modules.modules import ParticleSpatioTemporalTransformer
    bad = [
        lambda: build("abs_adaln", extra={"temporal_vgain": True}),
        lambda: build("rel_bias", extra={"temporal_vgain": 1}),
        lambda: build("rel_bias", extra={"temporal_vgain": True, "temporal_rope": True}),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, pos_embed_t_adaln=True, temporal_vgain=True),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, temporal_pos="rel_bias", causal=False,
                                                  temporal_vgain=True),
    ]
    for i, f in enumerate(bad):
        try:
            f()
        except ValueError:
            continue
        raise AssertionError(f"case {i} did not raise ValueError")


def test_default_off():
    a, cfg = build("rel_bias")
    b, _ = build("rel_bias", extra={"temporal_vgain": False})
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not vgain_params(a)
    x = frames(cfg, 4)
    za, zb = rollout(a, x, 2, 2, False), rollout(b, x, 2, 2, False)
    assert all(torch.equal(za[k], zb[k]) for k in LATENTS)
    v, _ = build("rel_bias", extra={"temporal_vgain": True})
    extra = set(v.state_dict()) - set(sa)
    assert extra and all(".vgain." in k for k in extra) and set(sa) <= set(v.state_dict())
    n_attn = len(vgain_params(v))  # unique parameters (the context module is registered under several names)
    assert n_attn == cfg["pint_dyn_layers"] + cfg["pint_ctx_layers"], n_attn


def test_zero_init_is_identity():
    for mode in ("rel_bias", "none"):
        base, vg, cfg = pair(mode)
        x = frames(cfg, cfg["timestep_horizon"] + 1)
        oa, ob = train_out(base, x), train_out(vg, x)
        keys = [k for k, v in oa.items() if torch.is_tensor(v)]
        assert keys and all(torch.equal(oa[k], ob[k]) for k in keys), mode
        cond, steps = 2, 10  # past the block size
        x = frames(cfg, cond + steps)
        for kv in (False, True, "slide"):
            za, zb = rollout(base, x, cond, steps, kv), rollout(vg, x, cond, steps, kv)
            for k in LATENTS:
                assert torch.equal(za[k], zb[k]), (mode, kv, k)


def test_gradient_reaches_delta_at_zero_init():
    _, vg, cfg = pair("rel_bias")
    vg.train()
    torch.manual_seed(0)
    out = vg(frames(cfg, cfg["timestep_horizon"] + 1))
    loss = sum(v.float().mean() for v in out.values() if torch.is_tensor(v) and v.dtype.is_floating_point
               and v.requires_grad and v.dim() > 0)
    loss.backward()
    for k, p in vgain_params(vg).items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), k
        assert p.grad.abs().max() > 0, k


def test_correction_matches_naive_loop():
    from modules.modules import TemporalValueGain
    g = torch.Generator().manual_seed(1)
    B, nh, T, hs, W = 2, 3, 6, 4, 8
    m = TemporalValueGain(nh, hs, W)
    with torch.no_grad():
        m.delta.copy_(torch.randn(m.delta.shape, generator=g))
    att = torch.softmax(torch.randn(B, nh, T, T, generator=g)
                        .masked_fill(torch.ones(T, T).triu(1).bool(), float("-inf")), -1)
    v = torch.randn(B, nh, T, hs, generator=g)
    ref = torch.zeros(B, nh, T, hs)
    for i in range(T):
        for j in range(i + 1):
            ref[:, :, i] += att[:, :, i, j, None] * m.delta[None, :, i - j] * v[:, :, j]
    d1 = (m.correction(att, v) - ref).abs().max().item()
    d2 = (m.correction_cached(att, v, torch.arange(T) + 5, torch.arange(T) + 5) - ref).abs().max().item()
    assert d1 < 1e-6 and d2 < 1e-6, (d1, d2)


def test_live_and_offset_only():
    from modules.modules import ParticleSpatioTemporalTransformer
    b, n, T, f = 2, 3, 6, 32
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, n, T, f, generator=g)
    c = torch.randn(b, n, T, f, generator=g)
    torch.manual_seed(0)
    pt = ParticleSpatioTemporalTransformer(f, 4, 2, T, f, context_cond=True, temporal_pos="rel_bias",
                                           temporal_vgain=True).eval()
    perturb(pt)
    with torch.no_grad():
        for p in vgain_params(pt).values():
            p.zero_()
        y0 = pt(x, c)
        set_vgain(pt)
        y1 = pt(x, c)
        assert (y1 - y0).abs().max() > SENS, "non-zero gains did not change the output"
        outs = {}
        for start in (0, 37):
            cache = pt.new_kv_cache()
            for kv in cache["layers"]:
                kv["p"] = start
            outs[start] = torch.cat([pt(x[:, :, i:i + 1], c[:, :, i:i + 1], kv_cache=cache) for i in range(T)], 2)
    d_off = (outs[0] - outs[37]).abs().max().item()
    d_full = (outs[0] - y1).abs().max().item()
    assert d_off < EXACT_TOL, d_off
    assert d_full < EXACT_TOL, d_full
    print(f"    positions from 37 vs from 0: {d_off:.1e}; kv cache vs full forward: {d_full:.1e}")


def test_slide_one_layer_exact():
    from modules.modules import ParticleSpatioTemporalTransformer
    b, n, T, f = 2, 3, 5, 32
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, n, 2 * T, f, generator=g)
    c = torch.randn(b, n, 2 * T, f, generator=g)
    for mode in ("rel_bias", "none"):
        torch.manual_seed(0)
        pt = ParticleSpatioTemporalTransformer(f, 4, 1, T, f, context_cond=True, temporal_pos=mode,
                                               temporal_vgain=True).eval()
        perturb(pt)
        set_vgain(pt)
        cache = pt.new_kv_cache(slide=True)
        diffs = []
        with torch.no_grad():
            for i in range(2 * T):
                y = pt(x[:, :, i:i + 1], c[:, :, i:i + 1], kv_cache=cache)
                lo = max(i + 1 - T, 0)
                ref = pt(x[:, :, lo:i + 1], c[:, :, lo:i + 1])[:, :, -1:]
                diffs.append((y - ref).abs().max().item())
        assert max(diffs) < EXACT_TOL, (mode, diffs)
        print(f"    {mode}: transformer, 1 layer, sliding vs recomputed window worst {max(diffs):.1e}")


def test_model_slide_one_layer_exact():
    for mode in ("rel_bias", "none"):
        m, cfg = build(mode, extra={"pint_dyn_layers": 1, "pint_ctx_layers": 1, "temporal_vgain": True})
        perturb(m)
        set_vgain(m)
        horizon, cond, steps = cfg["timestep_horizon"], 2, 12
        x = frames(cfg, cond + steps)
        ref, sl = rollout(m, x, cond, steps, False), rollout(m, x, cond, steps, "slide")
        past = [k for k in range(steps) if cond + k > horizon]
        d = max((ref[key][:, k - steps] - sl[key][:, k - steps]).abs().max().item() for key in LATENTS for k in past)
        assert d < EXACT_TOL, (mode, d)
        kv = rollout(m, x, cond, steps, True)
        d_kv = max((ref[key] - kv[key]).abs().max().item() for key in LATENTS)
        assert d_kv < EXACT_TOL, (mode, d_kv)
        print(f"    {mode}: model, 1 layer, slide vs recompute past the block size {d:.1e}; kv=True {d_kv:.1e}")


def main():
    try:
        import build_model  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: {exc}")
        return 0
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # report every test
            failed += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
