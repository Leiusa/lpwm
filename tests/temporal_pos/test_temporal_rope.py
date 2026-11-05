#!/usr/bin/env python
"""
Tests of the opt-in `temporal_rope` (learned rotary time position on q, k, v of the temporal attention of the
context/dynamics transformers; modules.TemporalRotary), used with temporal_pos 'rel_bias' / 'none'.

Asserted here (CPU, random weights):
  1. strict checks: temporal_pos 'abs_adaln', non-bool, non-causal transformer -> ValueError;
  2. off by default: no key == explicit False (same parameters, same outputs, bitwise); on adds only rope parameters
     (theta_qk, theta_v, carrier per temporal attention, all zero);
  3. zero init is the identity: a 'rel_bias' / 'none' model and the same weights with temporal_rope (rope params at 0)
     give the same training forward and the same rollouts (recompute, kv cache, sliding cache) up to fp32 rounding
     (bitwise with torch 2.6 on grogu; <= 1e-6 with torch 2.9.1 on macOS, multi-frame path only);
  4. the gradient of the loss reaches theta_qk, theta_v and carrier at zero init (fine-tuning can move them);
  5. with non-zero rope parameters: outputs change (the rotation is live), and depend only on time offsets (a kv
     cache whose positions start at 37 gives the same outputs as one starting at 0);
  6. kv cache == full forward within the block size; one layer: sliding cache == recomputed window past the block
     size (transformer and whole model), as for 'rel_bias' without rope.

    python tests/temporal_pos/test_temporal_rope.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from test_temporal_pos import build, frames, perturb  # noqa: E402

EXACT_TOL = 1e-5  # fp32 noise of cache vs no cache measured <= 1e-6 without rope
SENS = 1e-4
LATENTS = ("z_pos", "z_scale", "z_obj_on", "z_depth", "z_features", "z_bg_features", "z_context")
ROPE_KEYS = ("theta_qk", "theta_v", "carrier")


def rope_params(m):
    return {k: p for k, p in m.named_parameters() if ".rope." in k}


def set_rope(m, seed=3, theta=0.3, carrier=0.5):
    # non-zero rope parameters, large enough to matter over a few frames
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k, p in rope_params(m).items():
            p.copy_((carrier if k.endswith("carrier") else theta) * torch.randn(p.shape, generator=g))


def rollout(m, x, cond, steps, kv):
    with torch.no_grad():
        _, z = m.sample_from_x(x, num_steps=steps, deterministic=True, cond_steps=cond, return_z=True,
                               decode=False, kv_cache=kv)
    return z


def pair(mode, extra=None):
    # base model (temporal_pos = mode, perturbed) and the same weights with temporal_rope (rope params at 0)
    base, cfg = build(mode, extra=extra)
    perturb(base)
    rope, _ = build(mode, extra={**(extra or {}), "temporal_rope": True})
    missing, unexpected = rope.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert missing and all(".rope." in k and k.rsplit(".", 1)[1] in ROPE_KEYS for k in missing), missing
    assert all((p == 0).all() for p in rope_params(rope).values())
    return base, rope, cfg


def train_out(m, x, seed=0):
    m.train()
    torch.manual_seed(seed)
    out = m(x)
    m.eval()
    return out


def test_strict_checks():
    from modules.modules import ParticleSpatioTemporalTransformer
    bad = [
        lambda: build("abs_adaln", extra={"temporal_rope": True}),
        lambda: build("rel_bias", extra={"temporal_rope": 1}),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, pos_embed_t_adaln=True, temporal_rope=True),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, temporal_pos="rel_bias", causal=False,
                                                  temporal_rope=True),
    ]
    for i, f in enumerate(bad):
        try:
            f()
        except ValueError:
            continue
        raise AssertionError(f"case {i} did not raise ValueError")


def test_default_off():
    a, cfg = build("rel_bias")
    b, _ = build("rel_bias", extra={"temporal_rope": False})
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not rope_params(a)
    x = frames(cfg, 4)
    za, zb = rollout(a, x, 2, 2, False), rollout(b, x, 2, 2, False)
    assert all(torch.equal(za[k], zb[k]) for k in LATENTS)
    r, _ = build("rel_bias", extra={"temporal_rope": True})
    extra = set(r.state_dict()) - set(sa)
    assert extra and all(".rope." in k for k in extra) and set(sa) <= set(r.state_dict())
    # unique parameters (the context module is registered under several names in the state dict)
    n_attn = sum(1 for k in rope_params(r) if k.endswith("theta_qk"))
    n_blocks = cfg["pint_dyn_layers"] + cfg["pint_ctx_layers"]
    assert n_attn == n_blocks, (n_attn, n_blocks)  # one per temporal attention, none in the spatial attention


def max_diff(a, b, keys):
    return max((a[k].float() - b[k].float()).abs().max().item() for k in keys)


def test_zero_init_is_identity():
    for mode in ("rel_bias", "none"):
        base, rope, cfg = pair(mode)
        x = frames(cfg, cfg["timestep_horizon"] + 1)
        oa, ob = train_out(base, x), train_out(rope, x)
        keys = [k for k, v in oa.items() if torch.is_tensor(v) and v.dtype.is_floating_point]
        assert keys
        d = max_diff(oa, ob, keys)
        assert d < EXACT_TOL, (mode, "training forward", d)
        diffs = {"train": d}
        horizon, cond, steps = cfg["timestep_horizon"], 2, 10  # past the block size
        x = frames(cfg, cond + steps)
        for kv in (False, True, "slide"):
            za, zb = rollout(base, x, cond, steps, kv), rollout(rope, x, cond, steps, kv)
            diffs[str(kv)] = max_diff(za, zb, LATENTS)
            assert diffs[str(kv)] < EXACT_TOL, (mode, kv, diffs[str(kv)])
        print(f"    {mode}: zero-init rope vs base, max diff " + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()))


def test_gradient_reaches_rope_at_zero_init():
    _, rope, cfg = pair("rel_bias")
    rope.train()
    torch.manual_seed(0)
    out = rope(frames(cfg, cfg["timestep_horizon"] + 1))
    loss = sum(v.float().mean() for v in out.values() if torch.is_tensor(v) and v.dtype.is_floating_point
               and v.requires_grad and v.dim() > 0)
    loss.backward()
    for k, p in rope_params(rope).items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), k
        assert p.grad.abs().max() > 0, k


def test_rope_is_live_and_offset_only():
    from modules.modules import ParticleSpatioTemporalTransformer
    b, n, T, f = 2, 3, 6, 32
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, n, T, f, generator=g)
    c = torch.randn(b, n, T, f, generator=g)
    torch.manual_seed(0)
    pt = ParticleSpatioTemporalTransformer(f, 4, 2, T, f, context_cond=True, temporal_pos="rel_bias",
                                           temporal_rope=True).eval()
    perturb(pt)
    with torch.no_grad():
        for p in rope_params(pt).values():
            p.zero_()
        y0 = pt(x, c)
        set_rope(pt)
        y1 = pt(x, c)
        assert (y1 - y0).abs().max() > SENS, "non-zero rope parameters did not change the output"
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
                                               temporal_rope=True).eval()
        perturb(pt)
        set_rope(pt)
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
        m, cfg = build(mode, extra={"pint_dyn_layers": 1, "pint_ctx_layers": 1, "temporal_rope": True})
        perturb(m)
        set_rope(m)
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
