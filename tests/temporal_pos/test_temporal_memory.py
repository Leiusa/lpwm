#!/usr/bin/env python
"""
Tests of the opt-in `temporal_memory` (gated delta-rule recurrent memory per particle slot next to the temporal
attention of the context/dynamics transformers; modules.TemporalMemory), used with temporal_pos 'rel_bias' / 'none'
and normally a short temporal_window.

Asserted here (CPU, random weights):
  1. strict checks: temporal_pos 'abs_adaln', non-bool, non-causal transformer, together with temporal_rope /
     temporal_vgain -> ValueError;
  2. off by default: no key == explicit False (same parameters and outputs, bitwise); on adds only memory parameters;
  3. zero gate is the identity, bitwise: a model with temporal_window and the same weights plus temporal_memory give
     the same training forward and the same rollouts (recompute, kv cache, sliding cache);
  4. the gradient reaches the memory gate at init;
  5. TemporalMemory.scan equals a float64 reference loop of the update rule;
  6. with a non-zero gate: outputs change; kv cache step by step == full forward within the block size;
  7. 1 and 3 layers, window w, sequence twice the block size: the sliding cache (state carried over every step) ==
     the full forward of the same weights with a block size covering the whole sequence (banded mask of width w).

    python tests/temporal_pos/test_temporal_memory.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from test_temporal_pos import build, frames, perturb  # noqa: E402

EXACT_TOL = 1e-5
SENS = 1e-4
WINDOW = 4
LATENTS = ("z_pos", "z_scale", "z_obj_on", "z_depth", "z_features", "z_bg_features", "z_context")


def mem_params(m):
    return {k: p for k, p in m.named_parameters() if ".memory." in k or k.startswith("memory.")}


def set_gate(m, value=0.7, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k, p in mem_params(m).items():
            if k.endswith("gate"):
                p.fill_(value)
            elif k.endswith("alpha.weight") or k.endswith("beta.weight"):
                p.copy_(0.05 * torch.randn(p.shape, generator=g))


def rollout(m, x, cond, steps, kv):
    with torch.no_grad():
        _, z = m.sample_from_x(x, num_steps=steps, deterministic=True, cond_steps=cond, return_z=True,
                               decode=False, kv_cache=kv)
    return z


def pair(mode):
    extra = {"temporal_window": WINDOW}
    base, cfg = build(mode, extra=extra)
    perturb(base)
    mm, _ = build(mode, extra={**extra, "temporal_memory": True})
    missing, unexpected = mm.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert missing and all(".memory." in k for k in missing), missing
    return base, mm, cfg


def train_out(m, x, seed=0):
    m.train()
    torch.manual_seed(seed)
    out = m(x)
    m.eval()
    return out


def test_strict_checks():
    from modules.modules import ParticleSpatioTemporalTransformer
    bad = [
        lambda: build("abs_adaln", extra={"temporal_memory": True}),
        lambda: build("rel_bias", extra={"temporal_memory": 1}),
        lambda: build("rel_bias", extra={"temporal_memory": True, "temporal_rope": True}),
        lambda: build("rel_bias", extra={"temporal_memory": True, "temporal_vgain": True}),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, pos_embed_t_adaln=True, temporal_memory=True),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, temporal_pos="rel_bias", causal=False,
                                                  temporal_memory=True),
    ]
    for i, f in enumerate(bad):
        try:
            f()
        except ValueError:
            continue
        raise AssertionError(f"case {i} did not raise ValueError")


def test_default_off():
    a, cfg = build("rel_bias")
    b, _ = build("rel_bias", extra={"temporal_memory": False})
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not mem_params(a)
    x = frames(cfg, 4)
    za, zb = rollout(a, x, 2, 2, False), rollout(b, x, 2, 2, False)
    assert all(torch.equal(za[k], zb[k]) for k in LATENTS)
    m, _ = build("rel_bias", extra={"temporal_memory": True})
    extra = set(m.state_dict()) - set(sa)
    assert extra and all(".memory." in k for k in extra) and set(sa) <= set(m.state_dict())
    n_gates = sum(1 for k in mem_params(m) if k.endswith("gate"))
    assert n_gates == cfg["pint_dyn_layers"] + cfg["pint_ctx_layers"], n_gates


def test_zero_gate_is_identity():
    for mode in ("rel_bias", "none"):
        base, mm, cfg = pair(mode)
        x = frames(cfg, cfg["timestep_horizon"] + 1)
        oa, ob = train_out(base, x), train_out(mm, x)
        keys = [k for k, v in oa.items() if torch.is_tensor(v)]
        assert keys and all(torch.equal(oa[k], ob[k]) for k in keys), mode
        cond, steps = 2, 10  # past the block size
        x = frames(cfg, cond + steps)
        for kv in (False, True, "slide"):
            za, zb = rollout(base, x, cond, steps, kv), rollout(mm, x, cond, steps, kv)
            for k in LATENTS:
                assert torch.equal(za[k], zb[k]), (mode, kv, k)


def test_gradient_reaches_gate():
    _, mm, cfg = pair("rel_bias")
    mm.train()
    torch.manual_seed(0)
    out = mm(frames(cfg, cfg["timestep_horizon"] + 1))
    loss = sum(v.float().mean() for v in out.values() if torch.is_tensor(v) and v.dtype.is_floating_point
               and v.requires_grad and v.dim() > 0)
    loss.backward()
    gates = {k: p for k, p in mem_params(mm).items() if k.endswith("gate")}
    assert gates
    for k, p in gates.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0, k


def test_scan_matches_reference():
    from modules.modules import TemporalMemory
    g = torch.Generator().manual_seed(2)
    B, nh, T, hs = 2, 3, 7, 4
    mem = TemporalMemory(nh * hs, nh)
    q, k, v = (torch.randn(B, nh, T, hs, generator=g) for _ in range(3))
    a = torch.rand(B, nh, T, hs, generator=g)
    b = torch.rand(B, nh, T, 1, generator=g)
    m, S = mem.scan(q, k, v, a, b)
    qn = torch.nn.functional.normalize(q.double(), dim=-1)
    kn = torch.nn.functional.normalize(k.double(), dim=-1)
    ref = torch.zeros(B, nh, T, hs, dtype=torch.double)
    for bi in range(B):
        for h in range(nh):
            St = torch.zeros(hs, hs, dtype=torch.double)
            for t in range(T):
                kt, vt, bt = kn[bi, h, t], v[bi, h, t].double(), b[bi, h, t, 0].double()
                St = torch.diag(a[bi, h, t].double()) @ St
                St = (torch.eye(hs, dtype=torch.double) - bt * torch.outer(kt, kt)) @ St + bt * torch.outer(kt, vt)
                ref[bi, h, t] = St.T @ qn[bi, h, t]
    d = (m.double() - ref).abs().max().item()
    assert d < 1e-5, d
    # carrying the state over two chunks == one pass
    m1, S1 = mem.scan(q[:, :, :3], k[:, :, :3], v[:, :, :3], a[:, :, :3], b[:, :, :3])
    m2, S2 = mem.scan(q[:, :, 3:], k[:, :, 3:], v[:, :, 3:], a[:, :, 3:], b[:, :, 3:], S1)
    d2 = (torch.cat([m1, m2], 2) - m).abs().max().item()
    assert d2 < 1e-6 and (S2 - S).abs().max() < 1e-6, d2


def test_live_and_cache_equals_full():
    from modules.modules import ParticleSpatioTemporalTransformer
    b, n, T, f = 2, 3, 6, 32
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, n, T, f, generator=g)
    c = torch.randn(b, n, T, f, generator=g)
    torch.manual_seed(0)
    pt = ParticleSpatioTemporalTransformer(f, 4, 2, T, f, context_cond=True, temporal_pos="rel_bias",
                                           temporal_window=3, temporal_memory=True).eval()
    perturb(pt)
    with torch.no_grad():
        y0 = pt(x, c)
        set_gate(pt)
        y1 = pt(x, c)
        assert (y1 - y0).abs().max() > SENS, "a non-zero gate did not change the output"
        cache = pt.new_kv_cache()
        yc = torch.cat([pt(x[:, :, i:i + 1], c[:, :, i:i + 1], kv_cache=cache) for i in range(T)], 2)
    d = (yc - y1).abs().max().item()
    assert d < EXACT_TOL, d
    print(f"    kv cache step by step vs full forward: {d:.1e}")


def test_slide_long_sequence():
    # with a banded window w the sliding cache is exact in every layer (the cached keys/values were computed with
    # the same banded attention as training), and the memory state is carried, so 1 and 3 layers are both exact
    from modules.modules import ParticleSpatioTemporalTransformer
    b, n, T, f, w = 2, 3, 5, 32, 3
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, n, 2 * T, f, generator=g)
    c = torch.randn(b, n, 2 * T, f, generator=g)
    for n_layer in (1, 3):
        torch.manual_seed(0)
        pt = ParticleSpatioTemporalTransformer(f, 4, n_layer, T, f, context_cond=True, temporal_pos="none",
                                               temporal_window=w, temporal_memory=True).eval()
        perturb(pt)
        set_gate(pt)
        long = ParticleSpatioTemporalTransformer(f, 4, n_layer, 2 * T, f, context_cond=True, temporal_pos="none",
                                                 temporal_window=w, temporal_memory=True).eval()
        sd = {k: v for k, v in pt.state_dict().items() if not k.endswith(".mask")}
        missing, unexpected = long.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.endswith(".mask") for k in missing), (missing, unexpected)
        with torch.no_grad():
            ref = long(x, c)
            cache = pt.new_kv_cache(slide=True)
            ys = torch.cat([pt(x[:, :, i:i + 1], c[:, :, i:i + 1], kv_cache=cache) for i in range(2 * T)], 2)
        d = (ys - ref).abs().max().item()
        assert d < EXACT_TOL, (n_layer, d)
        print(f"    {n_layer} layer(s), window {w}, {2 * T} steps through a block of {T}: "
              f"sliding cache vs full forward {d:.1e}")


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
