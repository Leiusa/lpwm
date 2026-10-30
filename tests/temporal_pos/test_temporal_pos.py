#!/usr/bin/env python
"""
Tests of the opt-in `temporal_pos` switch (time position of the LPWM context/dynamics transformers).

    'abs_adaln' (default): original per-timestep embedding added to the AdaLN condition (keys/values carry the index)
    'rel_bias':            no per-timestep embedding; learned relative time bias in the temporal attention logits
    'none':                no explicit time position (only the causal mask orders frames)

Asserted here (CPU, random weights):
  1. default: no key == explicit 'abs_adaln' (same parameters, same outputs, bitwise);
  2. parameters: 'rel_bias' / 'none' drop every per-timestep embedding; 'rel_bias' adds a time-only bias table to the
     temporal attention of every block (none in the spatial block, no particle bias); AdaLN c_proj layers stay;
  3. the relative bias depends only on the frame offset (Toeplitz in (query, key));
  4. first-layer temporal keys/values of a frame do not depend on its position in the window for 'rel_bias' / 'none',
     and do for 'abs_adaln' (the property a sliding kv cache needs; deeper layers also see history, by design);
  5. kv cache exact while the sequence fits in the block size, for all three modes;
  6. strict checks: unknown value, combination with positional_bias, image-only model -> ValueError;
     checkpoints do not cross-load between modes (strict load fails).

    python tests/temporal_pos/test_temporal_pos.py
"""

import json
import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "stn"))

# small dynamics model built from a repo config (CPU-sized)
SMALL = {"pint_dim": 64, "pint_dyn_layers": 2, "pint_ctx_layers": 2, "pint_dyn_heads": 4, "pint_ctx_heads": 4,
         "pint_enc_layers": 1, "pint_enc_heads": 4, "timestep_horizon": 6, "batch_size": 2}
KV_TOL = 1e-4  # fp32 CPU; the cache changes only matrix shapes (reference run measured ~1e-6)


def build(temporal_pos=None, seed=0, base="balls.json", extra=None):
    from build_model import build as _build
    with open(os.path.join(_ROOT, "configs", base)) as f:
        cfg = json.load(f)
    cfg.update(SMALL)
    cfg.update(extra or {})
    if temporal_pos is not None:
        cfg["temporal_pos"] = temporal_pos
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        torch.manual_seed(seed)
        model, cfg, _ = _build(path, device="cpu")
    return model.eval(), cfg


def transformers(model):
    return {"dyn": model.dyn_module.particle_transformer, "ctx": model.ctx_module.pte}


def perturb(model, seed=1):
    # several heads are zero-initialized (AdaLN-Zero gates, outputs): move every weight a little so all paths are live
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))


def frames(cfg, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = cfg["image_size"]
    return torch.rand(2, n, cfg.get("ch", 3), s, s, generator=g)


def test_default_is_abs_adaln():
    a, cfg = build(None)
    b, _ = build("abs_adaln")
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert a.temporal_pos == "abs_adaln"
    for name, pt in transformers(a).items():
        assert pt.temporal_pos is None and pt.pos_embed_t_adaln, name
        assert hasattr(pt, "pos_embed_t_embedding"), name
        assert not any("rel_pos_bias" in k for k in pt.state_dict()), name
    perturb(a), perturb(b)
    x = frames(cfg, 4)
    with torch.no_grad():  # the forward samples latents: same seed for both
        torch.manual_seed(5)
        oa = a(x)
        torch.manual_seed(5)
        ob = b(x)
    for k in oa:
        if torch.is_tensor(oa[k]):
            assert torch.equal(oa[k], ob[k]), k


def test_parameters():
    for mode in ("rel_bias", "none"):
        m, _ = build(mode)
        for name, pt in transformers(m).items():
            keys = list(pt.state_dict())
            assert pt.temporal_pos == mode and not pt.pos_embed_t_adaln, (mode, name)
            assert not hasattr(pt, "pos_embed_t_embedding"), (mode, name)
            assert not isinstance(pt.pos_emb, torch.nn.Parameter), (mode, name)
            assert not any(k.startswith("pos_emb") for k in keys), (mode, name)
            assert not any("particle_rel_pos_bias" in k for k in keys), (mode, name)
            assert not any("spatio_block" in k and "rel_pos_bias" in k for k in keys), (mode, name)
            for i, blk in enumerate(pt.blocks):
                assert blk.temp_block.attn.positional_bias == (mode == "rel_bias"), (mode, name, i)
                assert not blk.spatio_block.attn.positional_bias, (mode, name, i)
                if mode == "rel_bias":
                    assert blk.temp_block.attn.max_particles is None, (mode, name, i)
                    tbl = blk.temp_block.attn.rel_pos_bias.rel_pos_bias
                    assert tbl.shape == (2 * pt.block_size - 1, pt.n_head), (mode, name, i)
                # AdaLN stays: the condition still carries context/particle identity
                assert blk.temp_block.c_proj is not None and blk.spatio_block.c_proj is not None, (mode, name, i)
            assert pt.head.c_proj is not None, (mode, name)


def test_bias_depends_on_offset_only():
    from modules.modules import SimpleRelativePositionalBias
    torch.manual_seed(0)
    rb = SimpleRelativePositionalBias(8, num_heads=3)
    for T in (1, 5, 8):
        b, _ = rb(T)
        b = b[0]  # [heads, T, T]
        tbl = rb.rel_pos_bias  # [2 * 8 - 1, heads]
        for i in range(T):
            for j in range(T):
                assert torch.equal(b[:, i, j], tbl[8 - 1 + (j - i)]), (T, i, j)


def first_layer_kv(pt, x, c):
    out = {}
    attn = pt.blocks[0].temp_block.attn
    hk = attn.key.register_forward_hook(lambda m, i, o: out.__setitem__("k", o))
    hv = attn.value.register_forward_hook(lambda m, i, o: out.__setitem__("v", o))
    try:
        with torch.no_grad():
            pt(x, c)
    finally:
        hk.remove(), hv.remove()
    return out["k"], out["v"]  # [b * n, 1, T, f]


def test_first_layer_kv_position_free():
    from modules.modules import ParticleSpatioTemporalTransformer
    b, n, T, f = 2, 3, 6, 32
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, n, T + 1, f, generator=g)
    c = torch.randn(b, n, T + 1, f, generator=g)
    for mode, flags in (("abs_adaln", dict(pos_embed_t_adaln=True)),
                        ("rel_bias", dict(temporal_pos="rel_bias")),
                        ("none", dict(temporal_pos="none"))):
        torch.manual_seed(0)
        pt = ParticleSpatioTemporalTransformer(f, 4, 2, T, f, context_cond=True, **flags).eval()
        perturb(pt)
        # frames 1..T-1 sit at window positions 1..T-1 in A and 0..T-2 in B
        ka, va = first_layer_kv(pt, x[:, :, :T], c[:, :, :T])
        kb, vb = first_layer_kv(pt, x[:, :, 1:], c[:, :, 1:])
        dk = (ka[:, :, 1:] - kb[:, :, :-1]).abs().max().item()
        dv = (va[:, :, 1:] - vb[:, :, :-1]).abs().max().item()
        if mode == "abs_adaln":
            assert dk > 1e-3 and dv > 1e-3, (mode, dk, dv)
        else:
            assert dk < 1e-5 and dv < 1e-5, (mode, dk, dv)


def test_kv_cache_exact_growing_phase():
    for mode in ("abs_adaln", "rel_bias", "none"):
        m, cfg = build(mode)
        perturb(m)
        horizon, cond, steps = cfg["timestep_horizon"], 2, 6
        x = frames(cfg, cond + steps)

        def run(kv):
            with torch.no_grad():
                _, z = m.sample_from_x(x, num_steps=steps, deterministic=True, cond_steps=cond, return_z=True,
                                       decode=False, kv_cache=kv)
            return z

        ref, kvz = run(False), run(True)
        worst = 0.0
        for k in range(steps):
            if cond + k > horizon:
                break
            for key in ("z_pos", "z_scale", "z_obj_on", "z_depth", "z_features", "z_bg_features", "z_context"):
                worst = max(worst, (ref[key][:, k - steps] - kvz[key][:, k - steps]).abs().max().item())
        assert worst < KV_TOL, (mode, worst)
        print(f"    kv cache {mode}: worst growing-phase diff {worst:.2e}")


def test_strict_checks():
    from modules.modules import ParticleSpatioTemporalTransformer, SpatioTemporalBlock
    bad = [
        lambda: build("relative"),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, temporal_pos="rel_bias", positional_bias=True),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, temporal_pos="none", pos_embed_t_adaln=True),
        lambda: SpatioTemporalBlock(32, 4, 6, positional_bias=True, temporal_bias=True),
        lambda: build("rel_bias", base="shapes.json", extra={"timestep_horizon": 1}),
    ]
    for i, f in enumerate(bad):
        try:
            f()
        except ValueError:
            continue
        raise AssertionError(f"case {i} did not raise ValueError")
    abs_m, _ = build("abs_adaln")
    for mode in ("rel_bias", "none"):
        m, _ = build(mode)
        for src, dst in ((abs_m, m), (m, abs_m)):
            try:
                dst.load_state_dict(src.state_dict())
            except RuntimeError:
                continue
            raise AssertionError(f"{mode}: checkpoint cross-loaded without error")


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
