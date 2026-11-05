#!/usr/bin/env python
"""
Tests of the opt-in `temporal_window` (banded causal temporal attention in the context/dynamics transformers).

With temporal_window = W, each timestep attends to itself and the W - 1 timesteps before it, in every layer. Trained
on clips longer than W, the model learns to use history beyond the window (it reaches deeper layers through the
earlier frames), and at inference a sliding kv cache of W timesteps computes exactly what training computed.

Asserted here (CPU, random weights):
  1. the mask is banded: query i sees keys j with i - W < j <= i;
  2. strict checks: temporal_pos 'abs_adaln', window > horizon, non-int window -> ValueError;
  3. transformer level: one forward over a long sequence == the same sequence fed one timestep at a time through a
     sliding cache of W timesteps (the inference path computes the training computation);
  4. model level: sample(kv_cache='slide') == the same unroll with caches that keep every timestep under the banded
     mask (exact banded computation), past the window;
  5. a training forward/backward with clips longer than the window runs and gives finite losses.

    python tests/temporal_pos/test_temporal_window.py
"""

import os
import sys
from contextlib import contextmanager

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from test_temporal_pos import build, frames, perturb  # noqa: E402

EXACT_TOL = 1e-5  # fp32 noise of cache vs no cache measured <= 1e-6
LATENTS = ("z_pos", "z_scale", "z_obj_on", "z_depth", "z_features", "z_bg_features", "z_context")
HORIZON, WINDOW = 12, 4


def test_mask_banded():
    from modules.modules import CausalParticleSelfAttention
    a = CausalParticleSelfAttention(16, 2, 8, temporal_window=3)
    m = a.mask[0, 0, 0, :, 0, :]
    for i in range(8):
        for j in range(8):
            assert m[i, j].item() == float(i - 3 < j <= i), (i, j)
    full = CausalParticleSelfAttention(16, 2, 8).mask[0, 0, 0, :, 0, :]
    assert torch.equal(full, torch.tril(torch.ones(8, 8)))


def test_strict_checks():
    from modules.modules import ParticleSpatioTemporalTransformer
    bad = [
        lambda: build("abs_adaln", extra={"temporal_window": 3}),
        lambda: build("rel_bias", extra={"temporal_window": 7}),  # horizon is 6
        lambda: build("rel_bias", extra={"temporal_window": 2.0}),
        lambda: build("rel_bias", extra={"temporal_window": True}),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, pos_embed_t_adaln=True, temporal_window=3),
        lambda: ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, temporal_pos="none", causal=False,
                                                  temporal_window=3),
    ]
    for i, f in enumerate(bad):
        try:
            f()
        except ValueError:
            continue
        raise AssertionError(f"case {i} did not raise ValueError")


def test_transformer_slide_is_training_computation():
    from modules.modules import ParticleSpatioTemporalTransformer
    b, n, f = 2, 3, 32
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, n, HORIZON, f, generator=g)
    c = torch.randn(b, n, HORIZON, f, generator=g)
    for mode in ("rel_bias", "none"):
        torch.manual_seed(0)
        pt = ParticleSpatioTemporalTransformer(f, 4, 3, HORIZON, f, context_cond=True, temporal_pos=mode,
                                               temporal_window=WINDOW).eval()
        perturb(pt)
        with torch.no_grad():
            full = pt(x, c)  # training-style forward over the whole sequence
            cache = pt.new_kv_cache(slide=True)
            assert cache["window"] == WINDOW
            steps = [pt(x[:, :, i:i + 1], c[:, :, i:i + 1], kv_cache=cache) for i in range(HORIZON)]
            assert all(kv["k"].shape[2] == WINDOW for kv in cache["layers"])
        d = (torch.cat(steps, dim=2) - full).abs().max().item()
        assert d < EXACT_TOL, (mode, d)
        print(f"    {mode}: 3 layers, {HORIZON} steps, window {WINDOW}: slide vs full forward {d:.1e}")


@contextmanager
def full_window(m):
    # make sample() treat the whole block as the window: kv_cache=True then keeps every timestep in its caches, and
    # the banded mask alone limits attention (the exact banded computation, as in training)
    pt = m.dyn_module.particle_transformer
    w = pt.temporal_window
    pt.temporal_window = None
    try:
        yield
    finally:
        pt.temporal_window = w


def test_model_slide_exact():
    for mode in ("rel_bias", "none"):
        m, cfg = build(mode, extra={"timestep_horizon": HORIZON, "temporal_window": WINDOW})
        perturb(m)
        cond, steps = 2, HORIZON - 2 - 1  # every input stays within the block, well past the window
        x = frames(cfg, cond + steps)

        def run(kv):
            with torch.no_grad():
                _, z = m.sample_from_x(x, num_steps=steps, deterministic=True, cond_steps=cond, return_z=True,
                                       decode=False, kv_cache=kv)
            return z

        with full_window(m):
            ref = run(True)
        sl = run("slide")
        d = max((ref[k] - sl[k]).abs().max().item() for k in LATENTS)
        assert d < EXACT_TOL, (mode, d)
        rec = run(False)  # recompute of the last W timesteps: drops the history the banded model was trained with
        d_rec = max((ref[k] - rec[k]).abs().max().item() for k in LATENTS)
        print(f"    {mode}: slide vs exact banded {d:.1e}; recompute-window vs exact banded {d_rec:.1e}")


def test_training_forward_longer_than_window():
    m, cfg = build("rel_bias", extra={"timestep_horizon": HORIZON, "temporal_window": WINDOW})
    m.train()
    x = frames(cfg, HORIZON + 1)
    torch.manual_seed(0)
    out = m(x)
    loss = sum(v.float().mean() for k, v in out.items() if torch.is_tensor(v) and v.dtype.is_floating_point
               and v.requires_grad and v.dim() > 0)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


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
