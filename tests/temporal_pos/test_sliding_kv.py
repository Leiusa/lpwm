#!/usr/bin/env python
"""
Tests of the opt-in sliding kv cache (DLPDynamics.sample(kv_cache='slide'), temporal_pos 'rel_bias' / 'none').

kv_cache=True recomputes the whole window once the sequence is longer than the block size. 'slide' instead drops
the oldest cached timestep and runs only the new one. With one layer this is the same computation (first-layer
keys/values of a frame do not depend on the other frames); with more layers it is an approximation, because the
kept keys/values were computed while frames that have since left the window were still visible.

Asserted here (CPU, random weights):
  1. strict checks: 'slide' with temporal_pos 'abs_adaln' and unknown kv_cache values -> ValueError;
  2. transformer level: the cache never holds more than block_size timesteps; with one layer every step matches a
     recomputed window (fp32 tolerance), with two layers it does not once frames have left the window;
  3. model level, growing phase: 'slide' == kv_cache=True bitwise (same computation while the sequence fits);
  4. model level, past the block size: one dynamics + one context layer -> 'slide' matches recompute (also checks
     that the context transformer, whose block size is horizon + 1, slides on the dynamics window);
     two layers -> it differs (the approximation is real and visible; size printed).

    python tests/temporal_pos/test_sliding_kv.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from test_temporal_pos import build, frames, perturb  # noqa: E402

# 'same computation' = within fp32 noise (measured <= 1e-6); the 2-layer cases must differ by more than that once
# frames have left the window (measured: transformer ~1e-3, whole model ~4e-5 with these small random weights)
EXACT_TOL = 1e-5
SENS = 1e-5
LATENTS = ("z_pos", "z_scale", "z_obj_on", "z_depth", "z_features", "z_bg_features", "z_context")


def rollout(m, x, cond, steps, kv):
    with torch.no_grad():
        _, z = m.sample_from_x(x, num_steps=steps, deterministic=True, cond_steps=cond, return_z=True,
                               decode=False, kv_cache=kv)
    return z


def max_diff(a, b, steps, ks):
    # largest difference over the predicted timesteps ks (0-based, counted from the first prediction)
    return max((a[key][:, k - steps] - b[key][:, k - steps]).abs().max().item() for key in LATENTS for k in ks)


def test_strict_checks():
    m, cfg = build("abs_adaln")
    x = frames(cfg, 4)
    for kv in ("slide", "sliding", 2):
        try:
            rollout(m, x, 2, 2, kv)
        except ValueError:
            continue
        raise AssertionError(f"kv_cache={kv!r} with abs_adaln did not raise ValueError")
    from modules.modules import ParticleSpatioTemporalTransformer
    pt = ParticleSpatioTemporalTransformer(32, 4, 1, 6, 32, pos_embed_t_adaln=True)
    try:
        pt.new_kv_cache(slide=True)
    except ValueError:
        return
    raise AssertionError("new_kv_cache(slide=True) with an absolute time embedding did not raise ValueError")


def transformer_steps(n_layer, mode):
    # feed 2 * block_size frames one at a time through a sliding cache and compare each output row with a
    # recomputed window of the last block_size frames
    from modules.modules import ParticleSpatioTemporalTransformer
    b, n, T, f = 2, 3, 5, 32
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, n, 2 * T, f, generator=g)
    c = torch.randn(b, n, 2 * T, f, generator=g)
    torch.manual_seed(0)
    pt = ParticleSpatioTemporalTransformer(f, 4, n_layer, T, f, context_cond=True, temporal_pos=mode).eval()
    perturb(pt)
    cache = pt.new_kv_cache(slide=True)
    diffs = []
    with torch.no_grad():
        for i in range(2 * T):
            y = pt(x[:, :, i:i + 1], c[:, :, i:i + 1], kv_cache=cache)
            lo = max(i + 1 - T, 0)
            ref = pt(x[:, :, lo:i + 1], c[:, :, lo:i + 1])[:, :, -1:]
            diffs.append((y - ref).abs().max().item())
            assert all(kv['k'].shape[2] == min(i + 1, T) for kv in cache['layers']), i
            assert cache['t'] == min(i + 1, T), i
    return T, diffs


def test_transformer_slide():
    for mode in ("rel_bias", "none"):
        T, d1 = transformer_steps(1, mode)
        assert max(d1) < EXACT_TOL, (mode, d1)
        T, d2 = transformer_steps(2, mode)
        assert max(d2[:T]) < EXACT_TOL, (mode, d2)  # nothing has left the window yet
        assert max(d2[T + 1:]) > SENS, (mode, d2)  # frames have left the window; deeper layers remember them
        print(f"    {mode}: 1 layer worst {max(d1):.1e}; 2 layers growing {max(d2[:T]):.1e}, "
              f"sliding {max(d2[T + 1:]):.1e}")


def test_growing_phase_same_as_kv():
    for mode in ("rel_bias", "none"):
        m, cfg = build(mode)
        perturb(m)
        horizon, cond = cfg["timestep_horizon"], 2
        steps = horizon - cond + 1  # the last step runs on exactly horizon frames
        x = frames(cfg, cond + steps)
        a, b = rollout(m, x, cond, steps, True), rollout(m, x, cond, steps, "slide")
        for key in LATENTS:
            assert torch.equal(a[key], b[key]), (mode, key)


def test_past_block_size():
    for mode in ("rel_bias", "none"):
        res = {}
        for layers in (1, 2):
            m, cfg = build(mode, extra={"pint_dyn_layers": layers, "pint_ctx_layers": layers})
            perturb(m)
            horizon, cond, steps = cfg["timestep_horizon"], 2, 12
            x = frames(cfg, cond + steps)
            ref, sl = rollout(m, x, cond, steps, False), rollout(m, x, cond, steps, "slide")
            past = [k for k in range(steps) if cond + k > horizon]  # steps whose input is longer than the window
            assert past, "test must run past the block size"
            res[layers] = max_diff(ref, sl, steps, past)
        assert res[1] < EXACT_TOL, (mode, res)
        assert res[2] > SENS, (mode, res)
        print(f"    {mode}: past block size, 1 layer {res[1]:.1e} (exact), 2 layers {res[2]:.1e} (approximation)")


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
