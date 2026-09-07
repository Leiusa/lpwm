"""
Shared case list for the paste + alpha/depth composite.

One definition of "what to run", used by the fixture generator, the tests, and
later by any fused-kernel comparison. Inputs are generated deterministically
from a seed so the oracle, the fixtures and a kernel all see identical bytes.
"""

import torch

__all__ = ["CASES", "make_inputs", "case_names"]

#: (name, kwargs). Shapes cover the two real configs plus the edge cases a fused
#: reduction is most likely to get wrong.
CASES = (
    ("bair_bs1",        dict(bs=1, n_kp=90, p=8,  img=128)),
    ("balls_bs2",       dict(bs=2, n_kp=12, p=16, img=64)),
    ("small_noscale",   dict(bs=2, n_kp=8,  p=8,  img=32, with_scale=False)),
    ("small_nomasks",   dict(bs=2, n_kp=8,  p=8,  img=32, return_alpha_masks=False)),
    ("k1_degenerate",   dict(bs=2, n_kp=1,  p=8,  img=32)),
    ("many_particles",  dict(bs=1, n_kp=256, p=8, img=64)),
    # obj_on fully off: every importance_map term is 0, so the +eps denominator
    # is all that stands between the division and 0/0
    ("obj_on_all_zero", dict(bs=2, n_kp=8,  p=8,  img=32, obj_on_mode="zeros")),
    ("obj_on_all_one",  dict(bs=2, n_kp=8,  p=8,  img=32, obj_on_mode="ones")),
    # particles stacked at one point: maximal overlap in the reduction
    ("overlapping_kp",  dict(bs=2, n_kp=8,  p=8,  img=32, kp_mode="identical")),
    # off-canvas: paste contributes nothing, alpha stays 0
    ("offscreen_kp",    dict(bs=2, n_kp=8,  p=8,  img=32, kp_mode="offscreen")),
    # wide depth spread saturates sigmoid(-z_depth)
    ("depth_saturated", dict(bs=2, n_kp=8,  p=8,  img=32, depth_scale=40.0)),
    ("near_zero_alpha", dict(bs=2, n_kp=8,  p=8,  img=32, alpha_scale=1e-6)),
)


def case_names():
    return [n for n, _ in CASES]


def make_inputs(bs, n_kp, p, img, ch=4, seed=0, device="cpu", dtype=torch.float32,
                with_scale=True, return_alpha_masks=True, obj_on_mode="random",
                kp_mode="random", depth_scale=2.0, alpha_scale=1.0, requires_grad=False):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    dec_objects = torch.rand(bs, n_kp, ch, p, p, generator=g, device=device, dtype=dtype)
    if alpha_scale != 1.0:
        # scale only the alpha channel, leaving RGB alone
        dec_objects[:, :, :1] = dec_objects[:, :, :1] * alpha_scale

    if kp_mode == "identical":
        one = 2 * torch.rand(bs, 1, 2, generator=g, device=device, dtype=dtype) - 1
        z_kp = one.expand(bs, n_kp, 2).contiguous()
    elif kp_mode == "offscreen":
        z_kp = 3.0 + torch.rand(bs, n_kp, 2, generator=g, device=device, dtype=dtype)
    else:
        z_kp = 2 * torch.rand(bs, n_kp, 2, generator=g, device=device, dtype=dtype) - 1

    z_scale = (4 * torch.rand(bs, n_kp, 2, generator=g, device=device, dtype=dtype) - 2
               ) if with_scale else None

    if obj_on_mode == "zeros":
        obj_on = torch.zeros(bs, n_kp, device=device, dtype=dtype)
    elif obj_on_mode == "ones":
        obj_on = torch.ones(bs, n_kp, device=device, dtype=dtype)
    else:
        obj_on = torch.rand(bs, n_kp, generator=g, device=device, dtype=dtype)

    z_depth = depth_scale * (2 * torch.rand(bs, n_kp, 1, generator=g, device=device, dtype=dtype) - 1)

    if requires_grad:
        dec_objects.requires_grad_(True)
        z_kp.requires_grad_(True)
        obj_on.requires_grad_(True)
        z_depth.requires_grad_(True)
        if z_scale is not None:
            z_scale.requires_grad_(True)

    return {"dec_objects": dec_objects, "z_kp": z_kp, "z_scale": z_scale, "obj_on": obj_on,
            "z_depth": z_depth, "img_size": img, "return_alpha_masks": return_alpha_masks}


#: inputs whose gradients are part of the contract
GRAD_WRT = ("dec_objects", "z_kp", "z_scale", "obj_on", "z_depth")
