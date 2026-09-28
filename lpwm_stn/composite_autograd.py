"""
Autograd wrapper for the standalone fused composite (paste + alpha/depth composite + reduction).

It only connects the existing kernels to torch.autograd; it does not change their math:

    forward  -> composite_forward_triton          (lpwm_stn/composite_triton.py)
    backward -> composite_backward_triton         (lpwm_stn/composite_backward.py)

What is saved for backward is the five INPUTS (patches, z_kp, z_scale, obj_on, z_depth) and
nothing else: no pasted [B, K, 4, H, W] stack and none of its derived per-particle planes.
The backward recomputes them tile by tile inside the kernels.

Behaviour worth knowing:

* Outputs that receive no gradient are passed to the backward kernel as None (materialization
  of zero gradients is disabled), so their terms are compiled out rather than computed on zeros.
* Gradients are returned only for inputs that require them; the kernel always computes all five.
* ``dec_objects`` gradients are accumulated with atomics and are therefore not bit-reproducible
  run to run (the same holds for torch's own grid_sampler backward). The other four are.
* Double backward (create_graph=True) is refused with a RuntimeError. @once_differentiable would not do:
  it only raises when the incoming gradients require grad, so it would silently drop the second-order
  terms here.
* There is deliberately no silent fallback here. ``fused_composite_supported`` reports why an
  input is unsupported and ``composite_fused`` raises on it; the caller decides what to do.
"""

import torch

from .composite_reference import EPS

__all__ = ["composite_fused", "fused_composite_supported"]


def fused_composite_supported(dec_objects, z_kp, z_scale, obj_on, z_depth, img_size):
    """Return ``(ok, reason)``; ``reason`` is empty when ``ok``.

    Checks the layout the kernels assume: fp32 contiguous CUDA tensors on one device,
    dec_objects [B, K, 4, p, p] with square patches, z_kp [B, K, 2], z_scale None or
    [B, K, 2], obj_on [B, K], z_depth [B, K, 1].
    """
    named = {"dec_objects": dec_objects, "z_kp": z_kp, "z_scale": z_scale, "obj_on": obj_on, "z_depth": z_depth}
    for name, t in named.items():
        if t is None and name == "z_scale":
            continue
        if not torch.is_tensor(t):
            return False, f"{name} is not a tensor"
    if not dec_objects.is_cuda:
        return False, "dec_objects is not on CUDA"
    for name, t in named.items():
        if t is None:
            continue
        if t.device != dec_objects.device:
            return False, f"{name} is on {t.device}, dec_objects on {dec_objects.device}"
        if t.dtype != torch.float32:
            return False, f"{name} has dtype {t.dtype}, expected torch.float32"
        if not t.is_contiguous():
            return False, f"{name} is not contiguous"
    if dec_objects.dim() != 5 or dec_objects.shape[2] != 4:
        return False, f"dec_objects must be [B, K, 4, p, p], got {tuple(dec_objects.shape)}"
    if dec_objects.shape[-1] != dec_objects.shape[-2]:
        return False, f"dec_objects patches must be square, got {tuple(dec_objects.shape[-2:])}"
    if dec_objects.numel() == 0:
        return False, "dec_objects is empty"
    b, k = dec_objects.shape[:2]
    expected = {"z_kp": (b, k, 2), "obj_on": (b, k), "z_depth": (b, k, 1)}
    if z_scale is not None:
        expected["z_scale"] = (b, k, 2)
    for name, shape in expected.items():
        if tuple(named[name].shape) != shape:
            return False, f"{name} has shape {tuple(named[name].shape)}, expected {shape}"
    if int(img_size) <= 0:
        return False, f"img_size must be positive, got {img_size}"
    try:
        from . import composite_backward, composite_triton  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not swallowed
        return False, f"Triton kernels are not importable: {exc}"
    return True, ""


class _FusedComposite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dec_objects, z_kp, z_scale, obj_on, z_depth, img_size, eps, return_alpha_masks,
                scale_normalized):
        from .composite_triton import composite_forward_triton
        masks, bg_mask, rgb = composite_forward_triton(
            dec_objects, z_kp, z_scale, obj_on, z_depth, img_size, eps=eps,
            return_alpha_masks=return_alpha_masks, scale_normalized=scale_normalized)
        ctx.set_materialize_grads(False)
        ctx.has_scale = z_scale is not None
        ctx.img_size, ctx.eps, ctx.scale_normalized = img_size, eps, scale_normalized
        ctx.save_for_backward(dec_objects, z_kp, obj_on, z_depth, *((z_scale,) if z_scale is not None else ()))
        return masks, bg_mask, rgb

    @staticmethod
    def backward(ctx, grad_masks, grad_bg_mask, grad_rgb):
        if torch.is_grad_enabled():
            raise RuntimeError("composite_fused does not support double backward (create_graph=True): "
                               "the fused backward kernel is not differentiable")
        from .composite_backward import composite_backward_triton
        need = ctx.needs_input_grad
        saved = ctx.saved_tensors
        dec_objects, z_kp, obj_on, z_depth = saved[:4]
        z_scale = saved[4] if ctx.has_scale else None
        g_patches, g_kp, g_scale, g_on, g_depth = composite_backward_triton(
            dec_objects, z_kp, z_scale, obj_on, z_depth, ctx.img_size, grad_masks, grad_bg_mask, grad_rgb,
            eps=ctx.eps, scale_normalized=ctx.scale_normalized)
        return (g_patches if need[0] else None, g_kp if need[1] else None,
                g_scale if (need[2] and z_scale is not None) else None,
                g_on if need[3] else None, g_depth if need[4] else None, None, None, None, None)


def composite_fused(dec_objects, z_kp, z_scale, obj_on, z_depth, img_size, eps=EPS,
                    return_alpha_masks=True, scale_normalized=False):
    """Fused paste + composite with autograd. Returns ``(alpha_masks | None, bg_mask, dec_objects_trans)``,
    the same triple as ``composite_reference``. Raises ValueError for inputs the kernels do not support."""
    ok, reason = fused_composite_supported(dec_objects, z_kp, z_scale, obj_on, z_depth, img_size)
    if not ok:
        raise ValueError(f"fused composite does not support these inputs: {reason}")
    return _FusedComposite.apply(dec_objects, z_kp, z_scale, obj_on, z_depth, int(img_size), float(eps),
                                 bool(return_alpha_masks), bool(scale_normalized))
