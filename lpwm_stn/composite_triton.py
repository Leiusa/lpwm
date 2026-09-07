"""
Standalone fused Triton forward for paste + alpha/depth composite + reduction.

Consumes the ORIGINAL RGBA patches, kp, scale, obj_on and z_depth directly and
never forms the [B, K, 4, H, W] pasted tensor: sampling and compositing happen
in the same kernel, and only the reduced [B,3,H,W] and [B,1,H,W] outputs (plus
the per-particle masks, when explicitly requested) are ever written.

Ordering is deliberately faithful rather than convenient. The reference divides
each particle's importance by the particle-sum BEFORE multiplying and summing:

    imp_norm_k = imp_k / (sum_j imp_j + eps)
    out_rgb    = sum_k (a_k * rgb_k) * imp_norm_k

That division could be factored out of the sum for a single pass over particles,
but that is a different order of operations. Instead the kernel walks the
particle axis twice -- once to accumulate the denominator, once to apply it per
particle -- paying a second round of sampling to keep the reference's arithmetic
structure. Nothing is materialized either way.

Coordinate math is taken verbatim from _stn_paste_forward_kernel so the sampling
matches the already-validated paste kernel exactly.
"""

import torch
import triton
import triton.language as tl

from .triton_backend import _base_axis

__all__ = ["composite_forward_triton", "can_use_triton_composite"]

EPS = 1e-5
_ST_EPS = 1e-9          # spatial_transform's own epsilon on the scale divisions


@triton.jit
def _sample_channel(patches_ptr, base, y0, x0, y1, x1, dx, dy,
                    valid_x0, valid_x1, valid_y0, valid_y1,
                    stride_ph, stride_pw, active, PATCH: tl.constexpr):
    v00 = tl.load(patches_ptr + base + y0 * stride_ph + x0 * stride_pw,
                  mask=active & valid_x0 & valid_y0, other=0.0).to(tl.float32)
    v01 = tl.load(patches_ptr + base + y0 * stride_ph + x1 * stride_pw,
                  mask=active & valid_x1 & valid_y0, other=0.0).to(tl.float32)
    v10 = tl.load(patches_ptr + base + y1 * stride_ph + x0 * stride_pw,
                  mask=active & valid_x0 & valid_y1, other=0.0).to(tl.float32)
    v11 = tl.load(patches_ptr + base + y1 * stride_ph + x1 * stride_pw,
                  mask=active & valid_x1 & valid_y1, other=0.0).to(tl.float32)
    value = v00 * (1.0 - dx) * (1.0 - dy)
    value += v01 * dx * (1.0 - dy)
    value += v10 * (1.0 - dx) * dy
    value += v11 * dx * dy
    return value


@triton.jit
def _composite_forward_kernel(
        patches_ptr, kp_ptr, scale_ptr, obj_on_ptr, depth_ptr, base_axis_ptr,
        out_rgb_ptr, out_alpha_ptr, out_masks_ptr,
        n_pixels,
        stride_pb, stride_pk, stride_pc, stride_ph, stride_pw,
        stride_kpb, stride_kpk, stride_kpd,
        stride_sb, stride_sk, stride_sd,
        stride_ob, stride_ok,
        stride_db, stride_dk,
        stride_rb, stride_rc, stride_rh, stride_rw,
        stride_ab, stride_ah, stride_aw,
        stride_mb, stride_mk, stride_mh, stride_mw,
        EPS_C, N_KP: tl.constexpr, PATCH: tl.constexpr, IMAGE: tl.constexpr,
        BLOCK: tl.constexpr, RETURN_MASKS: tl.constexpr):
    batch = tl.program_id(0)
    pix = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = pix < n_pixels

    canvas_x = pix % IMAGE
    canvas_y = (pix // IMAGE) % IMAGE
    base_x = tl.load(base_axis_ptr + canvas_x, mask=active, other=0.0)
    base_y = tl.load(base_axis_ptr + canvas_y, mask=active, other=0.0)

    # ---- pass 1: denominator, sum_k imp_k ----
    denom = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(N_KP):
        pos_y = tl.load(kp_ptr + batch * stride_kpb + k * stride_kpk).to(tl.float32)
        pos_x = tl.load(kp_ptr + batch * stride_kpb + k * stride_kpk + stride_kpd).to(tl.float32)
        sc_y = tl.load(scale_ptr + batch * stride_sb + k * stride_sk).to(tl.float32)
        sc_x = tl.load(scale_ptr + batch * stride_sb + k * stride_sk + stride_sd).to(tl.float32)
        on = tl.load(obj_on_ptr + batch * stride_ob + k * stride_ok).to(tl.float32)
        dep = tl.load(depth_ptr + batch * stride_db + k * stride_dk).to(tl.float32)

        den_y = sc_y + 1.0e-9
        den_x = sc_x + 1.0e-9
        norm_x = base_x * (1.0 / den_x) + (-pos_x / den_x)
        norm_y = base_y * (1.0 / den_y) + (-pos_y / den_y)
        px = ((norm_x + 1.0) * PATCH - 1.0) * 0.5
        py = ((norm_y + 1.0) * PATCH - 1.0) * 0.5
        x0f = tl.floor(px); y0f = tl.floor(py)
        x0 = x0f.to(tl.int32); y0 = y0f.to(tl.int32)
        x1 = x0 + 1; y1 = y0 + 1
        dx = px - x0f; dy = py - y0f
        vx0 = (x0 >= 0) & (x0 < PATCH); vx1 = (x1 >= 0) & (x1 < PATCH)
        vy0 = (y0 >= 0) & (y0 < PATCH); vy1 = (y1 >= 0) & (y1 < PATCH)
        abase = batch * stride_pb + k * stride_pk + 0 * stride_pc
        alpha = _sample_channel(patches_ptr, abase, y0, x0, y1, x1, dx, dy,
                                vx0, vx1, vy0, vy1, stride_ph, stride_pw, active, PATCH)
        a_obj = on * alpha
        imp = a_obj * tl.sigmoid(-dep)
        denom += imp

    denom = denom + EPS_C

    # ---- pass 2: apply the normalized weight per particle ----
    acc_r = tl.zeros([BLOCK], dtype=tl.float32)
    acc_g = tl.zeros([BLOCK], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK], dtype=tl.float32)
    acc_a = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(N_KP):
        pos_y = tl.load(kp_ptr + batch * stride_kpb + k * stride_kpk).to(tl.float32)
        pos_x = tl.load(kp_ptr + batch * stride_kpb + k * stride_kpk + stride_kpd).to(tl.float32)
        sc_y = tl.load(scale_ptr + batch * stride_sb + k * stride_sk).to(tl.float32)
        sc_x = tl.load(scale_ptr + batch * stride_sb + k * stride_sk + stride_sd).to(tl.float32)
        on = tl.load(obj_on_ptr + batch * stride_ob + k * stride_ok).to(tl.float32)
        dep = tl.load(depth_ptr + batch * stride_db + k * stride_dk).to(tl.float32)

        den_y = sc_y + 1.0e-9
        den_x = sc_x + 1.0e-9
        norm_x = base_x * (1.0 / den_x) + (-pos_x / den_x)
        norm_y = base_y * (1.0 / den_y) + (-pos_y / den_y)
        px = ((norm_x + 1.0) * PATCH - 1.0) * 0.5
        py = ((norm_y + 1.0) * PATCH - 1.0) * 0.5
        x0f = tl.floor(px); y0f = tl.floor(py)
        x0 = x0f.to(tl.int32); y0 = y0f.to(tl.int32)
        x1 = x0 + 1; y1 = y0 + 1
        dx = px - x0f; dy = py - y0f
        vx0 = (x0 >= 0) & (x0 < PATCH); vx1 = (x1 >= 0) & (x1 < PATCH)
        vy0 = (y0 >= 0) & (y0 < PATCH); vy1 = (y1 >= 0) & (y1 < PATCH)
        pbase = batch * stride_pb + k * stride_pk
        alpha = _sample_channel(patches_ptr, pbase + 0 * stride_pc, y0, x0, y1, x1, dx, dy,
                                vx0, vx1, vy0, vy1, stride_ph, stride_pw, active, PATCH)
        r = _sample_channel(patches_ptr, pbase + 1 * stride_pc, y0, x0, y1, x1, dx, dy,
                            vx0, vx1, vy0, vy1, stride_ph, stride_pw, active, PATCH)
        g = _sample_channel(patches_ptr, pbase + 2 * stride_pc, y0, x0, y1, x1, dx, dy,
                            vx0, vx1, vy0, vy1, stride_ph, stride_pw, active, PATCH)
        b = _sample_channel(patches_ptr, pbase + 3 * stride_pc, y0, x0, y1, x1, dx, dy,
                            vx0, vx1, vy0, vy1, stride_ph, stride_pw, active, PATCH)
        a_obj = on * alpha
        imp = a_obj * tl.sigmoid(-dep)
        w = imp / denom
        acc_r += (a_obj * r) * w
        acc_g += (a_obj * g) * w
        acc_b += (a_obj * b) * w
        acc_a += w * a_obj
        if RETURN_MASKS:
            tl.store(out_masks_ptr + batch * stride_mb + k * stride_mk
                     + canvas_y * stride_mh + canvas_x * stride_mw,
                     w * a_obj, mask=active)

    tl.store(out_rgb_ptr + batch * stride_rb + 0 * stride_rc
             + canvas_y * stride_rh + canvas_x * stride_rw, acc_r, mask=active)
    tl.store(out_rgb_ptr + batch * stride_rb + 1 * stride_rc
             + canvas_y * stride_rh + canvas_x * stride_rw, acc_g, mask=active)
    tl.store(out_rgb_ptr + batch * stride_rb + 2 * stride_rc
             + canvas_y * stride_rh + canvas_x * stride_rw, acc_b, mask=active)
    tl.store(out_alpha_ptr + batch * stride_ab
             + canvas_y * stride_ah + canvas_x * stride_aw, 1.0 - acc_a, mask=active)


def can_use_triton_composite(dec_objects, z_kp, z_scale, obj_on, z_depth, img_size):
    """Conservative gate: fp32 CUDA, contiguous, RGBA, and shapes the kernel assumes."""
    tensors = [t for t in (dec_objects, z_kp, obj_on, z_depth, z_scale) if t is not None]
    if not all(t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        return False
    if dec_objects.dim() != 5 or dec_objects.shape[2] != 4:
        return False
    if dec_objects.shape[-1] != dec_objects.shape[-2]:
        return False
    return int(img_size) > 0


def composite_forward_triton(dec_objects, z_kp, z_scale, obj_on, z_depth, img_size,
                             eps=EPS, return_alpha_masks=True, scale_normalized=False,
                             block=256):
    """Fused forward. Returns (alpha_masks | None, alpha_mask, dec_objects_trans).

    `return_alpha_masks` is a compile-time constant (tl.constexpr): when False the
    store is compiled out of the kernel entirely and no [B,K,1,H,W] tensor is
    allocated -- not merely discarded afterwards.
    """
    bs, n_kp, ch, patch, _ = dec_objects.shape
    img = int(img_size)
    if z_scale is None:
        normalized_scale = (patch / img) * torch.ones_like(z_kp)
    elif scale_normalized:
        normalized_scale = z_scale
    else:
        normalized_scale = torch.sigmoid(z_scale)
    normalized_scale = normalized_scale.contiguous()

    out_rgb = torch.empty(bs, ch - 1, img, img, device=dec_objects.device, dtype=torch.float32)
    out_alpha = torch.empty(bs, 1, img, img, device=dec_objects.device, dtype=torch.float32)
    if return_alpha_masks:
        out_masks = torch.empty(bs, n_kp, 1, img, img, device=dec_objects.device,
                                dtype=torch.float32)
        m_strides = (out_masks.stride(0), out_masks.stride(1), out_masks.stride(3),
                     out_masks.stride(4))
    else:
        out_masks = None
        m_strides = (0, 0, 0, 0)

    n_pixels = img * img
    grid = (bs, triton.cdiv(n_pixels, block))
    _composite_forward_kernel[grid](
        dec_objects, z_kp, normalized_scale, obj_on, z_depth,
        _base_axis(dec_objects.device, img),
        out_rgb, out_alpha, out_masks if out_masks is not None else out_rgb,
        n_pixels,
        dec_objects.stride(0), dec_objects.stride(1), dec_objects.stride(2),
        dec_objects.stride(3), dec_objects.stride(4),
        z_kp.stride(0), z_kp.stride(1), z_kp.stride(2),
        normalized_scale.stride(0), normalized_scale.stride(1), normalized_scale.stride(2),
        obj_on.stride(0), obj_on.stride(1),
        z_depth.stride(0), z_depth.stride(1),
        out_rgb.stride(0), out_rgb.stride(1), out_rgb.stride(2), out_rgb.stride(3),
        out_alpha.stride(0), out_alpha.stride(2), out_alpha.stride(3),
        *m_strides,
        eps,
        N_KP=n_kp, PATCH=patch, IMAGE=img, BLOCK=block,
        RETURN_MASKS=bool(return_alpha_masks),
    )
    return out_masks, out_alpha, out_rgb
