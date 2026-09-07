"""
Standalone fused Triton BACKWARD for the composite chain.

Derivation (per pixel; k indexes particles, c indexes RGB):

    a_k   = on_k * alpha_k              alpha_k, rgb_k are bilinear samples of patch k
    sig_k = sigmoid(-depth_k)
    imp_k = a_k * sig_k
    S     = sum_j imp_j + eps
    w_k   = imp_k / S
    out_c = sum_k (a_k * rgb_ck) * w_k
    am    = 1 - sum_k w_k * a_k
    mk_k  = w_k * a_k

With T_k = sum_c rgb_ck*G_rgb_c - G_am + G_mk_k :

    dL/dw_k    = a_k * T_k
    P          = sum_k a_k * T_k * w_k
    dL/dimp_k  = (a_k*T_k - P) / S          <- the only cross-particle coupling
    dL/da_k    = w_k*T_k + dL/dimp_k * sig_k
    dL/drgb_ck = a_k * w_k * G_rgb_c
    dL/ddep_k  = -dL/dimp_k * a_k * sig_k*(1-sig_k)

S and P are per-pixel scalars, so they are the only things that must cross the
particle axis. They are computed once into [B, H*W] buffers -- reduced, not a
[B,K,C,H,W] canvas -- and every per-particle quantity is then recomputed from
the original patch-sized inputs.

Two kernels:
  A  per (batch, pixel block): two passes over particles -> S and P.
  B  per (batch, particle):    one pass over pixel blocks. Reductions for kp,
     scale, obj_on and depth accumulate in-register within a single program, so
     they are deterministic -- no atomics. Only grad_patches uses atomic_add,
     where scatter is unavoidable and where the frozen policy already carries the
     grid_sampler atomicAdd exception.
"""

import torch
import triton
import triton.language as tl

from lpwm_stn.triton_backend import _base_axis

__all__ = ["composite_backward_triton"]

_ST_EPS = 1.0e-9



# --- Exact FP32 RGB three-term total -------------------------------------
# PyTorch's reference node is ((a*rgb) * G_rgb).sum(dim=2) over a strided
# [B,K,3,H,W] tensor, which reduces strictly left-to-right: (R+G)+B, with each
# product separately rounded.  Triton's own codegen contracts a*b + c*d into
# fma.rn.f32, which drops the intermediate rounding and diverges by 2 ULP.
# Emitting the PTX directly reproduces the reference rounding exactly.
_RGB_TOTAL_ASM: tl.constexpr = """{
    .reg .f32 %t0, %t1, %t2;
    mul.rn.f32 %t0, $1, $2;
    mul.rn.f32 %t0, %t0, $3;
    mul.rn.f32 %t1, $1, $4;
    mul.rn.f32 %t1, %t1, $5;
    mul.rn.f32 %t2, $1, $6;
    mul.rn.f32 %t2, %t2, $7;
    add.rn.f32 $0, %t0, %t1;
    add.rn.f32 $0, $0, %t2;
}"""


# --- Exact FP32 dL_dimp primitives ---------------------------------------
# tl.math.div_rn lowers to div.rn.FTZ.f32, which flushes denormals; torch does
# not.  Emitting div.rn.f32 directly keeps both the rounding and the denormal
# behaviour.  _sub_prod exists because `acc += -x*y` is contracted into
# fma.rn.f32, which drops the intermediate rounding of the product.
_DIV_RN_ASM: tl.constexpr = """{
    div.rn.f32 $0, $1, $2;
}"""


@triton.jit
def _div_rn(x, y):
    """x / y, IEEE round-to-nearest, no flush-to-zero."""
    return tl.inline_asm_elementwise(
        _DIV_RN_ASM, "=f,f,f", [x, y],
        dtype=tl.float32, is_pure=True, pack=1)


_SUB_PROD_ASM: tl.constexpr = """{
    .reg .f32 %t;
    mul.rn.f32 %t, $2, $3;
    sub.rn.f32 $0, $1, %t;
}"""


@triton.jit
def _sub_prod(acc, x, y):
    """acc - x*y, separately rounded, no contraction."""
    return tl.inline_asm_elementwise(
        _SUB_PROD_ASM, "=f,f,f,f", [acc, x, y],
        dtype=tl.float32, is_pure=True, pack=1)


# --- Exact FP32 masks + background contribution --------------------------
# Reference node is (a*G_mk) + ((-a)*G_am): two separately rounded products
# combined with one rounded subtract.  Triton contracts that accumulation into
# fma.rn.f32, and the leading `tl.zeros +` add additionally turns a -0.0 product
# into +0.0.  Assigning the completed node from explicit PTX reproduces the
# reference bit-for-bit, signed zeros included.
_MK_BG_ASM: tl.constexpr = """{
    .reg .f32 %t0, %t1;
    mul.rn.f32 %t0, $1, $2;
    mul.rn.f32 %t1, $1, $3;
    sub.rn.f32 $0, %t0, %t1;
}"""


@triton.jit
def _mk_bg_total(a, g_m, g_a):
    """(a*g_m) - (a*g_a), separately rounded, no contraction."""
    return tl.inline_asm_elementwise(
        _MK_BG_ASM, "=f,f,f,f", [a, g_m, g_a],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rgb_total(a, r, g, bl, g_r, g_g, g_b):
    """(a*r)*g_r + (a*g)*g_g + (a*bl)*g_b, left-to-right, no contraction."""
    return tl.inline_asm_elementwise(
        _RGB_TOTAL_ASM, "=f,f,f,f,f,f,f,f",
        [a, r, g_r, g, g_g, bl, g_b],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _bilinear(ptr, base, y0, x0, y1, x1, dx, dy, vx0, vx1, vy0, vy1,
              sh, sw, active, PATCH: tl.constexpr):
    v00 = tl.load(ptr + base + y0 * sh + x0 * sw, mask=active & vx0 & vy0, other=0.0).to(tl.float32)
    v01 = tl.load(ptr + base + y0 * sh + x1 * sw, mask=active & vx1 & vy0, other=0.0).to(tl.float32)
    v10 = tl.load(ptr + base + y1 * sh + x0 * sw, mask=active & vx0 & vy1, other=0.0).to(tl.float32)
    v11 = tl.load(ptr + base + y1 * sh + x1 * sw, mask=active & vx1 & vy1, other=0.0).to(tl.float32)
    val = v00 * (1.0 - dx) * (1.0 - dy) + v01 * dx * (1.0 - dy) \
        + v10 * (1.0 - dx) * dy + v11 * dx * dy
    ddx = -v00 * (1.0 - dy) + v01 * (1.0 - dy) - v10 * dy + v11 * dy
    ddy = -v00 * (1.0 - dx) - v01 * dx + v10 * (1.0 - dx) + v11 * dx
    return val, ddx, ddy


@triton.jit
def _coords(kp_ptr, sc_ptr, b, k, s_kpb, s_kpk, s_kpd, s_sb, s_sk, s_sd,
            base_x, base_y, PATCH: tl.constexpr):
    pos_y = tl.load(kp_ptr + b * s_kpb + k * s_kpk).to(tl.float32)
    pos_x = tl.load(kp_ptr + b * s_kpb + k * s_kpk + s_kpd).to(tl.float32)
    sc_y = tl.load(sc_ptr + b * s_sb + k * s_sk).to(tl.float32)
    sc_x = tl.load(sc_ptr + b * s_sb + k * s_sk + s_sd).to(tl.float32)
    den_y = sc_y + 1.0e-9
    den_x = sc_x + 1.0e-9
    inv_y = 1.0 / den_y
    inv_x = 1.0 / den_x
    norm_x = base_x * inv_x + (-pos_x / den_x)
    norm_y = base_y * inv_y + (-pos_y / den_y)
    px = ((norm_x + 1.0) * PATCH - 1.0) * 0.5
    py = ((norm_y + 1.0) * PATCH - 1.0) * 0.5
    return px, py, inv_x, inv_y, pos_x, pos_y


@triton.jit
def _composite_sp_kernel(
        patches_ptr, kp_ptr, sc_ptr, on_ptr, dep_ptr, base_axis_ptr,
        g_rgb_ptr, g_am_ptr, g_mk_ptr, S_ptr, P_ptr, n_pixels,
        s_pb, s_pk, s_pc, s_ph, s_pw, s_kpb, s_kpk, s_kpd,
        s_sb, s_sk, s_sd, s_ob, s_ok, s_db, s_dk,
        s_gr_b, s_gr_c, s_gr_h, s_gr_w, s_ga_b, s_ga_h, s_ga_w,
        s_gm_b, s_gm_k, s_gm_h, s_gm_w, s_spb, EPS_C,
        N_KP: tl.constexpr, PATCH: tl.constexpr, IMAGE: tl.constexpr, BLOCK: tl.constexpr,
        HAS_RGB: tl.constexpr, HAS_AM: tl.constexpr, HAS_MK: tl.constexpr):
    b = tl.program_id(0)
    pix = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = pix < n_pixels
    cx = pix % IMAGE
    cy = (pix // IMAGE) % IMAGE
    base_x = tl.load(base_axis_ptr + cx, mask=active, other=0.0)
    base_y = tl.load(base_axis_ptr + cy, mask=active, other=0.0)

    g_r = tl.zeros([BLOCK], dtype=tl.float32)
    g_g = tl.zeros([BLOCK], dtype=tl.float32)
    g_b = tl.zeros([BLOCK], dtype=tl.float32)
    if HAS_RGB:
        g_r = tl.load(g_rgb_ptr + b * s_gr_b + 0 * s_gr_c + cy * s_gr_h + cx * s_gr_w,
                      mask=active, other=0.0).to(tl.float32)
        g_g = tl.load(g_rgb_ptr + b * s_gr_b + 1 * s_gr_c + cy * s_gr_h + cx * s_gr_w,
                      mask=active, other=0.0).to(tl.float32)
        g_b = tl.load(g_rgb_ptr + b * s_gr_b + 2 * s_gr_c + cy * s_gr_h + cx * s_gr_w,
                      mask=active, other=0.0).to(tl.float32)
    g_a = tl.zeros([BLOCK], dtype=tl.float32)
    if HAS_AM:
        g_a = tl.load(g_am_ptr + b * s_ga_b + cy * s_ga_h + cx * s_ga_w,
                      mask=active, other=0.0).to(tl.float32)

    # pass 1: S
    S = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(N_KP):
        px, py, _, _, _, _ = _coords(kp_ptr, sc_ptr, b, k, s_kpb, s_kpk, s_kpd,
                                     s_sb, s_sk, s_sd, base_x, base_y, PATCH)
        x0f = tl.floor(px); y0f = tl.floor(py)
        x0 = x0f.to(tl.int32); y0 = y0f.to(tl.int32); x1 = x0 + 1; y1 = y0 + 1
        dx = px - x0f; dy = py - y0f
        vx0 = (x0 >= 0) & (x0 < PATCH); vx1 = (x1 >= 0) & (x1 < PATCH)
        vy0 = (y0 >= 0) & (y0 < PATCH); vy1 = (y1 >= 0) & (y1 < PATCH)
        alpha, _, _ = _bilinear(patches_ptr, b * s_pb + k * s_pk + 0 * s_pc,
                                y0, x0, y1, x1, dx, dy, vx0, vx1, vy0, vy1,
                                s_ph, s_pw, active, PATCH)
        on = tl.load(on_ptr + b * s_ob + k * s_ok).to(tl.float32)
        dep = tl.load(dep_ptr + b * s_db + k * s_dk).to(tl.float32)
        S += (on * alpha) * tl.sigmoid(-dep)
    S = S + EPS_C

    # pass 2: grad_S, the denominator branch of importance = u / S.
    # torch's div backward wrt `other` is -grad * ((self/other)/other): TWO
    # separate correctly-rounded divides, not (-grad*self)/(other*other).  The
    # latter is algebraically equal but rounds differently and was the residual
    # error here.  Summed over the broadcast particle axis, sequentially from
    # zero, which is what autograd's reduction does for this layout.
    P = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(N_KP):
        px, py, _, _, _, _ = _coords(kp_ptr, sc_ptr, b, k, s_kpb, s_kpk, s_kpd,
                                     s_sb, s_sk, s_sd, base_x, base_y, PATCH)
        x0f = tl.floor(px); y0f = tl.floor(py)
        x0 = x0f.to(tl.int32); y0 = y0f.to(tl.int32); x1 = x0 + 1; y1 = y0 + 1
        dx = px - x0f; dy = py - y0f
        vx0 = (x0 >= 0) & (x0 < PATCH); vx1 = (x1 >= 0) & (x1 < PATCH)
        vy0 = (y0 >= 0) & (y0 < PATCH); vy1 = (y1 >= 0) & (y1 < PATCH)
        pb = b * s_pb + k * s_pk
        alpha, _, _ = _bilinear(patches_ptr, pb + 0 * s_pc, y0, x0, y1, x1, dx, dy,
                                vx0, vx1, vy0, vy1, s_ph, s_pw, active, PATCH)
        r, _, _ = _bilinear(patches_ptr, pb + 1 * s_pc, y0, x0, y1, x1, dx, dy,
                            vx0, vx1, vy0, vy1, s_ph, s_pw, active, PATCH)
        g, _, _ = _bilinear(patches_ptr, pb + 2 * s_pc, y0, x0, y1, x1, dx, dy,
                            vx0, vx1, vy0, vy1, s_ph, s_pw, active, PATCH)
        bl, _, _ = _bilinear(patches_ptr, pb + 3 * s_pc, y0, x0, y1, x1, dx, dy,
                             vx0, vx1, vy0, vy1, s_ph, s_pw, active, PATCH)
        on = tl.load(on_ptr + b * s_ob + k * s_ok).to(tl.float32)
        dep = tl.load(dep_ptr + b * s_db + k * s_dk).to(tl.float32)
        a_k = on * alpha
        u = a_k * tl.sigmoid(-dep)
        grad_importance = tl.zeros([BLOCK], dtype=tl.float32)
        if HAS_MK:
            g_m = tl.load(g_mk_ptr + b * s_gm_b + k * s_gm_k + cy * s_gm_h + cx * s_gm_w,
                          mask=active, other=0.0).to(tl.float32)
        if HAS_MK and HAS_AM:
            grad_importance = _mk_bg_total(a_k, g_m, g_a)
        elif HAS_MK:
            grad_importance += a_k * g_m
        elif HAS_AM:
            grad_importance += -a_k * g_a
        if HAS_RGB:
            grad_importance += _rgb_total(a_k, r, g, bl, g_r, g_g, g_b)
        P = _sub_prod(P, grad_importance, _div_rn(_div_rn(u, S), S))

    tl.store(S_ptr + b * s_spb + pix, S, mask=active)
    tl.store(P_ptr + b * s_spb + pix, P, mask=active)


@triton.jit
def _composite_grad_kernel(
        patches_ptr, kp_ptr, sc_ptr, on_ptr, dep_ptr, base_axis_ptr,
        g_rgb_ptr, g_am_ptr, g_mk_ptr, S_ptr, P_ptr,
        gpatch_ptr, gkp_ptr, gsc_ptr, gon_ptr, gdep_ptr, n_pixels,
        s_pb, s_pk, s_pc, s_ph, s_pw, s_kpb, s_kpk, s_kpd,
        s_sb, s_sk, s_sd, s_ob, s_ok, s_db, s_dk,
        s_gr_b, s_gr_c, s_gr_h, s_gr_w, s_ga_b, s_ga_h, s_ga_w,
        s_gm_b, s_gm_k, s_gm_h, s_gm_w, s_spb,
        N_KP: tl.constexpr, PATCH: tl.constexpr, IMAGE: tl.constexpr, BLOCK: tl.constexpr,
        HAS_RGB: tl.constexpr, HAS_AM: tl.constexpr, HAS_MK: tl.constexpr):
    b = tl.program_id(0)
    k = tl.program_id(1)
    on = tl.load(on_ptr + b * s_ob + k * s_ok).to(tl.float32)
    dep = tl.load(dep_ptr + b * s_db + k * s_dk).to(tl.float32)
    sig = tl.sigmoid(-dep)

    acc_pos_y = tl.zeros([1], dtype=tl.float32)
    acc_pos_x = tl.zeros([1], dtype=tl.float32)
    acc_sc_y = tl.zeros([1], dtype=tl.float32)
    acc_sc_x = tl.zeros([1], dtype=tl.float32)
    acc_on = tl.zeros([1], dtype=tl.float32)
    acc_dep = tl.zeros([1], dtype=tl.float32)

    n_blocks = tl.cdiv(n_pixels, BLOCK)
    for blk in range(n_blocks):
        pix = blk * BLOCK + tl.arange(0, BLOCK)
        active = pix < n_pixels
        cx = pix % IMAGE
        cy = (pix // IMAGE) % IMAGE
        base_x = tl.load(base_axis_ptr + cx, mask=active, other=0.0)
        base_y = tl.load(base_axis_ptr + cy, mask=active, other=0.0)
        S = tl.load(S_ptr + b * s_spb + pix, mask=active, other=1.0)
        P = tl.load(P_ptr + b * s_spb + pix, mask=active, other=0.0)

        px, py, inv_x, inv_y, pos_x, pos_y = _coords(
            kp_ptr, sc_ptr, b, k, s_kpb, s_kpk, s_kpd, s_sb, s_sk, s_sd,
            base_x, base_y, PATCH)
        x0f = tl.floor(px); y0f = tl.floor(py)
        x0 = x0f.to(tl.int32); y0 = y0f.to(tl.int32); x1 = x0 + 1; y1 = y0 + 1
        dx = px - x0f; dy = py - y0f
        vx0 = (x0 >= 0) & (x0 < PATCH); vx1 = (x1 >= 0) & (x1 < PATCH)
        vy0 = (y0 >= 0) & (y0 < PATCH); vy1 = (y1 >= 0) & (y1 < PATCH)
        pb = b * s_pb + k * s_pk

        alpha, a_ddx, a_ddy = _bilinear(patches_ptr, pb + 0 * s_pc, y0, x0, y1, x1, dx, dy,
                                        vx0, vx1, vy0, vy1, s_ph, s_pw, active, PATCH)
        r, r_ddx, r_ddy = _bilinear(patches_ptr, pb + 1 * s_pc, y0, x0, y1, x1, dx, dy,
                                    vx0, vx1, vy0, vy1, s_ph, s_pw, active, PATCH)
        g, g_ddx, g_ddy = _bilinear(patches_ptr, pb + 2 * s_pc, y0, x0, y1, x1, dx, dy,
                                    vx0, vx1, vy0, vy1, s_ph, s_pw, active, PATCH)
        bl, b_ddx, b_ddy = _bilinear(patches_ptr, pb + 3 * s_pc, y0, x0, y1, x1, dx, dy,
                                     vx0, vx1, vy0, vy1, s_ph, s_pw, active, PATCH)

        g_r = tl.zeros([BLOCK], dtype=tl.float32)
        g_g = tl.zeros([BLOCK], dtype=tl.float32)
        g_b = tl.zeros([BLOCK], dtype=tl.float32)
        if HAS_RGB:
            g_r = tl.load(g_rgb_ptr + b*s_gr_b + 0*s_gr_c + cy*s_gr_h + cx*s_gr_w,
                          mask=active, other=0.0).to(tl.float32)
            g_g = tl.load(g_rgb_ptr + b*s_gr_b + 1*s_gr_c + cy*s_gr_h + cx*s_gr_w,
                          mask=active, other=0.0).to(tl.float32)
            g_b = tl.load(g_rgb_ptr + b*s_gr_b + 2*s_gr_c + cy*s_gr_h + cx*s_gr_w,
                          mask=active, other=0.0).to(tl.float32)
        g_a = tl.zeros([BLOCK], dtype=tl.float32)
        if HAS_AM:
            g_a = tl.load(g_am_ptr + b*s_ga_b + cy*s_ga_h + cx*s_ga_w,
                          mask=active, other=0.0).to(tl.float32)
        g_m = tl.zeros([BLOCK], dtype=tl.float32)
        if HAS_MK:
            g_m = tl.load(g_mk_ptr + b*s_gm_b + k*s_gm_k + cy*s_gm_h + cx*s_gm_w,
                          mask=active, other=0.0).to(tl.float32)

        a_k = on * alpha
        u = a_k * sig
        # same correctly-rounded divide as the fused forward's importance
        w = tl.math.div_rn(u, S)
        grad_importance = tl.zeros([BLOCK], dtype=tl.float32)
        grad_a_direct = tl.zeros([BLOCK], dtype=tl.float32)
        if HAS_MK and HAS_AM:
            grad_importance = _mk_bg_total(a_k, g_m, g_a)
        elif HAS_MK:
            grad_importance += a_k * g_m
        elif HAS_AM:
            grad_importance += -a_k * g_a
        if HAS_MK:
            grad_a_direct += w * g_m
        if HAS_AM:
            grad_a_direct += -w * g_a
        if HAS_RGB:
            grad_importance += _rgb_total(a_k, r, g, bl, g_r, g_g, g_b)
            grad_a_direct += (w * g_r) * r + (w * g_g) * g + (w * g_b) * bl
        # importance = u / S : numerator branch, then the S = sum(u) branch adds
        # grad_S uniformly across particles. Same node order as the reference.
        dL_dimp = _div_rn(grad_importance, S) + P
        dL_da = grad_a_direct + dL_dimp * sig
        dL_dalpha = dL_da * on
        dL_dr = a_k * w * g_r
        dL_dg = a_k * w * g_g
        dL_db = a_k * w * g_b

        acc_on += tl.sum(tl.where(active, dL_da * alpha, 0.0))
        # z_depth is [B,K,1] broadcast over pixels, so the reference reduces
        # grad_depth over pixels BEFORE applying sigmoid backward once. Match that
        # order rather than folding the constant factor into the per-pixel term.
        acc_dep += tl.sum(tl.where(active, dL_dimp * a_k, 0.0))

        # scatter into the patch (atomic: several output pixels hit one texel)
        for c in range(4):
            gv = tl.where(c == 0, dL_dalpha, tl.where(c == 1, dL_dr,
                          tl.where(c == 2, dL_dg, dL_db)))
            gv = tl.where(active, gv, 0.0)
            cb = pb + c * s_pc
            tl.atomic_add(gpatch_ptr + cb + y0*s_ph + x0*s_pw, gv*(1.0-dx)*(1.0-dy),
                          mask=active & vx0 & vy0)
            tl.atomic_add(gpatch_ptr + cb + y0*s_ph + x1*s_pw, gv*dx*(1.0-dy),
                          mask=active & vx1 & vy0)
            tl.atomic_add(gpatch_ptr + cb + y1*s_ph + x0*s_pw, gv*(1.0-dx)*dy,
                          mask=active & vx0 & vy1)
            tl.atomic_add(gpatch_ptr + cb + y1*s_ph + x1*s_pw, gv*dx*dy,
                          mask=active & vx1 & vy1)

        dL_dpx = dL_dalpha*a_ddx + dL_dr*r_ddx + dL_dg*g_ddx + dL_db*b_ddx
        dL_dpy = dL_dalpha*a_ddy + dL_dr*r_ddy + dL_dg*g_ddy + dL_db*b_ddy
        dnx = dL_dpx * (PATCH * 0.5)
        dny = dL_dpy * (PATCH * 0.5)
        acc_pos_x += tl.sum(tl.where(active, dnx * (-inv_x), 0.0))
        acc_pos_y += tl.sum(tl.where(active, dny * (-inv_y), 0.0))
        acc_sc_x += tl.sum(tl.where(active, dnx * (pos_x - base_x) * inv_x * inv_x, 0.0))
        acc_sc_y += tl.sum(tl.where(active, dny * (pos_y - base_y) * inv_y * inv_y, 0.0))

    tl.store(gkp_ptr + b*s_kpb + k*s_kpk, tl.sum(acc_pos_y))
    tl.store(gkp_ptr + b*s_kpb + k*s_kpk + s_kpd, tl.sum(acc_pos_x))
    tl.store(gsc_ptr + b*s_sb + k*s_sk, tl.sum(acc_sc_y))
    tl.store(gsc_ptr + b*s_sb + k*s_sk + s_sd, tl.sum(acc_sc_x))
    tl.store(gon_ptr + b*s_ob + k*s_ok, tl.sum(acc_on))
    # sigmoid_backward then neg, in the reference's order
    tl.store(gdep_ptr + b*s_db + k*s_dk, -(tl.sum(acc_dep) * sig * (1.0 - sig)))


def composite_backward_triton(dec_objects, z_kp, z_scale, obj_on, z_depth, img_size,
                              grad_masks, grad_bg, grad_rgb, eps=1e-5,
                              scale_normalized=False, block=256):
    """Standalone backward. No autograd, no reference call, no [B,K,C,H,W] canvas.

    Returns gradients for (dec_objects, z_kp, z_scale, obj_on, z_depth); the
    z_scale entry is None when z_scale was None, because the normalized scale is
    then a constant derived from the patch and image sizes.

    Any of grad_masks / grad_bg / grad_rgb may be None when its output was unused;
    the corresponding term is compiled out of both kernels.
    """
    bs, n_kp, ch, patch, _ = dec_objects.shape
    img = int(img_size)
    dev = dec_objects.device

    if z_scale is None:
        normalized_scale = (patch / img) * torch.ones_like(z_kp)
    elif scale_normalized:
        normalized_scale = z_scale
    else:
        normalized_scale = torch.sigmoid(z_scale)
    normalized_scale = normalized_scale.contiguous()

    n_pixels = img * img
    S = torch.empty(bs, n_pixels, device=dev, dtype=torch.float32)
    P = torch.empty(bs, n_pixels, device=dev, dtype=torch.float32)

    has_rgb, has_am, has_mk = grad_rgb is not None, grad_bg is not None, grad_masks is not None
    z = torch.zeros(1, device=dev, dtype=torch.float32)
    gr = grad_rgb.contiguous() if has_rgb else z
    ga = grad_bg.contiguous() if has_am else z
    gm = grad_masks.contiguous() if has_mk else z
    sr = (gr.stride(0), gr.stride(1), gr.stride(2), gr.stride(3)) if has_rgb else (0, 0, 0, 0)
    sa = (ga.stride(0), ga.stride(2), ga.stride(3)) if has_am else (0, 0, 0)
    sm = (gm.stride(0), gm.stride(1), gm.stride(3), gm.stride(4)) if has_mk else (0, 0, 0, 0)

    common = dict(N_KP=n_kp, PATCH=patch, IMAGE=img, BLOCK=block,
                  HAS_RGB=has_rgb, HAS_AM=has_am, HAS_MK=has_mk)
    strides = (dec_objects.stride(0), dec_objects.stride(1), dec_objects.stride(2),
               dec_objects.stride(3), dec_objects.stride(4),
               z_kp.stride(0), z_kp.stride(1), z_kp.stride(2),
               normalized_scale.stride(0), normalized_scale.stride(1), normalized_scale.stride(2),
               obj_on.stride(0), obj_on.stride(1),
               z_depth.stride(0), z_depth.stride(1))
    base = _base_axis(dev, img)

    _composite_sp_kernel[(bs, triton.cdiv(n_pixels, block))](
        dec_objects, z_kp, normalized_scale, obj_on, z_depth, base,
        gr, ga, gm, S, P, n_pixels, *strides, *sr, *sa, *sm, S.stride(0), eps, **common)

    grad_patches = torch.zeros_like(dec_objects)
    grad_kp = torch.empty_like(z_kp)
    grad_ns = torch.empty_like(normalized_scale)
    grad_on = torch.empty_like(obj_on)
    grad_dep = torch.empty_like(z_depth)

    _composite_grad_kernel[(bs, n_kp)](
        dec_objects, z_kp, normalized_scale, obj_on, z_depth, base,
        gr, ga, gm, S, P,
        grad_patches, grad_kp, grad_ns, grad_on, grad_dep, n_pixels,
        *strides, *sr, *sa, *sm, S.stride(0), **common)

    if z_scale is None:
        grad_scale = None
    elif scale_normalized:
        grad_scale = grad_ns
    else:
        grad_scale = grad_ns * normalized_scale * (1.0 - normalized_scale)
    return grad_patches, grad_kp, grad_scale, grad_on, grad_dep
