"""Triton backend for the isolated LPWM spatial transformer.

The optimized :func:`stn_crop` reads the source image through the original
batch index instead of materializing the reference's per-particle repeat.
The optimized :func:`stn_paste` evaluates the inverse affine map directly
instead of materializing theta and its full sampling grid.

The custom backwards mirror PyTorch's bilinear sampling derivatives. Crop
atomically accumulates the shared-image gradient; paste atomically accumulates
patch gradients and uses a two-stage per-particle parameter reduction.
"""

import torch

from . import reference

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # The reference backend remains usable without Triton.
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _stn_crop_forward_kernel(
            x_ptr, kp_ptr, scale_ptr, base_axis_ptr, out_ptr, n_elements,
            stride_xb, stride_xc, stride_xh, stride_xw,
            stride_kpb, stride_kpk, stride_kpd,
            stride_sb, stride_sk, stride_sd,
            N_KP: tl.constexpr, CHANNELS: tl.constexpr,
            HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
            PATCH: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < n_elements

        out_x = offsets % PATCH
        quotient = offsets // PATCH
        out_y = quotient % PATCH
        quotient = quotient // PATCH
        channel = quotient % CHANNELS
        particle_row = quotient // CHANNELS
        batch = particle_row // N_KP
        particle = particle_row % N_KP

        kp_base = batch * stride_kpb + particle * stride_kpk
        pos_y = tl.load(kp_ptr + kp_base, mask=active, other=0.0).to(tl.float32)
        pos_x = tl.load(kp_ptr + kp_base + stride_kpd, mask=active, other=0.0).to(tl.float32)

        scale_base = batch * stride_sb + particle * stride_sk
        scale_y = tl.load(scale_ptr + scale_base, mask=active, other=0.0).to(tl.float32)
        scale_x = tl.load(scale_ptr + scale_base + stride_sd, mask=active, other=0.0).to(tl.float32)

        # affine_grid(..., align_corners=False), followed by the corresponding
        # normalized-coordinate to input-pixel conversion in grid_sample.
        # affine_grid constructs this axis through linspace(-1, 1) followed
        # by two fp32 pointwise ops. Loading that tiny cached axis reproduces
        # its rounding while still avoiding the full [N, P, P, 2] grid.
        base_x = tl.load(base_axis_ptr + out_x, mask=active, other=0.0)
        base_y = tl.load(base_axis_ptr + out_y, mask=active, other=0.0)
        norm_x = base_x * scale_x + pos_x
        norm_y = base_y * scale_y + pos_y
        image_x = ((norm_x + 1.0) * WIDTH - 1.0) * 0.5
        image_y = ((norm_y + 1.0) * HEIGHT - 1.0) * 0.5

        # padding_mode='border': clamp before forming the bilinear neighbors.
        image_x = tl.minimum(tl.maximum(image_x, 0.0), WIDTH - 1.0)
        image_y = tl.minimum(tl.maximum(image_y, 0.0), HEIGHT - 1.0)

        x0_float = tl.floor(image_x)
        y0_float = tl.floor(image_y)
        x1_float = x0_float + 1.0
        y1_float = y0_float + 1.0
        x0 = x0_float.to(tl.int32)
        y0 = y0_float.to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        north_west = (x1_float - image_x) * (y1_float - image_y)
        north_east = (image_x - x0_float) * (y1_float - image_y)
        south_west = (x1_float - image_x) * (image_y - y0_float)
        south_east = (image_x - x0_float) * (image_y - y0_float)

        image_base = batch * stride_xb + channel * stride_xc
        valid_x0 = (x0 >= 0) & (x0 < WIDTH)
        valid_x1 = (x1 >= 0) & (x1 < WIDTH)
        valid_y0 = (y0 >= 0) & (y0 < HEIGHT)
        valid_y1 = (y1 >= 0) & (y1 < HEIGHT)

        v00 = tl.load(x_ptr + image_base + y0 * stride_xh + x0 * stride_xw,
                      mask=active & valid_x0 & valid_y0, other=0.0).to(tl.float32)
        v01 = tl.load(x_ptr + image_base + y0 * stride_xh + x1 * stride_xw,
                      mask=active & valid_x1 & valid_y0, other=0.0).to(tl.float32)
        v10 = tl.load(x_ptr + image_base + y1 * stride_xh + x0 * stride_xw,
                      mask=active & valid_x0 & valid_y1, other=0.0).to(tl.float32)
        v11 = tl.load(x_ptr + image_base + y1 * stride_xh + x1 * stride_xw,
                      mask=active & valid_x1 & valid_y1, other=0.0).to(tl.float32)

        value = v00 * north_west + v01 * north_east
        value += v10 * south_west + v11 * south_east
        tl.store(out_ptr + offsets, value, mask=active)


    @triton.jit
    def _stn_crop_grad_x_kernel(
            grad_out_ptr, kp_ptr, scale_ptr, base_axis_ptr, grad_x_ptr, n_elements,
            stride_gon, stride_goc, stride_goh, stride_gow,
            stride_kpb, stride_kpk, stride_kpd,
            stride_sb, stride_sk, stride_sd,
            stride_gxb, stride_gxc, stride_gxh, stride_gxw,
            N_KP: tl.constexpr, CHANNELS: tl.constexpr,
            HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
            PATCH: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < n_elements

        out_x = offsets % PATCH
        quotient = offsets // PATCH
        out_y = quotient % PATCH
        quotient = quotient // PATCH
        channel = quotient % CHANNELS
        particle_row = quotient // CHANNELS
        batch = particle_row // N_KP
        particle = particle_row % N_KP

        kp_base = batch * stride_kpb + particle * stride_kpk
        pos_y = tl.load(kp_ptr + kp_base, mask=active, other=0.0).to(tl.float32)
        pos_x = tl.load(kp_ptr + kp_base + stride_kpd, mask=active, other=0.0).to(tl.float32)

        scale_base = batch * stride_sb + particle * stride_sk
        scale_y = tl.load(scale_ptr + scale_base, mask=active, other=0.0).to(tl.float32)
        scale_x = tl.load(scale_ptr + scale_base + stride_sd, mask=active, other=0.0).to(tl.float32)

        base_x = tl.load(base_axis_ptr + out_x, mask=active, other=0.0)
        base_y = tl.load(base_axis_ptr + out_y, mask=active, other=0.0)
        norm_x = base_x * scale_x + pos_x
        norm_y = base_y * scale_y + pos_y
        image_x = ((norm_x + 1.0) * WIDTH - 1.0) * 0.5
        image_y = ((norm_y + 1.0) * HEIGHT - 1.0) * 0.5
        image_x = tl.minimum(tl.maximum(image_x, 0.0), WIDTH - 1.0)
        image_y = tl.minimum(tl.maximum(image_y, 0.0), HEIGHT - 1.0)

        x0_float = tl.floor(image_x)
        y0_float = tl.floor(image_y)
        x0 = x0_float.to(tl.int32)
        y0 = y0_float.to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        dx = image_x - x0_float
        dy = image_y - y0_float

        grad_offset = (particle_row * stride_gon + channel * stride_goc +
                       out_y * stride_goh + out_x * stride_gow)
        grad = tl.load(grad_out_ptr + grad_offset, mask=active, other=0.0).to(tl.float32)
        image_base = batch * stride_gxb + channel * stride_gxc

        valid_x0 = (x0 >= 0) & (x0 < WIDTH)
        valid_x1 = (x1 >= 0) & (x1 < WIDTH)
        valid_y0 = (y0 >= 0) & (y0 < HEIGHT)
        valid_y1 = (y1 >= 0) & (y1 < HEIGHT)
        tl.atomic_add(grad_x_ptr + image_base + y0 * stride_gxh + x0 * stride_gxw,
                      grad * (1.0 - dx) * (1.0 - dy),
                      mask=active & valid_x0 & valid_y0)
        tl.atomic_add(grad_x_ptr + image_base + y0 * stride_gxh + x1 * stride_gxw,
                      grad * dx * (1.0 - dy),
                      mask=active & valid_x1 & valid_y0)
        tl.atomic_add(grad_x_ptr + image_base + y1 * stride_gxh + x0 * stride_gxw,
                      grad * (1.0 - dx) * dy,
                      mask=active & valid_x0 & valid_y1)
        tl.atomic_add(grad_x_ptr + image_base + y1 * stride_gxh + x1 * stride_gxw,
                      grad * dx * dy,
                      mask=active & valid_x1 & valid_y1)


    @triton.jit
    def _stn_crop_grad_params_kernel(
            grad_out_ptr, x_ptr, kp_ptr, scale_ptr, base_axis_ptr,
            grad_kp_ptr, grad_scale_ptr,
            stride_gon, stride_goc, stride_goh, stride_gow,
            stride_xb, stride_xc, stride_xh, stride_xw,
            stride_kpb, stride_kpk, stride_kpd,
            stride_sb, stride_sk, stride_sd,
            N_KP: tl.constexpr, CHANNELS: tl.constexpr,
            HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
            PATCH: tl.constexpr, SPATIAL_ELEMS: tl.constexpr,
            HAS_SCALE: tl.constexpr, NEED_KP: tl.constexpr,
            NEED_SCALE: tl.constexpr, BLOCK: tl.constexpr):
        particle_row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        active = offsets < SPATIAL_ELEMS

        out_x = offsets % PATCH
        out_y = offsets // PATCH
        batch = particle_row // N_KP
        particle = particle_row % N_KP

        kp_base = batch * stride_kpb + particle * stride_kpk
        pos_y = tl.load(kp_ptr + kp_base).to(tl.float32)
        pos_x = tl.load(kp_ptr + kp_base + stride_kpd).to(tl.float32)
        scale_base = batch * stride_sb + particle * stride_sk
        scale_y = tl.load(scale_ptr + scale_base).to(tl.float32)
        scale_x = tl.load(scale_ptr + scale_base + stride_sd).to(tl.float32)

        base_x = tl.load(base_axis_ptr + out_x, mask=active, other=0.0)
        base_y = tl.load(base_axis_ptr + out_y, mask=active, other=0.0)
        norm_x = base_x * scale_x + pos_x
        norm_y = base_y * scale_y + pos_y
        raw_x = ((norm_x + 1.0) * WIDTH - 1.0) * 0.5
        raw_y = ((norm_y + 1.0) * HEIGHT - 1.0) * 0.5
        image_x = tl.minimum(tl.maximum(raw_x, 0.0), WIDTH - 1.0)
        image_y = tl.minimum(tl.maximum(raw_y, 0.0), HEIGHT - 1.0)

        x0_float = tl.floor(image_x)
        y0_float = tl.floor(image_y)
        x0 = x0_float.to(tl.int32)
        y0 = y0_float.to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        dx = image_x - x0_float
        dy = image_y - y0_float

        valid_x0 = (x0 >= 0) & (x0 < WIDTH)
        valid_x1 = (x1 >= 0) & (x1 < WIDTH)
        valid_y0 = (y0 >= 0) & (y0 < HEIGHT)
        valid_y1 = (y1 >= 0) & (y1 < HEIGHT)
        grad_image_x = tl.zeros((BLOCK,), dtype=tl.float32)
        grad_image_y = tl.zeros((BLOCK,), dtype=tl.float32)
        for channel in range(CHANNELS):
            image_base = batch * stride_xb + channel * stride_xc
            v00 = tl.load(x_ptr + image_base + y0 * stride_xh + x0 * stride_xw,
                          mask=active & valid_x0 & valid_y0, other=0.0).to(tl.float32)
            v01 = tl.load(x_ptr + image_base + y0 * stride_xh + x1 * stride_xw,
                          mask=active & valid_x1 & valid_y0, other=0.0).to(tl.float32)
            v10 = tl.load(x_ptr + image_base + y1 * stride_xh + x0 * stride_xw,
                          mask=active & valid_x0 & valid_y1, other=0.0).to(tl.float32)
            v11 = tl.load(x_ptr + image_base + y1 * stride_xh + x1 * stride_xw,
                          mask=active & valid_x1 & valid_y1, other=0.0).to(tl.float32)

            grad_offset = (particle_row * stride_gon + channel * stride_goc +
                           out_y * stride_goh + out_x * stride_gow)
            grad = tl.load(grad_out_ptr + grad_offset, mask=active, other=0.0).to(tl.float32)

            # Preserve grid_sampler_2d_backward_kernel's channel-wise update
            # order instead of algebraically reassociating the four terms.
            grad_image_x -= v00 * (1.0 - dy) * grad
            grad_image_y -= v00 * (1.0 - dx) * grad
            grad_image_x += v01 * (1.0 - dy) * grad
            grad_image_y -= v01 * dx * grad
            grad_image_x -= v10 * dy * grad
            grad_image_y += v10 * (1.0 - dx) * grad
            grad_image_x += v11 * dy * grad
            grad_image_y += v11 * dx * grad

        # padding_mode='border' has zero derivative at and beyond either edge.
        grad_norm_x = tl.where((raw_x > 0.0) & (raw_x < WIDTH - 1.0),
                               grad_image_x * (WIDTH * 0.5), 0.0)
        grad_norm_y = tl.where((raw_y > 0.0) & (raw_y < HEIGHT - 1.0),
                               grad_image_y * (HEIGHT * 0.5), 0.0)

        grad_pos_x = tl.sum(grad_norm_x, axis=0)
        grad_pos_y = tl.sum(grad_norm_y, axis=0)
        grad_scale_x = tl.sum(grad_norm_x * base_x, axis=0)
        grad_scale_y = tl.sum(grad_norm_y * base_y, axis=0)

        output_base = particle_row * 2
        if NEED_KP:
            tl.store(grad_kp_ptr + output_base, grad_pos_y)
            tl.store(grad_kp_ptr + output_base + 1, grad_pos_x)
        if NEED_SCALE:
            if HAS_SCALE:
                grad_scale_y *= scale_y * (1.0 - scale_y)
                grad_scale_x *= scale_x * (1.0 - scale_x)
            tl.store(grad_scale_ptr + output_base, grad_scale_y)
            tl.store(grad_scale_ptr + output_base + 1, grad_scale_x)


    @triton.jit
    def _stn_paste_forward_kernel(
            patches_ptr, kp_ptr, scale_ptr, base_axis_ptr, out_ptr, n_elements,
            stride_pb, stride_pk, stride_pc, stride_ph, stride_pw,
            stride_kpb, stride_kpk, stride_kpd,
            stride_sb, stride_sk, stride_sd,
            N_KP: tl.constexpr, CHANNELS: tl.constexpr,
            PATCH: tl.constexpr, IMAGE: tl.constexpr,
            BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < n_elements

        canvas_x = offsets % IMAGE
        quotient = offsets // IMAGE
        canvas_y = quotient % IMAGE
        quotient = quotient // IMAGE
        channel = quotient % CHANNELS
        particle_row = quotient // CHANNELS
        batch = particle_row // N_KP
        particle = particle_row % N_KP

        kp_base = batch * stride_kpb + particle * stride_kpk
        pos_y = tl.load(kp_ptr + kp_base, mask=active, other=0.0).to(tl.float32)
        pos_x = tl.load(kp_ptr + kp_base + stride_kpd, mask=active, other=0.0).to(tl.float32)
        scale_base = batch * stride_sb + particle * stride_sk
        scale_y = tl.load(scale_ptr + scale_base, mask=active, other=1.0).to(tl.float32)
        scale_x = tl.load(scale_ptr + scale_base + stride_sd, mask=active, other=1.0).to(tl.float32)

        # Match spatial_transform(inverse=True): scale and translation are two
        # separately rounded divisions before affine_grid applies the matrix.
        denominator_y = scale_y + 1.0e-9
        denominator_x = scale_x + 1.0e-9
        inv_y = 1.0 / denominator_y
        inv_x = 1.0 / denominator_x
        translate_y = -pos_y / denominator_y
        translate_x = -pos_x / denominator_x
        base_x = tl.load(base_axis_ptr + canvas_x, mask=active, other=0.0)
        base_y = tl.load(base_axis_ptr + canvas_y, mask=active, other=0.0)
        norm_x = base_x * inv_x + translate_x
        norm_y = base_y * inv_y + translate_y
        patch_x = ((norm_x + 1.0) * PATCH - 1.0) * 0.5
        patch_y = ((norm_y + 1.0) * PATCH - 1.0) * 0.5

        x0_float = tl.floor(patch_x)
        y0_float = tl.floor(patch_y)
        x0 = x0_float.to(tl.int32)
        y0 = y0_float.to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        dx = patch_x - x0_float
        dy = patch_y - y0_float

        patch_base = (batch * stride_pb + particle * stride_pk +
                      channel * stride_pc)
        valid_x0 = (x0 >= 0) & (x0 < PATCH)
        valid_x1 = (x1 >= 0) & (x1 < PATCH)
        valid_y0 = (y0 >= 0) & (y0 < PATCH)
        valid_y1 = (y1 >= 0) & (y1 < PATCH)
        v00 = tl.load(patches_ptr + patch_base + y0 * stride_ph + x0 * stride_pw,
                      mask=active & valid_x0 & valid_y0, other=0.0).to(tl.float32)
        v01 = tl.load(patches_ptr + patch_base + y0 * stride_ph + x1 * stride_pw,
                      mask=active & valid_x1 & valid_y0, other=0.0).to(tl.float32)
        v10 = tl.load(patches_ptr + patch_base + y1 * stride_ph + x0 * stride_pw,
                      mask=active & valid_x0 & valid_y1, other=0.0).to(tl.float32)
        v11 = tl.load(patches_ptr + patch_base + y1 * stride_ph + x1 * stride_pw,
                      mask=active & valid_x1 & valid_y1, other=0.0).to(tl.float32)

        value = v00 * (1.0 - dx) * (1.0 - dy)
        value += v01 * dx * (1.0 - dy)
        value += v10 * (1.0 - dx) * dy
        value += v11 * dx * dy
        tl.store(out_ptr + offsets, value, mask=active)


    @triton.jit
    def _stn_paste_grad_patches_kernel(
            grad_out_ptr, kp_ptr, scale_ptr, base_axis_ptr,
            grad_patches_ptr, n_elements,
            stride_gob, stride_gok, stride_goc, stride_goh, stride_gow,
            stride_kpb, stride_kpk, stride_kpd,
            stride_sb, stride_sk, stride_sd,
            stride_gpb, stride_gpk, stride_gpc, stride_gph, stride_gpw,
            N_KP: tl.constexpr, CHANNELS: tl.constexpr,
            PATCH: tl.constexpr, IMAGE: tl.constexpr,
            BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < n_elements

        canvas_x = offsets % IMAGE
        quotient = offsets // IMAGE
        canvas_y = quotient % IMAGE
        quotient = quotient // IMAGE
        channel = quotient % CHANNELS
        particle_row = quotient // CHANNELS
        batch = particle_row // N_KP
        particle = particle_row % N_KP

        kp_base = batch * stride_kpb + particle * stride_kpk
        pos_y = tl.load(kp_ptr + kp_base, mask=active, other=0.0).to(tl.float32)
        pos_x = tl.load(kp_ptr + kp_base + stride_kpd, mask=active, other=0.0).to(tl.float32)
        scale_base = batch * stride_sb + particle * stride_sk
        scale_y = tl.load(scale_ptr + scale_base, mask=active, other=1.0).to(tl.float32)
        scale_x = tl.load(scale_ptr + scale_base + stride_sd, mask=active, other=1.0).to(tl.float32)
        denominator_y = scale_y + 1.0e-9
        denominator_x = scale_x + 1.0e-9
        inv_y = 1.0 / denominator_y
        inv_x = 1.0 / denominator_x
        translate_y = -pos_y / denominator_y
        translate_x = -pos_x / denominator_x
        base_x = tl.load(base_axis_ptr + canvas_x, mask=active, other=0.0)
        base_y = tl.load(base_axis_ptr + canvas_y, mask=active, other=0.0)
        norm_x = base_x * inv_x + translate_x
        norm_y = base_y * inv_y + translate_y
        patch_x = ((norm_x + 1.0) * PATCH - 1.0) * 0.5
        patch_y = ((norm_y + 1.0) * PATCH - 1.0) * 0.5

        x0_float = tl.floor(patch_x)
        y0_float = tl.floor(patch_y)
        x0 = x0_float.to(tl.int32)
        y0 = y0_float.to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        dx = patch_x - x0_float
        dy = patch_y - y0_float

        grad_offset = (batch * stride_gob + particle * stride_gok +
                       channel * stride_goc + canvas_y * stride_goh +
                       canvas_x * stride_gow)
        grad = tl.load(grad_out_ptr + grad_offset, mask=active, other=0.0).to(tl.float32)
        patch_base = (batch * stride_gpb + particle * stride_gpk +
                      channel * stride_gpc)
        valid_x0 = (x0 >= 0) & (x0 < PATCH)
        valid_x1 = (x1 >= 0) & (x1 < PATCH)
        valid_y0 = (y0 >= 0) & (y0 < PATCH)
        valid_y1 = (y1 >= 0) & (y1 < PATCH)
        tl.atomic_add(
            grad_patches_ptr + patch_base + y0 * stride_gph + x0 * stride_gpw,
            grad * (1.0 - dx) * (1.0 - dy),
            mask=active & valid_x0 & valid_y0)
        tl.atomic_add(
            grad_patches_ptr + patch_base + y0 * stride_gph + x1 * stride_gpw,
            grad * dx * (1.0 - dy),
            mask=active & valid_x1 & valid_y0)
        tl.atomic_add(
            grad_patches_ptr + patch_base + y1 * stride_gph + x0 * stride_gpw,
            grad * (1.0 - dx) * dy,
            mask=active & valid_x0 & valid_y1)
        tl.atomic_add(
            grad_patches_ptr + patch_base + y1 * stride_gph + x1 * stride_gpw,
            grad * dx * dy,
            mask=active & valid_x1 & valid_y1)


    @triton.jit
    def _stn_paste_grad_params_kernel(
            grad_out_ptr, patches_ptr, kp_ptr, scale_ptr, base_axis_ptr,
            partials_ptr,
            stride_gob, stride_gok, stride_goc, stride_goh, stride_gow,
            stride_pb, stride_pk, stride_pc, stride_ph, stride_pw,
            stride_kpb, stride_kpk, stride_kpd,
            stride_sb, stride_sk, stride_sd,
            N_TILES: tl.constexpr, N_KP: tl.constexpr, CHANNELS: tl.constexpr,
            PATCH: tl.constexpr, IMAGE: tl.constexpr,
            BLOCK: tl.constexpr):
        program = tl.program_id(0)
        particle_row = program // N_TILES
        tile = program % N_TILES
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        active = offsets < IMAGE * IMAGE
        canvas_x = offsets % IMAGE
        canvas_y = offsets // IMAGE
        batch = particle_row // N_KP
        particle = particle_row % N_KP

        kp_base = batch * stride_kpb + particle * stride_kpk
        pos_y = tl.load(kp_ptr + kp_base).to(tl.float32)
        pos_x = tl.load(kp_ptr + kp_base + stride_kpd).to(tl.float32)
        scale_base = batch * stride_sb + particle * stride_sk
        scale_y = tl.load(scale_ptr + scale_base).to(tl.float32)
        scale_x = tl.load(scale_ptr + scale_base + stride_sd).to(tl.float32)
        denominator_y = scale_y + 1.0e-9
        denominator_x = scale_x + 1.0e-9
        inv_y = 1.0 / denominator_y
        inv_x = 1.0 / denominator_x
        translate_y = -pos_y / denominator_y
        translate_x = -pos_x / denominator_x
        base_x = tl.load(base_axis_ptr + canvas_x, mask=active, other=0.0)
        base_y = tl.load(base_axis_ptr + canvas_y, mask=active, other=0.0)
        norm_x = base_x * inv_x + translate_x
        norm_y = base_y * inv_y + translate_y
        patch_x = ((norm_x + 1.0) * PATCH - 1.0) * 0.5
        patch_y = ((norm_y + 1.0) * PATCH - 1.0) * 0.5

        x0_float = tl.floor(patch_x)
        y0_float = tl.floor(patch_y)
        x0 = x0_float.to(tl.int32)
        y0 = y0_float.to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        dx = patch_x - x0_float
        dy = patch_y - y0_float
        valid_x0 = (x0 >= 0) & (x0 < PATCH)
        valid_x1 = (x1 >= 0) & (x1 < PATCH)
        valid_y0 = (y0 >= 0) & (y0 < PATCH)
        valid_y1 = (y1 >= 0) & (y1 < PATCH)

        grad_patch_x = tl.zeros((BLOCK,), dtype=tl.float32)
        grad_patch_y = tl.zeros((BLOCK,), dtype=tl.float32)
        for channel in range(CHANNELS):
            patch_base = (batch * stride_pb + particle * stride_pk +
                          channel * stride_pc)
            v00 = tl.load(patches_ptr + patch_base + y0 * stride_ph + x0 * stride_pw,
                          mask=active & valid_x0 & valid_y0, other=0.0).to(tl.float32)
            v01 = tl.load(patches_ptr + patch_base + y0 * stride_ph + x1 * stride_pw,
                          mask=active & valid_x1 & valid_y0, other=0.0).to(tl.float32)
            v10 = tl.load(patches_ptr + patch_base + y1 * stride_ph + x0 * stride_pw,
                          mask=active & valid_x0 & valid_y1, other=0.0).to(tl.float32)
            v11 = tl.load(patches_ptr + patch_base + y1 * stride_ph + x1 * stride_pw,
                          mask=active & valid_x1 & valid_y1, other=0.0).to(tl.float32)
            grad_offset = (batch * stride_gob + particle * stride_gok +
                           channel * stride_goc + canvas_y * stride_goh +
                           canvas_x * stride_gow)
            grad = tl.load(grad_out_ptr + grad_offset, mask=active, other=0.0).to(tl.float32)

            grad_patch_x -= v00 * (1.0 - dy) * grad
            grad_patch_y -= v00 * (1.0 - dx) * grad
            grad_patch_x += v01 * (1.0 - dy) * grad
            grad_patch_y -= v01 * dx * grad
            grad_patch_x -= v10 * dy * grad
            grad_patch_y += v10 * (1.0 - dx) * grad
            grad_patch_x += v11 * dy * grad
            grad_patch_y += v11 * dx * grad

        grad_norm_x = grad_patch_x * (PATCH * 0.5)
        grad_norm_y = grad_patch_y * (PATCH * 0.5)
        grad_pos_x = tl.sum(-grad_norm_x * inv_x, axis=0)
        grad_pos_y = tl.sum(-grad_norm_y * inv_y, axis=0)
        grad_scale_x = tl.sum(
            grad_norm_x * (pos_x - base_x) * inv_x * inv_x, axis=0)
        grad_scale_y = tl.sum(
            grad_norm_y * (pos_y - base_y) * inv_y * inv_y, axis=0)

        partial_base = (particle_row * N_TILES + tile) * 4
        tl.store(partials_ptr + partial_base, grad_pos_y)
        tl.store(partials_ptr + partial_base + 1, grad_pos_x)
        tl.store(partials_ptr + partial_base + 2, grad_scale_y)
        tl.store(partials_ptr + partial_base + 3, grad_scale_x)


    @triton.jit
    def _stn_paste_reduce_params_kernel(
            partials_ptr, scale_ptr, grad_kp_ptr, grad_scale_ptr,
            stride_sb, stride_sk, stride_sd,
            N_TILES: tl.constexpr, N_KP: tl.constexpr, HAS_SCALE: tl.constexpr,
            SCALE_NORMALIZED: tl.constexpr, NEED_KP: tl.constexpr,
            NEED_SCALE: tl.constexpr, BLOCK: tl.constexpr):
        particle_row = tl.program_id(0)
        tiles = tl.arange(0, BLOCK)
        active = tiles < N_TILES
        partial_base = (particle_row * N_TILES + tiles) * 4

        if NEED_KP:
            grad_pos_y = tl.sum(
                tl.load(partials_ptr + partial_base, mask=active, other=0.0), axis=0)
            grad_pos_x = tl.sum(
                tl.load(partials_ptr + partial_base + 1, mask=active, other=0.0), axis=0)
            tl.store(grad_kp_ptr + particle_row * 2, grad_pos_y)
            tl.store(grad_kp_ptr + particle_row * 2 + 1, grad_pos_x)
        if NEED_SCALE:
            grad_scale_y = tl.sum(
                tl.load(partials_ptr + partial_base + 2, mask=active, other=0.0), axis=0)
            grad_scale_x = tl.sum(
                tl.load(partials_ptr + partial_base + 3, mask=active, other=0.0), axis=0)
            if HAS_SCALE and not SCALE_NORMALIZED:
                batch = particle_row // N_KP
                particle = particle_row % N_KP
                scale_base = batch * stride_sb + particle * stride_sk
                scale_y = tl.load(scale_ptr + scale_base).to(tl.float32)
                scale_x = tl.load(scale_ptr + scale_base + stride_sd).to(tl.float32)
                grad_scale_y *= scale_y * (1.0 - scale_y)
                grad_scale_x *= scale_x * (1.0 - scale_x)
            tl.store(grad_scale_ptr + particle_row * 2, grad_scale_y)
            tl.store(grad_scale_ptr + particle_row * 2 + 1, grad_scale_x)


def _can_use_triton(x, kp, z_scale, patch_size, padding_mode):
    return (triton is not None and x.is_cuda and kp.is_cuda and
            (z_scale is None or z_scale.is_cuda) and
            x.device == kp.device and (z_scale is None or x.device == z_scale.device) and
            x.dtype == torch.float32 and kp.dtype == torch.float32 and
            (z_scale is None or z_scale.dtype == torch.float32) and
            padding_mode == "border" and x.ndim == 4 and kp.ndim == 3 and
            kp.shape[0] == x.shape[0] and kp.shape[-1] == 2 and
            (z_scale is None or z_scale.shape == kp.shape) and
            x.shape[0] > 0 and kp.shape[1] > 0 and
            0 < x.shape[1] <= 16 and x.shape[-2] > 0 and x.shape[-1] > 0 and
            patch_size > 0 and patch_size * patch_size <= 65536)


def _can_use_triton_paste(kp, patches, scale, img_size):
    return (triton is not None and kp.is_cuda and patches.is_cuda and
            (scale is None or scale.is_cuda) and kp.device == patches.device and
            (scale is None or kp.device == scale.device) and
            kp.dtype == torch.float32 and patches.dtype == torch.float32 and
            (scale is None or scale.dtype == torch.float32) and
            kp.ndim == 3 and patches.ndim == 5 and kp.shape[-1] == 2 and
            patches.shape[:2] == kp.shape[:2] and patches.shape[-2] == patches.shape[-1] and
            (scale is None or scale.shape == kp.shape) and
            patches.shape[0] > 0 and kp.shape[1] > 0 and
            0 < patches.shape[2] <= 16 and patches.shape[-1] > 0 and img_size > 0)


_BASE_AXES = {}


def _base_axis(device, patch_size):
    key = (device.type, device.index, patch_size)
    axis = _BASE_AXES.get(key)
    if axis is None:
        if patch_size == 1:
            axis = torch.zeros(1, device=device, dtype=torch.float32)
        else:
            axis = torch.linspace(-1, 1, patch_size, device=device, dtype=torch.float32)
            axis = axis * (patch_size - 1) / patch_size
        _BASE_AXES[key] = axis
    return axis


def _launch_crop_forward(x, kp, normalized_scale, patch_size):
    batch_size, channels, height, width = x.shape
    n_kp = kp.shape[1]
    out = torch.empty(
        (batch_size * n_kp, channels, patch_size, patch_size),
        device=x.device,
        dtype=x.dtype,
    )
    block = 256
    grid = (triton.cdiv(out.numel(), block),)
    base_axis = _base_axis(x.device, patch_size)
    _stn_crop_forward_kernel[grid](
        x, kp, normalized_scale, base_axis, out, out.numel(),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        kp.stride(0), kp.stride(1), kp.stride(2),
        normalized_scale.stride(0), normalized_scale.stride(1), normalized_scale.stride(2),
        N_KP=n_kp, CHANNELS=channels, HEIGHT=height, WIDTH=width,
        PATCH=patch_size, BLOCK=block,
    )
    return out


def _launch_crop_backward(grad_output, x, kp, normalized_scale, patch_size,
                          has_scale, need_x, need_kp, need_scale):
    batch_size, channels, height, width = x.shape
    n_kp = kp.shape[1]
    grad_output = grad_output.contiguous()

    grad_x = None
    if need_x:
        grad_x = torch.zeros_like(x)
        block = 256
        grid = (triton.cdiv(grad_output.numel(), block),)
        _stn_crop_grad_x_kernel[grid](
            grad_output, kp, normalized_scale, _base_axis(x.device, patch_size),
            grad_x, grad_output.numel(),
            grad_output.stride(0), grad_output.stride(1),
            grad_output.stride(2), grad_output.stride(3),
            kp.stride(0), kp.stride(1), kp.stride(2),
            normalized_scale.stride(0), normalized_scale.stride(1),
            normalized_scale.stride(2),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2), grad_x.stride(3),
            N_KP=n_kp, CHANNELS=channels, HEIGHT=height, WIDTH=width,
            PATCH=patch_size, BLOCK=block,
        )

    grad_kp = torch.empty_like(kp, memory_format=torch.contiguous_format) if need_kp else None
    grad_scale = (torch.empty_like(kp, memory_format=torch.contiguous_format)
                  if need_scale else None)
    if need_kp or need_scale:
        spatial_elements = patch_size * patch_size
        block = triton.next_power_of_2(spatial_elements)
        num_warps = 8 if block >= 1024 else 4
        _stn_crop_grad_params_kernel[(batch_size * n_kp,)](
            grad_output, x, kp, normalized_scale, _base_axis(x.device, patch_size),
            grad_kp if grad_kp is not None else kp,
            grad_scale if grad_scale is not None else kp,
            grad_output.stride(0), grad_output.stride(1),
            grad_output.stride(2), grad_output.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            kp.stride(0), kp.stride(1), kp.stride(2),
            normalized_scale.stride(0), normalized_scale.stride(1),
            normalized_scale.stride(2),
            N_KP=n_kp, CHANNELS=channels, HEIGHT=height, WIDTH=width,
            PATCH=patch_size, SPATIAL_ELEMS=spatial_elements,
            HAS_SCALE=has_scale, NEED_KP=need_kp, NEED_SCALE=need_scale,
            BLOCK=block, num_warps=num_warps,
        )
    return grad_x, grad_kp, grad_scale


def _launch_paste_forward(kp, patches, normalized_scale, img_size):
    batch_size, n_kp, channels, patch_size, _ = patches.shape
    out = torch.empty(
        (batch_size, n_kp, channels, img_size, img_size),
        device=patches.device,
        dtype=patches.dtype,
    )
    block = 256
    grid = (triton.cdiv(out.numel(), block),)
    _stn_paste_forward_kernel[grid](
        patches, kp, normalized_scale, _base_axis(patches.device, img_size),
        out, out.numel(),
        patches.stride(0), patches.stride(1), patches.stride(2),
        patches.stride(3), patches.stride(4),
        kp.stride(0), kp.stride(1), kp.stride(2),
        normalized_scale.stride(0), normalized_scale.stride(1),
        normalized_scale.stride(2),
        N_KP=n_kp, CHANNELS=channels, PATCH=patch_size, IMAGE=img_size,
        BLOCK=block,
    )
    return out


def _launch_paste_backward(grad_output, kp, patches, normalized_scale, img_size,
                           has_scale, scale_normalized,
                           need_kp, need_patches, need_scale):
    batch_size, n_kp, channels, patch_size, _ = patches.shape
    grad_output = grad_output.contiguous()
    base_axis = _base_axis(patches.device, img_size)

    grad_patches = None
    if need_patches:
        grad_patches = torch.zeros_like(patches)
        block = 256
        grid = (triton.cdiv(grad_output.numel(), block),)
        _stn_paste_grad_patches_kernel[grid](
            grad_output, kp, normalized_scale, base_axis,
            grad_patches, grad_output.numel(),
            grad_output.stride(0), grad_output.stride(1),
            grad_output.stride(2), grad_output.stride(3), grad_output.stride(4),
            kp.stride(0), kp.stride(1), kp.stride(2),
            normalized_scale.stride(0), normalized_scale.stride(1),
            normalized_scale.stride(2),
            grad_patches.stride(0), grad_patches.stride(1),
            grad_patches.stride(2), grad_patches.stride(3), grad_patches.stride(4),
            N_KP=n_kp, CHANNELS=channels, PATCH=patch_size, IMAGE=img_size,
            BLOCK=block,
        )

    grad_kp = torch.zeros_like(kp, memory_format=torch.contiguous_format) if need_kp else None
    grad_scale = (torch.zeros_like(kp, memory_format=torch.contiguous_format)
                  if need_scale else None)
    if need_kp or need_scale:
        block = 256
        n_tiles = triton.cdiv(img_size * img_size, block)
        partials = torch.empty(
            (batch_size * n_kp, n_tiles, 4), device=patches.device, dtype=torch.float32)
        grid = (batch_size * n_kp * n_tiles,)
        _stn_paste_grad_params_kernel[grid](
            grad_output, patches, kp, normalized_scale, base_axis,
            partials,
            grad_output.stride(0), grad_output.stride(1),
            grad_output.stride(2), grad_output.stride(3), grad_output.stride(4),
            patches.stride(0), patches.stride(1), patches.stride(2),
            patches.stride(3), patches.stride(4),
            kp.stride(0), kp.stride(1), kp.stride(2),
            normalized_scale.stride(0), normalized_scale.stride(1),
            normalized_scale.stride(2),
            N_TILES=n_tiles, N_KP=n_kp, CHANNELS=channels,
            PATCH=patch_size, IMAGE=img_size, BLOCK=block,
        )
        reduction_block = triton.next_power_of_2(n_tiles)
        _stn_paste_reduce_params_kernel[(batch_size * n_kp,)](
            partials, normalized_scale,
            grad_kp if grad_kp is not None else kp,
            grad_scale if grad_scale is not None else kp,
            normalized_scale.stride(0), normalized_scale.stride(1),
            normalized_scale.stride(2),
            N_TILES=n_tiles, N_KP=n_kp, HAS_SCALE=has_scale,
            SCALE_NORMALIZED=scale_normalized, NEED_KP=need_kp,
            NEED_SCALE=need_scale, BLOCK=reduction_block,
        )
    return grad_kp, grad_patches, grad_scale


class _StnCrop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kp, z_scale, patch_size, padding_mode):
        ctx.patch_size = patch_size
        ctx.padding_mode = padding_mode
        ctx.has_scale = z_scale is not None
        if ctx.has_scale:
            normalized_scale = torch.sigmoid(z_scale)
        else:
            normalized_scale = (patch_size / x.shape[-1]) * torch.ones_like(kp)
        ctx.save_for_backward(x, kp, normalized_scale)
        return _launch_crop_forward(x, kp, normalized_scale, patch_size)

    @staticmethod
    def backward(ctx, grad_output):
        x, kp, normalized_scale = ctx.saved_tensors
        need_x, need_kp, need_scale = ctx.needs_input_grad[:3]
        grad_x, grad_kp, grad_scale = _launch_crop_backward(
            grad_output, x, kp, normalized_scale, ctx.patch_size,
            ctx.has_scale, need_x, need_kp, need_scale)
        return grad_x, grad_kp, grad_scale, None, None


class _StnPaste(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kp, patches, scale, img_size, scale_normalized):
        ctx.img_size = img_size
        ctx.has_scale = scale is not None
        ctx.scale_normalized = scale_normalized
        if scale is None:
            normalized_scale = (patches.shape[-1] / img_size) * torch.ones_like(kp)
        elif scale_normalized:
            normalized_scale = scale
        else:
            normalized_scale = torch.sigmoid(scale)
        ctx.save_for_backward(kp, patches, normalized_scale)
        return _launch_paste_forward(kp, patches, normalized_scale, img_size)

    @staticmethod
    def backward(ctx, grad_output):
        kp, patches, normalized_scale = ctx.saved_tensors
        need_kp, need_patches, need_scale = ctx.needs_input_grad[:3]
        grad_kp, grad_patches, grad_scale = _launch_paste_backward(
            grad_output, kp, patches, normalized_scale, ctx.img_size,
            ctx.has_scale, ctx.scale_normalized,
            need_kp, need_patches, need_scale)
        return grad_kp, grad_patches, grad_scale, None, None


def stn_crop(x, kp, patch_size, z_scale=None, padding_mode="border"):
    """Fused crop forward and backward for the supported CUDA fp32 path."""
    if triton is None and x.is_cuda:
        raise RuntimeError(
            "the Triton STN backend was selected for a CUDA tensor, but Triton "
            f"could not be imported: {_TRITON_IMPORT_ERROR}"
        ) from _TRITON_IMPORT_ERROR
    if not _can_use_triton(x, kp, z_scale, patch_size, padding_mode):
        return reference.stn_crop(
            x, kp, patch_size, z_scale=z_scale, padding_mode=padding_mode)
    return _StnCrop.apply(x, kp, z_scale, int(patch_size), padding_mode)


def stn_paste(kp_batch, patches_batch, img_size, scale=None, translation=None,
              scale_normalized=False):
    """Fused inverse-transform paste forward/backward for CUDA fp32."""
    if triton is None and patches_batch.is_cuda:
        raise RuntimeError(
            "the Triton STN backend was selected for a CUDA tensor, but Triton "
            f"could not be imported: {_TRITON_IMPORT_ERROR}"
        ) from _TRITON_IMPORT_ERROR
    if not _can_use_triton_paste(kp_batch, patches_batch, scale, img_size):
        return reference.stn_paste(
            kp_batch, patches_batch, img_size, scale=scale, translation=translation,
            scale_normalized=scale_normalized)
    return _StnPaste.apply(
        kp_batch, patches_batch, scale, int(img_size), bool(scale_normalized))
