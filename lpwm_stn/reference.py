"""
Isolated reference implementation of the LPWM STN (spatial-transformer) stage.

Stage 1 of the optimization work: pull every spatial-transformer operation out
of ``utils/util_func.py`` and out of the call sites inlined in
``modules/modules.py`` into one module, without changing the algorithm, the
order of operations, the dtypes, or the numerics.

Everything here is numerically identical to :mod:`lpwm_stn._baseline` (the
frozen copy of the pre-optimization code) and is asserted to be so, forward and
backward, by ``tests/stn/test_stn.py``.

Layout
------
Primitives (unchanged, moved verbatim from ``utils/util_func.py``):
    ``affine_grid_sample``      -- ``affine_grid`` + ``grid_sample``, JIT-scripted
    ``spatial_transform``       -- builds the 2x3 theta and samples
    ``create_masks_fast``       -- square particle masks, no scale (no-grad use)
    ``create_masks_with_scale`` -- square particle masks, with scale (no-grad use)

Fused ops (the units a custom kernel will replace; they wrap the exact
sequences that used to be written out at the call sites):
    ``stn_crop``   -- image -> per-particle glimpses  (encoder, inverse=False)
    ``stn_paste``  -- per-particle glimpses -> canvas (decoder, inverse=True)

``stn_crop`` and ``stn_paste`` take the *un-expanded* image / patch tensors,
so a future kernel can fuse away the ``repeat`` that currently materializes a
``[bs * n_kp, ch, H, W]`` copy of the input.  The reference below still performs
that ``repeat`` explicitly, because stage 1 must not change numerics.
"""

import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple

__all__ = [
    "affine_grid_sample",
    "spatial_transform",
    "create_masks_fast",
    "create_masks_with_scale",
    "stn_crop",
    "stn_paste",
]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
@torch.jit.script
def affine_grid_sample(x, theta, out_dims: Tuple[int, int, int, int], mode: str, align_corners: bool = False,
                       padding_mode: str = 'zeros'):
    # construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims), align_corners=align_corners)
    # sample image from grid
    return F.grid_sample(x, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def spatial_transform(image, z_pos, z_scale, out_dims, inverse=False, eps=1e-9, padding_mode="zeros"):
    """
    https://github.com/zhixuan-lin/G-SWM
    spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    ------
    image: [batch_size * n_kp, ch, h, w]
    z_pos: [batch_size * n_kp, 2]
    z_scale: [batch_size * n_kp, 2]
    out_dims: tuple (batch_size * n_kp, ch, h*, w*)
    """
    # 0. validate values range
    # z_pos = z_pos.clamp(-1, 1)
    # z_scale = z_scale.clamp(0, 1)
    # 1. construct 2x3 affine matrix for each datapoint in the batch
    theta = torch.zeros(2, 3, device=image.device).repeat(image.shape[0], 1, 1)
    # set scaling
    theta[:, 0, 0] = z_scale[:, 1] if not inverse else 1 / (z_scale[:, 1] + eps)
    theta[:, 1, 1] = z_scale[:, 0] if not inverse else 1 / (z_scale[:, 0] + eps)

    # set translation
    theta[:, 0, -1] = z_pos[:, 1] if not inverse else - z_pos[:, 1] / (z_scale[:, 1] + eps)
    theta[:, 1, -1] = z_pos[:, 0] if not inverse else - z_pos[:, 0] / (z_scale[:, 0] + eps)
    # construct sampling grid and sample image from grid
    return affine_grid_sample(image, theta, out_dims, mode='bilinear', padding_mode=padding_mode)


def create_masks_fast(center, anchor_s, feature_dim=16, patch_size=None):
    # center: [batch_size, n_kp, 2] in kp_range
    # anchor_h, anchor_w: size of anchor in [0, 1]
    batch_size, n_kp = center.shape[0], center.shape[1]
    if patch_size is None:
        patch_size = np.round(anchor_s * (feature_dim - 1)).astype(int)
    # create white rectangles
    masks = torch.ones(batch_size * n_kp, 1, patch_size, patch_size, device=center.device).float()
    # pad the masks to image size
    pad_size = (feature_dim - patch_size) // 2
    padded_patches_batch = F.pad(masks, pad=[pad_size] * 4)
    # move the masks to be centered around the kp
    delta_t_batch = 0.0 - center
    delta_t_batch = delta_t_batch.reshape(-1, delta_t_batch.shape[-1])  # [bs * n_kp, 2]
    zeros = torch.zeros([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
    ones = torch.ones([delta_t_batch.shape[0], 1], device=delta_t_batch.device).float()
    theta = torch.cat([ones, zeros, delta_t_batch[:, 1].unsqueeze(-1),
                       zeros, ones, delta_t_batch[:, 0].unsqueeze(-1)], dim=-1)
    theta = theta.view(-1, 2, 3)  # [batch_size * n_kp, 2, 3]
    mode = "nearest"
    # mode = 'bilinear'

    trans_padded_patches_batch = affine_grid_sample(padded_patches_batch, theta, padded_patches_batch.shape, mode=mode)

    trans_padded_patches_batch = trans_padded_patches_batch.view(batch_size, n_kp, *padded_patches_batch.shape[1:])
    # [bs, n_kp, 1, feature_dim, feature_dim]
    return trans_padded_patches_batch


def create_masks_with_scale(kp_batch, anchor_s, image_size, scale=None, scale_normalized=False):
    """
    translate patches to be centered around given keypoints
    kp_batch: [bs, n_kp, 2] in [-1, 1]
    patches: [bs, n_kp, ch_patches, patch_size, patch_size]
    scale: None or [bs, n_kp, 2] or [bs, n_kp, 1]
    scale_normalized: False if scale is not in [0, 1]
    :return: translated_padded_patches [bs, n_kp, ch, img_size, img_size]
    """
    patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
    patches_batch = torch.ones(kp_batch.shape[0], kp_batch.shape[1], 1, patch_size, patch_size,
                               device=kp_batch.device, dtype=torch.float)
    batch_size, n_kp, ch_patch, patch_size, _ = patches_batch.shape
    img_size = image_size
    if scale is None:
        z_scale = (patch_size / img_size) * torch.ones_like(kp_batch)
    else:
        # normalize to [0, 1]
        if scale_normalized:
            z_scale = scale
        else:
            z_scale = torch.sigmoid(scale)  # -> [0, 1]
    z_pos = kp_batch.reshape(-1, kp_batch.shape[-1])  # [bs * n_kp, 2]
    z_scale = z_scale.view(-1, z_scale.shape[-1])  # [bs * n_kp, 2]
    patches_batch = patches_batch.reshape(-1, *patches_batch.shape[2:])
    out_dims = (batch_size * n_kp, ch_patch, img_size, img_size)
    trans_patches_batch = spatial_transform(patches_batch, z_pos, z_scale, out_dims, inverse=True)
    trans_padded_patches_batch = trans_patches_batch.view(batch_size, n_kp, *trans_patches_batch.shape[1:])
    # [bs, n_kp, 1, img_size, img_size]
    return trans_padded_patches_batch


# --------------------------------------------------------------------------- #
# fused ops -- the kernel-replaceable units
# --------------------------------------------------------------------------- #
def stn_crop(x, kp, patch_size: int, z_scale=None, padding_mode: str = 'border'):
    """
    Extract one glimpse per particle from a shared image (encoder direction).

    Replaces the block that was written out identically in
    ``ParticleAttributeEncoder.forward`` and ``ParticleFeaturesEncoder.forward``.

    x:          [bs, ch, image_size, image_size]  -- shared across particles
    kp:         [bs, n_kp, 2] in [-1, 1]          -- particle centers (y, x)
    patch_size: side of the extracted glimpse
    z_scale:    None -> isotropic ``patch_size / image_size``;
                otherwise **unnormalized** [bs, n_kp, 2], passed through sigmoid
    :return:    [bs * n_kp, ch, patch_size, patch_size]

    Note: the ``repeat`` below materializes a [bs * n_kp, ch, H, W] copy of the
    input.  It is the single largest allocation in the encoder and the main
    target of the fused kernel; it is kept here because stage 1 preserves
    numerics exactly.
    """
    batch_size = x.shape[0]
    n_kp = kp.shape[1]
    img_size = x.shape[-1]
    x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
    x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
    if z_scale is None:
        z_scale = (patch_size / img_size) * torch.ones_like(kp)
    else:
        # assume unnormalized z_scale
        z_scale = torch.sigmoid(z_scale)
    z_pos = kp.reshape(-1, kp.shape[-1])
    z_scale = z_scale.view(-1, z_scale.shape[-1])
    out_dims = (batch_size * n_kp, x.shape[1], patch_size, patch_size)
    return spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False, padding_mode=padding_mode)


def stn_paste(kp_batch, patches_batch, img_size: int, scale=None, translation=None, scale_normalized: bool = False):
    """
    Place one decoded glimpse per particle onto a shared canvas (decoder direction).

    Replaces ``DLPDecoder.translate_patches``.

    kp_batch:      [bs, n_kp, 2] in [-1, 1]
    patches_batch: [bs, n_kp, ch_patches, patch_size, patch_size]
    img_size:      side of the output canvas (the decoder's ``feature_map_size``)
    scale:         None or [bs, n_kp, 2] or [bs, n_kp, 1]
    translation:   accepted and ignored -- the baseline declares it but never
                   uses it; kept so the signature stays call-compatible
    scale_normalized: False if ``scale`` is not already in [0, 1]
    :return: [bs, n_kp, ch, img_size, img_size]
    """
    batch_size, n_kp, ch_patch, patch_size, _ = patches_batch.shape
    if scale is None:
        z_scale = (patch_size / img_size) * torch.ones_like(kp_batch)
    else:
        # normalize to [0, 1]
        if scale_normalized:
            z_scale = scale
        else:
            z_scale = torch.sigmoid(scale)  # -> [0, 1]
    z_pos = kp_batch.reshape(-1, kp_batch.shape[-1])  # [bs * n_kp, 2]
    z_scale = z_scale.view(-1, z_scale.shape[-1])  # [bs * n_kp, 2]
    patches_batch = patches_batch.reshape(-1, *patches_batch.shape[2:])
    out_dims = (batch_size * n_kp, ch_patch, img_size, img_size)
    trans_patches_batch = spatial_transform(patches_batch, z_pos, z_scale, out_dims, inverse=True)
    trans_padded_patches_batch = trans_patches_batch.view(batch_size, n_kp, *trans_patches_batch.shape[1:])
    # [bs, n_kp, ch, img_size, img_size]
    return trans_padded_patches_batch
