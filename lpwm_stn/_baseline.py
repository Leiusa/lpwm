"""
Frozen baseline snapshot of the LPWM spatial-transformer (STN) primitives.

This file is a VERBATIM copy of the four STN functions as they existed at the
commit that started the optimization work (``4cf53c4``, extracted from
``utils/util_func.py`` with ``git show``).  It exists for exactly one reason:
to be the immovable oracle that every future implementation -- the isolated
reference in :mod:`lpwm_stn.reference`, and later the custom fused kernels --
is compared against in forward and backward.

DO NOT EDIT, refactor, reformat or "improve" anything below this header.
Nothing in the training or inference path imports this module; it is only
loaded by the equivalence tests.
"""
# imports (kept minimal, matching the originals' dependencies)
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


# --------------------------------------------------------------------------- #
# verbatim from utils/util_func.py @ 4cf53c4
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
# verbatim from modules/modules.py @ 4cf53c4 -- the two STN call sites, lifted
# out of their methods.  The only edits are the mechanical ones needed to make
# them free functions:
#   ParticleAttributeEncoder.forward   (L2828-2833) / ParticleFeaturesEncoder.forward
#     (L3011-3027, identical from `x_repeated` onward)  ->  self.patch_size -> patch_size
#   DLPDecoder.translate_patches       (L5302-5330)      ->  self.feature_map_size -> img_size
# --------------------------------------------------------------------------- #
def stn_crop(x, kp, patch_size, z_scale=None, padding_mode='border'):
    # x: [bs, ch, image_size, image_size]
    # kp: [bs, n_kp, 2] in [-1, 1]
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
    cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False,
                                        padding_mode=padding_mode)
    # [batch_size * n_kp, ch, patch_size, patch_size]
    return cropped_objects


def stn_paste(kp_batch, patches_batch, img_size, scale=None, translation=None, scale_normalized=False):
    """
    translate patches to be centered around given keypoints
    kp_batch: [bs, n_kp, 2] in [-1, 1]
    patches: [bs, n_kp, ch_patches, patch_size, patch_size]
    scale: None or [bs, n_kp, 2] or [bs, n_kp, 1]
    translation: None or [bs, n_kp, 2] or [bs, n_kp, 1] (delta from kp)
    scale_normalized: False if scale is not in [0, 1]
    :return: translated_padded_patches [bs, n_kp, ch, img_size, img_size]
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
