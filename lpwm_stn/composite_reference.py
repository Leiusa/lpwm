"""
PyTorch oracle for the paste + alpha/depth composite + particle reduction.

A verbatim transcription of the chain the model runs in
``DLPDecoder.get_objects_alpha_rgb`` / ``get_objects_alpha_rgb_with_depth``
(``modules/modules.py``), lifted into a free function so fixtures can be
generated from it and a fused kernel compared against it.

The only edits are mechanical: ``self.feature_map_size`` becomes a parameter,
and the particle-decoder call is left to the caller so this unit begins exactly
at the paste.

Two properties are deliberate and load-bearing:

* **PyTorch only, Triton never.** This module is the oracle a kernel is judged
  against, so it must not import or depend on any accelerated backend.
* **The reference paste is called explicitly**, via ``reference.stn_paste``,
  not through the ``lpwm_stn`` dispatcher. Going through the dispatcher would
  make the oracle's output depend on whichever backend happened to be active
  when fixtures were generated, which would silently defeat the comparison.

Do not change the compositing order, the reduction order, the placement of
``eps``, or any dtype: those are the contract, not implementation detail.
"""

import torch

from . import reference

__all__ = ["composite_reference", "EPS"]

#: matches the default in DLPDecoder.get_objects_alpha_rgb_with_depth. Applied to
#: the SUM before the division, never to the individual terms.
EPS = 1e-5


def composite_reference(dec_objects, z_kp, z_scale, obj_on, z_depth, img_size,
                        eps=EPS, return_alpha_masks=True, translation=None,
                        scale_normalized=False):
    """paste -> split -> depth-weighted alpha composite -> reduce over particles.

    dec_objects : [bs, n_kp, 4, p, p]  RGBA glimpses from the particle decoder
    z_kp        : [bs, n_kp, 2]        particle centers in [-1, 1]
    z_scale     : [bs, n_kp, 2] or None (None -> patch_size / img_size, as in stn_paste)
    obj_on      : [bs, n_kp]           per-particle on/off gate
    z_depth     : [bs, n_kp, 1]        inferred depth
    img_size    : int                  canvas side (the decoder's feature_map_size)

    :return: ``(alpha_masks | None, alpha_mask, dec_objects_trans)`` --
             the per-particle stack (or None), the reduced background mask
             ``[bs, 1, H, W]``, and the composited RGB ``[bs, 3, H, W]``.
    """
    # paste: [bs, n_kp, 4, p, p] -> [bs, n_kp, 4, H, W]
    dec_objects_trans = reference.stn_paste(z_kp, dec_objects, img_size, scale=z_scale,
                                            translation=translation,
                                            scale_normalized=scale_normalized)
    a_obj, rgb_obj = torch.split(dec_objects_trans, [1, dec_objects_trans.shape[2] - 1], dim=2)

    # composite, verbatim from get_objects_alpha_rgb_with_depth
    a_obj = obj_on[:, :, None, None, None] * a_obj
    rgba_obj = a_obj * rgb_obj
    importance_map = a_obj * torch.sigmoid(-z_depth[:, :, :, None, None])
    importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + eps)
    out_rgb = (rgba_obj * importance_map).sum(dim=1)
    alpha_mask = 1.0 - (importance_map * a_obj).sum(dim=1)
    masks = importance_map * a_obj if return_alpha_masks else None
    return masks, alpha_mask, out_rgb
