"""
``lpwm_stn`` -- the isolated spatial-transformer stage of LPWM.

Stage 1 of the STN optimization: one module owns every spatial-transformer
operation in the model, behind a backend registry, so that a custom fused
kernel can be dropped in later without touching the model code again.

Public ops
----------
Primitives
    ``affine_grid_sample(x, theta, out_dims, mode, align_corners, padding_mode)``
    ``spatial_transform(image, z_pos, z_scale, out_dims, inverse, eps, padding_mode)``
    ``create_masks_fast(center, anchor_s, feature_dim, patch_size)``
    ``create_masks_with_scale(kp_batch, anchor_s, image_size, scale, scale_normalized)``
Fused
    ``stn_crop(x, kp, patch_size, z_scale, padding_mode)``       encoder direction
    ``stn_paste(kp_batch, patches_batch, img_size, scale, ...)`` decoder direction

Backends
--------
``reference`` (default) is the pre-optimization PyTorch code, bit-for-bit.
Register an accelerated implementation and switch to it with::

    import lpwm_stn
    lpwm_stn.register_backend("triton", my_module)
    lpwm_stn.set_backend("triton")          # process-wide
    with lpwm_stn.use_backend("triton"):    # or scoped
        ...

A backend only needs to implement the ops it accelerates; anything missing
falls back to ``reference``.

Every backend must pass ``tests/stn/test_stn.py``, which checks forward values
and input gradients against the frozen pre-refactor baseline.
"""

from contextlib import contextmanager

from . import reference

__all__ = [
    "affine_grid_sample",
    "spatial_transform",
    "create_masks_fast",
    "create_masks_with_scale",
    "stn_crop",
    "stn_paste",
    "OPS",
    "register_backend",
    "set_backend",
    "get_backend",
    "get_backend_name",
    "list_backends",
    "use_backend",
    "resolve",
]

#: names every backend is checked against and every op the model may call
OPS = (
    "affine_grid_sample",
    "spatial_transform",
    "create_masks_fast",
    "create_masks_with_scale",
    "stn_crop",
    "stn_paste",
)

_BACKENDS = {"reference": reference}
_ACTIVE = "reference"


def register_backend(name, module, overwrite=False):
    """Register an STN implementation under ``name``.

    ``module`` may be any object exposing a subset of :data:`OPS`; ops it does
    not define fall back to ``reference``.
    """
    if name in _BACKENDS and not overwrite:
        raise ValueError(f"STN backend '{name}' is already registered (pass overwrite=True to replace)")
    implemented = [op for op in OPS if getattr(module, op, None) is not None]
    if not implemented:
        raise ValueError(f"STN backend '{name}' implements none of {OPS}")
    _BACKENDS[name] = module
    return module


def list_backends():
    return sorted(_BACKENDS)


def set_backend(name):
    """Select the process-wide STN backend. Returns the previously active name."""
    global _ACTIVE
    if name not in _BACKENDS:
        raise KeyError(f"unknown STN backend '{name}'; registered: {list_backends()}")
    previous, _ACTIVE = _ACTIVE, name
    return previous


def get_backend_name():
    return _ACTIVE


def get_backend():
    return _BACKENDS[_ACTIVE]


@contextmanager
def use_backend(name):
    """Scoped :func:`set_backend`."""
    previous = set_backend(name)
    try:
        yield _BACKENDS[name]
    finally:
        set_backend(previous)


def resolve(op):
    """Return the callable for ``op`` on the active backend, falling back to reference."""
    return getattr(_BACKENDS[_ACTIVE], op, None) or getattr(reference, op)


# --------------------------------------------------------------------------- #
# dispatching front-ends -- these are what the model imports
# --------------------------------------------------------------------------- #
def affine_grid_sample(x, theta, out_dims, mode, align_corners=False, padding_mode='zeros'):
    return resolve("affine_grid_sample")(x, theta, out_dims, mode, align_corners, padding_mode)


def spatial_transform(image, z_pos, z_scale, out_dims, inverse=False, eps=1e-9, padding_mode="zeros"):
    return resolve("spatial_transform")(image, z_pos, z_scale, out_dims, inverse=inverse, eps=eps,
                                        padding_mode=padding_mode)


def create_masks_fast(center, anchor_s, feature_dim=16, patch_size=None):
    return resolve("create_masks_fast")(center, anchor_s, feature_dim=feature_dim, patch_size=patch_size)


def create_masks_with_scale(kp_batch, anchor_s, image_size, scale=None, scale_normalized=False):
    return resolve("create_masks_with_scale")(kp_batch, anchor_s, image_size, scale=scale,
                                              scale_normalized=scale_normalized)


def stn_crop(x, kp, patch_size, z_scale=None, padding_mode='border'):
    return resolve("stn_crop")(x, kp, patch_size, z_scale=z_scale, padding_mode=padding_mode)


def stn_paste(kp_batch, patches_batch, img_size, scale=None, translation=None, scale_normalized=False):
    return resolve("stn_paste")(kp_batch, patches_batch, img_size, scale=scale, translation=translation,
                                scale_normalized=scale_normalized)
