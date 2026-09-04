"""
Canonical STN workloads for LPWM, derived from the shipped configs.

The STN stage is driven by a handful of numbers -- batch size, rollout length,
particle count, image size, glimpse size and channel count -- and those come
straight out of ``configs/*.json``.  This module turns them into named specs
plus deterministic input generators, so the reference tests, the golden
fixtures and the benchmarks all agree on what "the real workload" is.

These presets use ``n_kp_enc`` and therefore represent the retained-particle
crop/paste paths. The attribute crop can instead run over ``n_kp_prior``;
stage 2 should record that call-site shape from a representative GPU profile
before publishing baseline numbers.
"""

import json
import os

import numpy as np
import torch

__all__ = ["StnWorkload", "WORKLOADS", "get", "names", "from_config", "make_crop_inputs",
           "make_paste_inputs", "make_mask_inputs"]

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


class StnWorkload:
    """One concrete STN problem size.

    batch_size / timestep_horizon / n_kp / image_size / ch come from a config;
    ``patch_size`` is the glimpse side used by the encoder-side crop, computed
    the same way the model does it: ``round(anchor_s * (image_size - 1))``.
    """

    def __init__(self, name, batch_size, timestep_horizon, n_kp, image_size, ch, anchor_s,
                 patch_size=None, dec_patch_size=None):
        self.name = name
        self.batch_size = int(batch_size)
        self.timestep_horizon = int(timestep_horizon)
        self.n_kp = int(n_kp)
        self.image_size = int(image_size)
        self.ch = int(ch)
        self.anchor_s = float(anchor_s)
        # how modules.ParticleAttributeEncoder / ParticleFeaturesEncoder size a glimpse
        self.patch_size = int(np.round(anchor_s * (image_size - 1))) if patch_size is None else int(patch_size)
        # decoded RGBA glimpse side fed back to the canvas by DLPDecoder
        self.dec_patch_size = self.patch_size if dec_patch_size is None else int(dec_patch_size)

    @property
    def stn_batch(self):
        """Rows the sampler actually sees: bs * T flattened, times one per particle."""
        return self.batch_size * self.timestep_horizon * self.n_kp

    @property
    def flat_batch(self):
        """bs * T -- the image batch handed to the encoder."""
        return self.batch_size * self.timestep_horizon

    def scaled(self, batch_size=None, timestep_horizon=None):
        """A copy with a smaller batch / horizon, for tests that must stay cheap."""
        return StnWorkload(
            self.name,
            self.batch_size if batch_size is None else batch_size,
            self.timestep_horizon if timestep_horizon is None else timestep_horizon,
            self.n_kp, self.image_size, self.ch, self.anchor_s,
            patch_size=self.patch_size, dec_patch_size=self.dec_patch_size,
        )

    def __repr__(self):
        return (f"StnWorkload({self.name}: bs={self.batch_size} T={self.timestep_horizon} "
                f"n_kp={self.n_kp} img={self.image_size} ch={self.ch} patch={self.patch_size} "
                f"stn_batch={self.stn_batch})")


def from_config(name, config_dir=_CONFIG_DIR, n_kp_field="n_kp_enc"):
    """Build a workload from ``configs/<name>.json``."""
    with open(os.path.join(config_dir, f"{name}.json")) as f:
        cfg = json.load(f)
    return StnWorkload(
        name=name,
        batch_size=cfg["batch_size"],
        timestep_horizon=cfg["timestep_horizon"],
        n_kp=cfg[n_kp_field],
        image_size=cfg["image_size"],
        ch=cfg.get("ch", 3),
        anchor_s=cfg["anchor_s"],
    )


#: the four shapes worth tracking: the two extremes of particle count and the
#: two glimpse sizes.  Built eagerly so a missing/renamed config fails loudly.
WORKLOADS = {name: from_config(name) for name in ("bair", "bair64", "obj3d128", "balls")}

# The attribute encoder crops every retained prior proposal before filtering.
# Keep those larger, real call-site shapes available to the benchmark without
# making the golden suite twice as expensive.
for _name in ("bair", "bair64", "obj3d128", "balls"):
    _prior = from_config(_name, n_kp_field="n_kp_prior")
    _prior.name = f"{_name}_prior"
    WORKLOADS[_prior.name] = _prior

#: a deliberately tiny shape, so correctness tests and fixtures stay fast and
#: the fixture file stays small enough to commit
WORKLOADS["tiny"] = StnWorkload("tiny", batch_size=2, timestep_horizon=2, n_kp=5, image_size=32,
                                ch=3, anchor_s=0.25)


def names():
    return sorted(WORKLOADS)


def get(name):
    if name not in WORKLOADS:
        raise KeyError(f"unknown STN workload '{name}'; known: {names()}")
    return WORKLOADS[name]


# --------------------------------------------------------------------------- #
# deterministic input generation
# --------------------------------------------------------------------------- #
def _gen(shape, generator, device, dtype):
    return torch.rand(*shape, generator=generator, device=device, dtype=dtype)


def _generator(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def make_crop_inputs(wl, seed=0, device="cpu", dtype=torch.float32, requires_grad=True, with_scale=True):
    """Inputs for the encoder-direction crop.

    Returns ``dict(x, kp, z_scale, patch_size, padding_mode)`` where
    ``x`` is [bs*T, ch, H, W] and ``kp``/``z_scale`` are [bs*T, n_kp, 2].
    ``kp`` lands in [-1, 1] (particle centers) and ``z_scale`` is left
    *unnormalized* -- the op pushes it through a sigmoid, as the model does.
    """
    g = _generator(seed, device)
    x = _gen((wl.flat_batch, wl.ch, wl.image_size, wl.image_size), g, device, dtype)
    kp = 2.0 * _gen((wl.flat_batch, wl.n_kp, 2), g, device, dtype) - 1.0
    # unnormalized scale, centered so sigmoid lands near the anchor size
    z_scale = None
    if with_scale:
        z_scale = 4.0 * _gen((wl.flat_batch, wl.n_kp, 2), g, device, dtype) - 2.0
    if requires_grad:
        x.requires_grad_(True)
        kp.requires_grad_(True)
        if z_scale is not None:
            z_scale.requires_grad_(True)
    return {"x": x, "kp": kp, "z_scale": z_scale, "patch_size": wl.patch_size, "padding_mode": "border"}


def make_paste_inputs(wl, seed=1, device="cpu", dtype=torch.float32, requires_grad=True, with_scale=True):
    """Inputs for the decoder-direction paste.

    Returns ``dict(kp_batch, patches_batch, img_size, scale, scale_normalized)``
    where ``patches_batch`` is [bs*T, n_kp, 4, p, p] (RGBA glimpses, as the
    particle decoder emits) and the canvas is ``image_size``.
    """
    g = _generator(seed, device)
    patches = _gen((wl.flat_batch, wl.n_kp, 4, wl.dec_patch_size, wl.dec_patch_size), g, device, dtype)
    kp = 2.0 * _gen((wl.flat_batch, wl.n_kp, 2), g, device, dtype) - 1.0
    scale = None
    if with_scale:
        scale = 4.0 * _gen((wl.flat_batch, wl.n_kp, 2), g, device, dtype) - 2.0
    if requires_grad:
        patches.requires_grad_(True)
        kp.requires_grad_(True)
        if scale is not None:
            scale.requires_grad_(True)
    return {"kp_batch": kp, "patches_batch": patches, "img_size": wl.image_size, "scale": scale,
            "scale_normalized": False}


def make_mask_inputs(wl, seed=2, device="cpu", dtype=torch.float32, mask_size=None, with_scale=True):
    """Inputs for the two mask builders (used under ``no_grad`` in the model)."""
    g = _generator(seed, device)
    kp = 2.0 * _gen((wl.flat_batch, wl.n_kp, 2), g, device, dtype) - 1.0
    scale = 4.0 * _gen((wl.flat_batch, wl.n_kp, 2), g, device, dtype) - 2.0 if with_scale else None
    return {"center": kp, "anchor_s": wl.anchor_s,
            "feature_dim": wl.image_size if mask_size is None else int(mask_size), "scale": scale}
