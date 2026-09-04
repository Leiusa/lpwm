"""
The shared case list for the STN stage.

One definition of "what to run" used by three consumers:
  * ``tests/stn/gen_golden.py``  -- records the frozen baseline's answers
  * ``tests/stn/test_stn.py``    -- checks an implementation against them
  * the later GPU benchmark      -- will time the same shapes after cluster access

A case is a named (op, workload, options) triple plus the inputs it needs.
Inputs are generated deterministically from ``lpwm_stn.workloads`` so the
baseline, the reference and any future kernel all see identical bytes.
"""

import zlib

import torch

from lpwm_stn import workloads

__all__ = ["Case", "CORRECTNESS_CASES", "BENCH_CASES", "PRIOR_BENCH_CASES", "build_cases", "by_name"]

#: shapes small enough to hash, diff and store; ``balls_1t`` is a real config
#: (12 particles, 64px, 16px glimpses) with batch and horizon cut to 1
_CORRECTNESS_WORKLOADS = ("tiny", "balls_1t")

#: full-size retained-particle shapes -- correctness is verified on the small
#: ones, while these are available for opt-in reference checks and profiling
_BENCH_WORKLOADS = ("bair", "bair64", "obj3d128", "balls")

# Actual ParticleAttributeEncoder proposal counts. These are benchmark-only;
# the retained-particle cases already pin the operation's numerical contract.
_PRIOR_BENCH_WORKLOADS = ("bair_prior", "bair64_prior", "obj3d128_prior", "balls_prior")

workloads.WORKLOADS["balls_1t"] = workloads.get("balls").scaled(batch_size=1, timestep_horizon=1)
workloads.WORKLOADS["balls_1t"].name = "balls_1t"


class Case:
    """One STN invocation: how to build its inputs and how to call it.

    ``grad_wrt`` names the input tensors whose gradients are part of the
    contract.  Empty means the op is only ever used under ``no_grad`` in the
    model (the mask builders), so only the forward value is pinned.
    """

    #: ops whose theta is built inside ``spatial_transform``, which hardcodes
    #: ``torch.zeros(2, 3, device=...)`` -- float32, whatever the input dtype.
    #: Consequences, both inherited from the baseline and deliberately kept:
    #:   * float64 ``gradcheck`` cannot run on them (grid_sample would get a
    #:     float32 grid and a float64 input, and raises)
    #:   * under AMP the sampling grid stays fp32 even when activations are not
    #: A replacement kernel must reproduce this, not "fix" it, or the numerics
    #: move.  Revisit it as an explicit, separately-validated change.
    _FLOAT32_THETA_OPS = ("spatial_transform", "stn_crop", "stn_paste",
                          "create_masks_fast", "create_masks_with_scale")

    def __init__(self, name, op, workload, make_inputs, call, grad_wrt=()):
        self.name = name
        self.op = op
        self.workload = workload
        self._make_inputs = make_inputs
        self._call = call
        self.grad_wrt = tuple(grad_wrt)

    @property
    def float64_ok(self):
        """Whether this case can be run in float64 (i.e. is gradcheck-able)."""
        return self.op not in self._FLOAT32_THETA_OPS

    @property
    def needs_grad(self):
        return bool(self.grad_wrt)

    def inputs(self, device="cpu", dtype=torch.float32, requires_grad=None):
        if requires_grad is None:
            requires_grad = self.needs_grad
        return self._make_inputs(self.workload, device=device, dtype=dtype, requires_grad=requires_grad)

    def run(self, impl, ins):
        """Call ``impl``'s version of this op on ``ins``. ``impl`` is a module."""
        return self._call(impl, ins)

    def cotangent(self, out, seed=0x5747):
        """A fixed random upstream gradient, so backward is exercised on all elements.

        Seeded off ``crc32(name)`` rather than ``hash(name)``: str hashing is
        salted per process, which would silently hand the fixture generator and
        the test different cotangents.
        """
        g = torch.Generator(device=out.device)
        g.manual_seed(seed + (zlib.crc32(self.name.encode()) & 0xFFFF))
        return torch.rand(out.shape, generator=g, device=out.device, dtype=out.dtype)

    def __repr__(self):
        return f"Case({self.name})"


# --------------------------------------------------------------------------- #
# input builders  (thin adapters over lpwm_stn.workloads)
# --------------------------------------------------------------------------- #
def _crop_inputs(with_scale):
    def build(wl, device, dtype, requires_grad):
        return workloads.make_crop_inputs(wl, device=device, dtype=dtype,
                                          requires_grad=requires_grad, with_scale=with_scale)
    return build


def _paste_inputs(with_scale):
    def build(wl, device, dtype, requires_grad):
        return workloads.make_paste_inputs(wl, device=device, dtype=dtype,
                                           requires_grad=requires_grad, with_scale=with_scale)
    return build


def _mask_inputs(wl, device, dtype, requires_grad):
    return workloads.make_mask_inputs(wl, device=device, dtype=dtype)


def _raw_st_inputs(inverse):
    """Flattened [bs*T*n_kp, ...] inputs for the bare ``spatial_transform``."""

    def build(wl, device, dtype, requires_grad):
        g = torch.Generator(device=device)
        g.manual_seed(7 if inverse else 3)
        rows = wl.flat_batch * wl.n_kp
        if inverse:
            # glimpse -> canvas
            image = torch.rand(rows, 4, wl.dec_patch_size, wl.dec_patch_size,
                               generator=g, device=device, dtype=dtype)
            out_dims = (rows, 4, wl.image_size, wl.image_size)
            padding_mode = "zeros"
        else:
            # canvas -> glimpse
            image = torch.rand(rows, wl.ch, wl.image_size, wl.image_size,
                               generator=g, device=device, dtype=dtype)
            out_dims = (rows, wl.ch, wl.patch_size, wl.patch_size)
            padding_mode = "border"
        z_pos = 2.0 * torch.rand(rows, 2, generator=g, device=device, dtype=dtype) - 1.0
        # strictly positive scale: `inverse` divides by it
        z_scale = 0.05 + 0.9 * torch.rand(rows, 2, generator=g, device=device, dtype=dtype)
        if requires_grad:
            image.requires_grad_(True)
            z_pos.requires_grad_(True)
            z_scale.requires_grad_(True)
        return {"image": image, "z_pos": z_pos, "z_scale": z_scale, "out_dims": out_dims,
                "inverse": inverse, "padding_mode": padding_mode}

    return build


def _ags_inputs(mode):
    def build(wl, device, dtype, requires_grad):
        g = torch.Generator(device=device)
        g.manual_seed(11)
        rows = wl.flat_batch * wl.n_kp
        x = torch.rand(rows, 1, wl.image_size, wl.image_size, generator=g, device=device, dtype=dtype)
        theta = torch.zeros(rows, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, :, 2] = 2.0 * torch.rand(rows, 2, generator=g, device=device, dtype=dtype) - 1.0
        if requires_grad:
            x.requires_grad_(True)
            if mode != "nearest":
                theta.requires_grad_(True)
        return {"x": x, "theta": theta, "out_dims": tuple(x.shape), "mode": mode}

    return build


# --------------------------------------------------------------------------- #
# call adapters
# --------------------------------------------------------------------------- #
def _call_crop(impl, ins):
    return impl.stn_crop(ins["x"], ins["kp"], ins["patch_size"], z_scale=ins["z_scale"],
                         padding_mode=ins["padding_mode"])


def _call_paste(impl, ins):
    return impl.stn_paste(ins["kp_batch"], ins["patches_batch"], ins["img_size"], scale=ins["scale"],
                          scale_normalized=ins["scale_normalized"])


def _call_masks_fast(impl, ins):
    return impl.create_masks_fast(ins["center"], anchor_s=ins["anchor_s"], feature_dim=ins["feature_dim"])


def _call_masks_scale(impl, ins):
    return impl.create_masks_with_scale(ins["center"], anchor_s=ins["anchor_s"], image_size=ins["feature_dim"],
                                        scale=ins["scale"])


def _call_spatial_transform(impl, ins):
    return impl.spatial_transform(ins["image"], ins["z_pos"], ins["z_scale"], ins["out_dims"],
                                  inverse=ins["inverse"], padding_mode=ins["padding_mode"])


def _call_ags(impl, ins):
    return impl.affine_grid_sample(ins["x"], ins["theta"], ins["out_dims"], ins["mode"])


# --------------------------------------------------------------------------- #
# the case list
# --------------------------------------------------------------------------- #
_SPECS = (
    # (suffix, op, input builder, call adapter, grad_wrt)
    ("crop.scale", "stn_crop", _crop_inputs(True), _call_crop, ("x", "kp", "z_scale")),
    ("crop.noscale", "stn_crop", _crop_inputs(False), _call_crop, ("x", "kp")),
    ("paste.scale", "stn_paste", _paste_inputs(True), _call_paste, ("patches_batch", "kp_batch", "scale")),
    ("paste.noscale", "stn_paste", _paste_inputs(False), _call_paste, ("patches_batch", "kp_batch")),
    ("st.forward", "spatial_transform", _raw_st_inputs(False), _call_spatial_transform,
     ("image", "z_pos", "z_scale")),
    ("st.inverse", "spatial_transform", _raw_st_inputs(True), _call_spatial_transform,
     ("image", "z_pos", "z_scale")),
    ("ags.bilinear", "affine_grid_sample", _ags_inputs("bilinear"), _call_ags, ("x", "theta")),
    # nearest is used only by create_masks_fast, where theta carries no gradient
    ("ags.nearest", "affine_grid_sample", _ags_inputs("nearest"), _call_ags, ("x",)),
    # both mask builders run under no_grad in DLPEncoder.get_bg_mask_from_particle_glimpses
    ("masks.fast", "create_masks_fast", _mask_inputs, _call_masks_fast, ()),
    ("masks.scale", "create_masks_with_scale", _mask_inputs, _call_masks_scale, ()),
)


def build_cases(workload_names):
    cases = []
    for wl_name in workload_names:
        wl = workloads.get(wl_name)
        for suffix, op, make_inputs, call, grad_wrt in _SPECS:
            cases.append(Case(f"{wl_name}/{suffix}", op, wl, make_inputs, call, grad_wrt))
    return cases


CORRECTNESS_CASES = build_cases(_CORRECTNESS_WORKLOADS)
BENCH_CASES = build_cases(_BENCH_WORKLOADS)
PRIOR_BENCH_CASES = [case for case in build_cases(_PRIOR_BENCH_WORKLOADS)
                     if case.op == "stn_crop"]


def by_name(cases):
    return {c.name: c for c in cases}
