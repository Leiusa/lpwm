"""
Build a DLP/LPWM model from a repo config without touching repo source.

``train_lpwm.py`` unpacks ~60 locals out of the config and passes them to
``DLP(...)``. Replicating that by hand would drift; instead the constructor's
own signature is the source of truth and config keys are matched to it by name,
with the handful of genuine renames listed in :data:`RENAMES`.
"""

import inspect
import json

from models import DLP

__all__ = ["RENAMES", "build"]

#: config keys whose name differs from the DLP constructor parameter
RENAMES = {
    "cdim": "ch",
    "n_static_frames": "num_static_frames",
    "ctx_dist": "context_dist",
}

#: constructor params that must be tuples, but arrive from JSON as lists
_TUPLE_PARAMS = ("obj_ch_mult", "obj_ch_mult_prior", "bg_ch_mult")


def build(config_path, device="cuda"):
    """Return ``(model, cfg, kwargs)``. Params absent from the config keep their defaults."""
    with open(config_path) as f:
        cfg = json.load(f)

    params = [p for p in inspect.signature(DLP.__init__).parameters if p != "self"]
    kwargs = {}
    for p in params:
        if p in cfg:
            kwargs[p] = cfg[p]
        elif p in RENAMES and RENAMES[p] in cfg:
            kwargs[p] = cfg[RENAMES[p]]
    for k in _TUPLE_PARAMS:
        if isinstance(kwargs.get(k), list):
            kwargs[k] = tuple(kwargs[k])

    return DLP(**kwargs).to(device), cfg, kwargs


def sequence_length(cfg):
    """Frames the loss expects: an initial frame plus ``timestep_horizon`` predicted steps.

    ``calc_dyn_elbo`` reshapes the reconstruction to ``[bs, timestep_horizon + 1, -1]``,
    so feeding exactly ``timestep_horizon`` frames raises a shape error.
    """
    return cfg["timestep_horizon"] + 1
