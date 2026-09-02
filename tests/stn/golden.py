"""
Golden-fixture format for the STN stage.

A fixture pins, per case, the exact bytes of the forward output and of every
gradient in the case's contract.  The contract is a SHA-256 over the raw
tensor buffer -- bit-exact, 32 bytes on disk regardless of tensor size, which
is what lets full training shapes be pinned without committing gigabytes.

Small tensors are additionally stored in full so a failure can be diffed
element-wise instead of just reported as "hash mismatch".
"""

import hashlib
import subprocess

import torch

__all__ = ["FIXTURE_VERSION", "FULL_STORE_LIMIT", "describe", "compare", "git_commit", "TensorMismatch"]

FIXTURE_VERSION = 1

#: store the full tensor alongside the hash below this element count (~64 KB
#: at float32); above it the hash alone is the contract
FULL_STORE_LIMIT = 16384


class TensorMismatch(Exception):
    pass


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def describe(t):
    """Reduce a tensor to its pinned description."""
    t = t.detach()
    flat = t.reshape(-1).float().cpu()
    rec = {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "sha256": _digest(t),
        "numel": int(t.numel()),
        # cheap human-readable sanity numbers, never used as the pass condition
        "sum_f64": float(flat.double().sum()),
        "absmax": float(flat.abs().max()) if t.numel() else 0.0,
        "excerpt": flat[:32].clone(),
    }
    if t.numel() <= FULL_STORE_LIMIT:
        rec["full"] = t.contiguous().cpu().clone()
    return rec


def compare(label, expected, actual):
    """Raise :class:`TensorMismatch` unless ``actual`` matches the pinned record bit for bit."""
    got = actual.detach()
    if tuple(got.shape) != tuple(expected["shape"]):
        raise TensorMismatch(f"{label}: shape {tuple(got.shape)} != expected {tuple(expected['shape'])}")
    if str(got.dtype) != expected["dtype"]:
        raise TensorMismatch(f"{label}: dtype {got.dtype} != expected {expected['dtype']}")
    if _digest(got) == expected["sha256"]:
        return
    # bit-exactness failed -- produce the most informative message we can
    detail = ""
    if "full" in expected:
        ref = expected["full"].to(got.device)
        diff = (got.float() - ref.float()).abs()
        idx = int(diff.argmax())
        detail = (f"; max|diff|={float(diff.max()):.3e} at flat index {idx} "
                  f"(expected {float(ref.reshape(-1)[idx]):.9g}, got {float(got.reshape(-1)[idx]):.9g}); "
                  f"mismatching elements: {int((diff != 0).sum())}/{got.numel()}")
    else:
        detail = (f"; expected sum={expected['sum_f64']:.9g} absmax={expected['absmax']:.9g}, "
                  f"got sum={float(got.reshape(-1).double().sum()):.9g} "
                  f"absmax={float(got.abs().max()):.9g}")
    raise TensorMismatch(f"{label}: bit-exact mismatch vs baseline{detail}")
