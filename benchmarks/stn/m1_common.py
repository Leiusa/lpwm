"""
Shared helpers of the milestone-1 controlled quality study (A vs D static DLP on RTX A6000, six seeds).

* verify_hardware(): the allocated GPU must be exactly the study's expected model (M1_EXPECTED_GPU, default "NVIDIA RTX A6000"; never the Ada variant or a 3080 Ti) on a node
  that is not grogu-4-13; records node, GPU name/UUID, memory, driver. Raises HardwareMismatch otherwise.
* set_agreed_environment() / effective_settings(): the training environment agreed for this study is set EXPLICITLY and
  read back: fp32, no AMP, cudnn.benchmark False, cudnn.deterministic True, cuDNN TF32 allowed, matmul TF32 off.
* install_evidence(): runtime evidence of what actually executed. lpwm_stn.resolve() is wrapped so every STN op call is
  counted together with the implementation function that was really selected; composite_fused calls are counted too.
  Reading config fields is not evidence; these counters are.
"""
import hashlib
import json
import os
import socket
import subprocess
import time
from collections import defaultdict

EXPECTED_GPU = os.environ.get("M1_EXPECTED_GPU", "NVIDIA RTX A6000")   # the ONE GPU model of a study; the job scripts export it
FORBIDDEN_NODES = ("grogu-4-13",)
MEM_MB_RANGES = {"NVIDIA RTX A6000": (45000, 52000),          # 49,140 MiB as seen by nvidia-smi
                 "NVIDIA GeForce RTX 3090": (23000, 25500)}   # 24,576 MiB


class HardwareMismatch(RuntimeError):
    pass


def sha256_file(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    return d if n is None else d[:n]


def _smi(fields):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise HardwareMismatch(f"nvidia-smi failed: {out.stderr.strip()}")
    return [[c.strip() for c in line.split(",")] for line in out.stdout.strip().splitlines() if line.strip()]


def check_gpu_facts(node, smi_rows, torch_name, expected=EXPECTED_GPU):
    """Pure decision logic (unit-testable): returns a list of problems, empty if the hardware is acceptable."""
    problems = []
    if node.split(".")[0] in FORBIDDEN_NODES:
        problems.append(f"node {node} is excluded")
    if len(smi_rows) != 1:
        problems.append(f"expected exactly one visible GPU, nvidia-smi shows {len(smi_rows)}")
    for row in smi_rows:
        name, uuid, mem = row[0], row[1], row[2]
        if name != expected:
            problems.append(f"GPU model {name!r} is not {expected!r}")
        if not uuid.startswith("GPU-"):
            problems.append(f"unexpected UUID {uuid!r}")
        try:
            rng = MEM_MB_RANGES.get(expected)
            if rng is None:
                problems.append(f"no memory range registered for expected GPU {expected!r}")
            elif not (rng[0] <= float(mem) <= rng[1]):
                problems.append(f"GPU memory {mem} MiB outside {rng}")
        except ValueError:
            problems.append(f"unreadable GPU memory {mem!r}")
    if torch_name is not None and torch_name != expected:
        problems.append(f"torch reports {torch_name!r}, expected {expected!r}")
    return problems


def verify_hardware(expected=EXPECTED_GPU):
    import torch
    node = socket.gethostname()
    rows = _smi("name,uuid,memory.total,driver_version")
    tname = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    problems = check_gpu_facts(node, [r[:3] for r in rows], tname, expected)
    rec = {"expected_gpu": expected, "node": node, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "torch_device_count": torch.cuda.device_count(), "torch_gpu_name": tname, "nvidia_smi": [dict(zip(("name", "uuid", "memory_mib", "driver"), r)) for r in rows],
           "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None, "verified": not problems, "problems": problems,
           "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if torch.cuda.device_count() != 1:
        rec["problems"].append(f"torch sees {torch.cuda.device_count()} devices")
        rec["verified"] = False
    if not rec["verified"]:
        raise HardwareMismatch("; ".join(rec["problems"]) + " | " + json.dumps(rec))
    return rec


def gpu_state():
    try:
        r = _smi("clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu")[0]
        return dict(zip(("clock_sm_mhz", "clock_max_mhz", "power_w", "temp_c", "util_pct"), r))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def set_agreed_environment():
    """The agreed training environment (unchanged from the earlier study): fp32, no AMP, deterministic cuDNN without
    autotuning, cuDNN TF32 allowed (torch default), matmul TF32 off (torch default). Set explicitly, then verified."""
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    return effective_settings()


def effective_settings():
    import numpy
    import torch
    try:
        import triton
        tv = triton.__version__
    except Exception:  # noqa: BLE001
        tv = None
    b = torch.backends
    return {"python": __import__("sys").version.split()[0], "torch": torch.__version__, "triton": tv, "numpy": numpy.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn_version": b.cudnn.version(), "default_dtype": str(torch.get_default_dtype()), "cudnn.enabled": b.cudnn.enabled, "cudnn.benchmark": b.cudnn.benchmark,
            "cudnn.deterministic": b.cudnn.deterministic, "cudnn.allow_tf32": b.cudnn.allow_tf32, "matmul.allow_tf32": b.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "amp_autocast_used": False, "env": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING", "TORCH_HOME")}}


AGREED = {"default_dtype": "torch.float32", "cudnn.benchmark": False, "cudnn.deterministic": True, "cudnn.allow_tf32": True, "matmul.allow_tf32": False,
          "float32_matmul_precision": "highest", "cudnn.enabled": True}


def assert_agreed_environment(settings=None):
    s = settings or effective_settings()
    bad = {k: (s.get(k), v) for k, v in AGREED.items() if s.get(k) != v}
    if bad:
        raise RuntimeError(f"environment differs from the agreed settings (effective, agreed): {bad}")
    return s


class Evidence:
    """Runtime counters of which STN implementation / fusion path executed."""

    def __init__(self):
        self.c = defaultdict(int)

    def reset(self):
        self.c = defaultdict(int)

    def snapshot(self):
        return dict(self.c)

    def total(self, prefix):
        return sum(v for k, v in self.c.items() if k.startswith(prefix))

    def only_impl(self, op, needle):
        """True if op ran at least once and every implementation that ran contains `needle` in its name."""
        keys = [k for k in self.c if k.startswith(op + "|")]
        return bool(keys) and all(needle in k for k in keys)


def install_evidence(ev):
    """Wrap lpwm_stn.resolve (records op + the implementation function actually selected) and modules.modules.composite_fused."""
    import lpwm_stn
    import modules.modules as M
    if getattr(lpwm_stn, "_m1_evidence_installed", False):
        lpwm_stn._m1_evidence.__init__()
        return
    orig_resolve = lpwm_stn.resolve

    def resolve(op):
        fn = orig_resolve(op)
        ev.c[f"{op}|{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"] += 1
        return fn
    lpwm_stn.resolve = resolve
    orig_cf = M.composite_fused

    def composite_fused(*a, **k):
        ev.c["composite_fused|call"] += 1
        return orig_cf(*a, **k)
    M.composite_fused = composite_fused
    lpwm_stn._m1_evidence_installed = True
    lpwm_stn._m1_evidence = ev


def module_counters(model):
    dec = model.decoder_module
    return {"fused_composite_calls": dec.fused_composite_calls, "channels_last_calls": getattr(dec.particle_dec, "channels_last_calls", None),
            "fused_composite_flag": dec.fused_composite, "particle_dec_channels_last_flag": getattr(dec, "particle_dec_channels_last", None)}


def code_hashes(repo):
    files = ("models.py", "modules/modules.py", "train_dlp.py", "lpwm_stn/composite_autograd.py", "lpwm_stn/composite_triton.py", "lpwm_stn/composite_backward.py",
             "benchmarks/stn/train_dlp_compare.py", "benchmarks/stn/m1_common.py", "benchmarks/stn/m1_eval.py", "benchmarks/stn/m1_train_pair.py")
    return {f: (sha256_file(os.path.join(repo, f), 16) if os.path.exists(os.path.join(repo, f)) else None) for f in files}
