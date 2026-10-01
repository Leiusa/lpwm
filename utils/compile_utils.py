"""
Opt-in torch.compile for the DLP / LPWM training scripts (config key `torch_compile`).
"""
import torch


def compile_for_training(model, mode):
    """torch.compile the TRAINING forward pass only.

    The custom Triton STN crop/paste and fused-composite ops are kept out of the traced graph with
    torch.compiler.disable: Dynamo cannot trace their launchers (it passes symbolic strides into the hand-written
    kernel launch and fails), so they run exactly as in eager while everything around them is compiled.
    The caller keeps the uncompiled `model` for validation, plotting and checkpoints (same parameters, unchanged
    state_dict keys).
    """
    import lpwm_stn.triton_backend as tb
    import modules.modules as mm
    for mod, name in ((tb, 'stn_crop'), (tb, 'stn_paste'), (mm, 'composite_fused')):
        fn = getattr(mod, name, None)
        if fn is not None and not getattr(fn, '_lpwm_compile_disabled', False):
            wrapped = torch.compiler.disable(fn)
            wrapped._lpwm_compile_disabled = True
            setattr(mod, name, wrapped)
    return torch.compile(model) if mode == 'default' else torch.compile(model, mode=mode)


def compile_mode_from_config(config):
    """`torch_compile`: false/absent = eager (default); true or "default", "reduce-overhead", ... = that mode."""
    value = config.get('torch_compile', False)
    if value in (False, None):
        return None
    return 'default' if value is True else str(value)
