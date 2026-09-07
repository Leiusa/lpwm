#!/usr/bin/env python
"""
Static checks that the four training scripts request alpha_masks correctly (commit 3).

The scripts need a real dataset and many epochs to run, so this verifies the
control flow by parsing the source rather than executing it. What matters is a
structural property, and AST is the right tool for it:

    the post-loop plotting block reads the LAST batch's tensors, so masks must be
    requested for exactly the last batch of a plotting epoch, and for no other.

Getting this wrong is silent: `return_alpha_masks=False` on a plotting epoch
leaves `alpha_masks = None` and the plotting block dies on `torch.where`.

    python tests/stn/test_training_scripts_alpha_masks.py
    pytest tests/stn/test_training_scripts_alpha_masks.py
"""

import ast
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPTS = ("train_dlp.py", "train_dlp_accelerate.py", "train_lpwm.py", "train_lpwm_accelerate.py")
ACCELERATE = ("train_dlp_accelerate.py", "train_lpwm_accelerate.py")


def _src(name):
    with open(os.path.join(_ROOT, name)) as f:
        return f.read()


def _tree(name):
    return ast.parse(_src(name), filename=name)


def test_all_scripts_parse():
    for name in SCRIPTS:
        _tree(name)


def test_batch_loop_is_enumerated():
    """`for batch in pbar` -> `for batch_idx, batch in enumerate(pbar)`."""
    for name in SCRIPTS:
        src = _src(name)
        assert "for batch_idx, batch in enumerate(pbar):" in src, name
        assert "for batch in pbar:" not in src, f"{name}: un-enumerated loop remains"


def test_exactly_one_plot_this_epoch_definition():
    """One condition, computed once per epoch, before the batch loop."""
    for name in SCRIPTS:
        tree = _tree(name)
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "plot_this_epoch" for t in n.targets)]
        assert len(assigns) == 1, f"{name}: {len(assigns)} definitions of plot_this_epoch, want 1"
        expr = ast.unparse(assigns[0].value)
        assert "eval_epoch_freq" in expr and "num_epochs - 1" in expr, f"{name}: {expr}"


def test_old_duplicated_guard_is_gone():
    """The post-loop block must reuse plot_this_epoch, not recompute the condition."""
    for name in SCRIPTS:
        src = _src(name)
        assert "if plot_this_epoch:" in src, name
        assert "if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:" not in src, (
            f"{name}: duplicated plotting condition still present")


def test_need_masks_is_last_batch_of_a_plotting_epoch():
    for name in SCRIPTS:
        tree = _tree(name)
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "need_masks" for t in n.targets)]
        assert len(assigns) == 1, f"{name}: {len(assigns)} definitions of need_masks, want 1"
        expr = ast.unparse(assigns[0].value)
        assert "plot_this_epoch" in expr, f"{name}: {expr}"
        assert "batch_idx" in expr and "len(dataloader) - 1" in expr, f"{name}: {expr}"
        assert isinstance(assigns[0].value, ast.BoolOp) and isinstance(assigns[0].value.op, ast.And), (
            f"{name}: need_masks must be a conjunction, got {expr}")


def test_forward_passes_return_alpha_masks_need_masks():
    """Exactly one model(...) call, carrying the keyword."""
    for name in SCRIPTS:
        tree = _tree(name)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "model"]
        with_kw = [c for c in calls
                   if any(k.arg == "return_alpha_masks" for k in c.keywords)]
        assert len(with_kw) == 1, f"{name}: {len(with_kw)} model() calls pass return_alpha_masks, want 1"
        kw = next(k for k in with_kw[0].keywords if k.arg == "return_alpha_masks")
        assert isinstance(kw.value, ast.Name) and kw.value.id == "need_masks", (
            f"{name}: return_alpha_masks={ast.unparse(kw.value)}, want need_masks")


def test_need_masks_defined_before_the_forward_call():
    for name in SCRIPTS:
        tree = _tree(name)
        assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "need_masks" for t in n.targets))
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "model"
                    and any(k.arg == "return_alpha_masks" for k in n.keywords))
        assert assign.lineno < call.lineno, f"{name}: need_masks defined after the forward call"


def test_plot_this_epoch_defined_before_the_batch_loop():
    for name in SCRIPTS:
        src = _src(name)
        assert src.index("plot_this_epoch = ") < src.index("for batch_idx, batch in enumerate(pbar):"), name


def test_accelerate_flag_is_rank_independent():
    """The same flag on every rank: no is_main_process in need_masks, and the
    existing main-process plotting guard is preserved."""
    for name in ACCELERATE:
        tree = _tree(name)
        assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "need_masks" for t in n.targets))
        expr = ast.unparse(assign.value)
        assert "is_main_process" not in expr, f"{name}: need_masks is rank-dependent: {expr}"
        assert "is_local_main_process" not in expr, f"{name}: need_masks is rank-dependent: {expr}"
        assert "if accelerator.is_main_process:" in _src(name), (
            f"{name}: the main-process plotting guard was removed")


def test_no_plotting_moved_into_the_batch_loop():
    """Plotting must stay after the loop; only the flag moved inward."""
    for name in SCRIPTS:
        src = _src(name)
        loop_at = src.index("for batch_idx, batch in enumerate(pbar):")
        guard_at = src.index("if plot_this_epoch:")
        assert guard_at > loop_at, f"{name}: plotting guard precedes the batch loop"
        body = src[loop_at:guard_at]
        assert "plot_bb_on_image_batch_from_masks_nms" not in body, f"{name}: plotting moved into the loop"
        assert "create_segmentation_map" not in body, f"{name}: plotting moved into the loop"


def test_dataloader_length_used_not_a_hardcoded_count():
    for name in SCRIPTS:
        tree = _tree(name)
        assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "need_masks" for t in n.targets))
        expr = ast.unparse(assign.value)
        assert "len(dataloader)" in expr, f"{name}: {expr} does not use len(dataloader)"


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:                                     # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL {name}: {exc}")
        else:
            print(f"  ok   {name}")
    print(f"\n{'FAILED' if failures else 'PASSED'}: {len(failures)} of {len(tests)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
