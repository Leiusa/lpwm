# `return_alpha_masks` — consumer audit, test design, and diff plan

Prepared work only. **Nothing implemented, no branch, no commit, no GPU.**
Option 2 from the fusion design: conditional materialization behind a flag
defaulting to today's behaviour.

## A. Consumer audit

**Producers** — three call sites, all internal to `modules/modules.py`:

| line | site |
|---|---|
| [5306](../modules/modules.py#L5306) | `get_objects_alpha_rgb_with_depth` returns `(a_obj, alpha_mask, dec_objects_trans)` |
| [5328](../modules/modules.py#L5328) | unpacked as `alpha_masks, bg_mask, dec_objects_trans` |
| [5330](../modules/modules.py#L5330) | `decode_objects` returns the 4-tuple `(dec_objects, dec_objects_trans, alpha_masks, bg_mask)` |
| [5353-5355](../modules/modules.py#L5353) | sole caller; unpacks into `object_dec_out`, then the 4-tuple |
| [5359](../modules/modules.py#L5359) | enters the decoder dict as `'alpha_masks'` |
| [models.py:1156, 1268](../models.py#L1156) | lifted into the model output dict |

Positional unpacking exists at exactly two sites, both internal. No external
caller unpacks these tuples, so returning `None` in position is safe *there*.

**Consumers** — 61 read sites, every one classified:

| group | sites | uses it? | breaks on `None`? |
|---|---|---|---|
| `calc_dyn_elbo` ([models.py:1363](../models.py#L1363)) | 1 | **no** — bound, never read (AST-verified) | no |
| `calc_static_elbo` ([models.py:1844](../models.py#L1844)) | 1 | **no** — bound, never read (AST-verified) | no |
| `train_lpwm.py`, `train_dlp.py`, `train_lpwm_accelerate.py`, `train_dlp_accelerate.py` | 24 | yes — plotting | **yes** |
| `eval/eval_model.py` | 12 | yes — plotting | yes |
| notebooks (3 files) | 21 | yes — plotting | yes |
| producers/plumbing | 2 | pass-through | no |

Both ELBO paths — dynamic *and* static — bind and discard it. Confirmed by AST
name-resolution, not by grep.

### The finding that changes the plan

In **all four** training scripts the read and the use sit at *different loop depths*:

```
for epoch:
    for batch:
        alpha_masks = model_output['alpha_masks']     # EVERY batch
    if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
        alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)   # LAST batch's value
```

The plotting block is outside the batch loop and consumes the **last batch's**
tensor, leaking out of the loop. So a blanket `return_alpha_masks=False`
throughout training leaves `None` at plot time and crashes on the `torch.where`.

The flag must therefore be requested per batch:

```python
plot_this_epoch = (epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1)
...
need_masks = plot_this_epoch and (batch_idx == len(dataloader) - 1)
model_output = model(x, ..., return_alpha_masks=need_masks)
```

`plot_this_epoch` is known at the top of the epoch and `batch_idx` is known in
the loop, so this needs no lookahead and no behaviour change: exactly the tensor
that gets plotted today still gets built.

### Honest sizing — do not expect bs=5 from this

At bair's own bs=5 (`B=85, K=90, 128²`, fp32):

| intermediate | size | removable by the flag alone? |
|---|---:|---|
| `dec_objects_trans` `[B,K,4,H,W]` | 2.01 GB | no — feeds the reduced outputs |
| `rgba_obj` `[B,K,3,H,W]` | 1.50 GB | no |
| `importance_map` `[B,K,1,H,W]` | 0.50 GB | no |
| **`alpha_masks` `[B,K,1,H,W]`** | **0.50 GB** | **yes** |

**The flag alone recovers 0.50 GB of 4.51 GB — about 11% of the composite
intermediates, roughly a 2% dent in the measured 18.5 GB training peak. It will
not make bair bs=5 fit.** Its real value is structural: it removes the
requirement that a fused kernel emit a per-particle output at all, which is what
lets fusion skip the other 4.0 GB. Sequencing it first is right; expecting a
memory win from it on its own is not.

## B. Test design

New file `tests/stn/test_alpha_masks_api.py` (never touching `tests/__init__.py`).

**Equivalence, `True` vs `False`** — same config, same seed, same weights, same input, one model per flag value, both ELBO configurations (dynamic and static):

1. every output-dict tensor except `alpha_masks` is **bit-identical**
2. scalar loss **bit-identical**
3. every entry of `loss_dict` bit-identical
4. **parameter gradients** bit-identical for all `model.parameters()` after one backward — the check that proves no gradient path was lost
5. input gradients bit-identical where inputs require grad
6. with `True`: `alpha_masks` matches shape/dtype/values of today's output
7. with `False`: `alpha_masks is None` and no `[B,K,1,H,W]` tensor is allocated — asserted via `torch.cuda.max_memory_allocated()` delta, not by inspection

Bit-exactness is the right bar here: the flag must not perturb arithmetic at all,
so unlike a kernel change there is no reassociation to tolerate.

**Contract preservation** — the existing suite must still pass unchanged with the
default (`True`), on both backends, at the established tolerances. No fixture is
regenerated and no tolerance moves.

**Consumer smoke** — a plotting-path test driving the `train_lpwm.py` block
structure with `need_masks` True on the last batch and False elsewhere, asserting
the plotted tensor is identical to the unflagged run.

## C. Proposed diff plan — three separate commits

**Commit 1 — harness** (already prepared, pending the GPU validation run)
`benchmarks/stn/{README.md,build_model.py,bench_e2e.py}` + `docs/stn_composite_fusion_design.md`.

**Commit 2 — the API change, no kernel**

| file | change |
|---|---|
| `modules/modules.py` | `get_objects_alpha_rgb_with_depth(..., return_alpha_masks=True)`; when `False`, skip only the final `a_obj = importance_map * a_obj` and return `None` in position 0. Every other line untouched — compositing order, `eps`, reduction order, dtype all unchanged. |
| `modules/modules.py` | thread the flag through `decode_objects` and its caller, defaulting `True` |
| `models.py` | thread through `forward(..., return_alpha_masks=True)` into the decoder call; dict key still always present, value may be `None` |
| `tests/stn/test_alpha_masks_api.py` | new, per §B |

Backend-independent by construction: the flag lives in the model/decoder, never
in `lpwm_stn`, and kernel selection stays with the registry. No Triton symbol
appears in `models.py` or `modules/modules.py`.

**Commit 3 — training-script opt-in** (separate, because it changes caller behaviour)
The four `train_*.py` scripts pass `need_masks` as derived above. Deliberately
split from Commit 2 so the API change can be reverted independently of the
callers, and so a bisect can tell an API fault from a caller fault.

`eval/eval_model.py` and the notebooks are **not touched** — they run with the
default and keep working.

## D. Open question before Commit 2

`None` in an existing output position is the honest signal, but any downstream
code doing `model_output['alpha_masks'].shape` without a guard fails loudly
rather than silently. The audit above says nothing in this repo does that on the
training path — but if you have external consumers of the output dict, a
zero-element tensor of the right rank would be quieter than `None`. `None` is my
recommendation; say if you'd rather have the quiet variant.
