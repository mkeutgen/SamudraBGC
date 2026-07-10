"""Tests for the dynamic per-channel loss scaling (audit finding 2).

Channels are packed time-major by the data pipeline:
``rearrange(..., "window time variable lat lon -> window (time variable) lat lon")``
i.e. flat channel index = t * n_vars + v. The dynamic losses must therefore
reshape per-channel losses to (time, var) — reshape(-1, n_vars) — so the
per-variable scale lines up with the variable axis. The pre-fix code reshaped
to (var, time), applying each variable's scale to a stripe of *different*
variables.

Expected values are hand-computed; they match the canonical implementation in
utils/loss_openathena.py (DynamicLoss), which documents the time-major layout.
(That module is not imported here because it depends on ocean_emulators.utils.ctx,
which does not exist in this repo.)
"""

import pytest
import torch

from ocean_emulators.utils.loss import MaeDynamic, MseDynamic

N_VARS = 4
HIST = 1  # two time frames, as in all production configs
N_CHANNELS = (HIST + 1) * N_VARS


def _pred_target_with_channel_losses(abs_diff_per_channel: list[float]):
    """Build (pred, target) whose per-channel MAE is abs_diff_per_channel
    and per-channel MSE is abs_diff_per_channel**2 (constant fields)."""
    n_ch = len(abs_diff_per_channel)
    target = torch.zeros(1, n_ch, 2, 2)
    pred = torch.tensor(abs_diff_per_channel).view(1, n_ch, 1, 1).expand(1, n_ch, 2, 2)
    return pred.clone(), target


@pytest.fixture
def wet():
    return torch.ones(N_CHANNELS, 2, 2, dtype=torch.bool)


def test_mae_dynamic_scales_time_major(wet):
    """Per-variable scales must be applied per variable, not per time-stripe.

    This reproduces the audit finding 2 example: per-channel losses [1..8]
    (time-major: t0 holds vars 0-3, t1 holds vars 4-7 of the flat index) and
    per-variable scales [1,2,3,4] must give [1,4,9,16, 5,12,21,32]. The buggy
    var-major reshape gave [1,2,6,8, 15,18,28,32].
    """
    loss_fn = MaeDynamic(wet=wet, n_vars=N_VARS, gradient_weight=0.0)
    loss_fn._per_channel_scale = torch.tensor([1.0, 2.0, 3.0, 4.0])

    pred, target = _pred_target_with_channel_losses([1, 2, 3, 4, 5, 6, 7, 8])
    scaled = loss_fn(pred, target)

    expected = torch.tensor([1.0, 4.0, 9.0, 16.0, 5.0, 12.0, 21.0, 32.0])
    assert torch.allclose(scaled, expected), f"{scaled=} vs {expected=}"


def test_mae_dynamic_update_time_major(wet):
    """update() must average 1/MAE over the *time* frames of each variable."""
    n_window = 25
    loss_fn = MaeDynamic(wet=wet, n_vars=N_VARS, gradient_weight=0.0, n_window=n_window)

    pred, target = _pred_target_with_channel_losses([1, 2, 3, 4, 5, 6, 7, 8])
    loss_fn.update(pred, target)

    # Variable v appears at flat channels v (t=0) and N_VARS + v (t=1).
    mae = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    new_weights = (1.0 / mae).reshape(HIST + 1, N_VARS).mean(dim=0)
    expected_scale = (torch.ones(N_VARS) * (n_window - 1) + new_weights) / n_window

    assert loss_fn._per_channel_scale.shape == (N_VARS,)
    assert torch.allclose(loss_fn._per_channel_scale, expected_scale)


def test_mse_dynamic_scales_time_major(wet):
    """Same time-major contract for the MSE variant."""
    loss_fn = MseDynamic(wet=wet, stds=torch.ones(N_VARS), should_limit=False)
    loss_fn._per_channel_scale = torch.tensor([1.0, 2.0, 3.0, 4.0])

    pred, target = _pred_target_with_channel_losses([1, 2, 3, 4, 5, 6, 7, 8])
    scaled = loss_fn(pred, target)

    mse = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) ** 2
    expected = (mse.reshape(HIST + 1, N_VARS) * loss_fn._per_channel_scale).reshape(-1)
    assert torch.allclose(scaled, expected), f"{scaled=} vs {expected=}"


def test_mse_dynamic_update_time_major(wet):
    n_window = MseDynamic.N_WINDOW
    loss_fn = MseDynamic(wet=wet, stds=torch.ones(N_VARS), should_limit=False)

    pred, target = _pred_target_with_channel_losses([1, 2, 3, 4, 5, 6, 7, 8])
    loss_fn.update(pred, target)

    mse = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) ** 2
    new_weights = (1.0 / mse).reshape(HIST + 1, N_VARS).mean(dim=0)
    expected_scale = (torch.ones(N_VARS) * (n_window - 1) + new_weights) / n_window

    assert loss_fn._per_channel_scale.shape == (N_VARS,)
    assert torch.allclose(loss_fn._per_channel_scale, expected_scale)


def test_mse_dynamic_update_with_limits(wet):
    """Regression test for audit finding 10.

    With should_limit=True, _limits is a multi-element tensor; the buggy
    ``if self._limits:`` raised an ambiguous-truthiness RuntimeError on the
    first update. The weights must instead be capped elementwise at 1/std^2.
    (On this branch the ``is not None`` guard was already present; this test
    guards against a regression.)
    """
    stds = torch.tensor([0.5, 1.0, 2.0, 4.0])
    loss_fn = MseDynamic(wet=wet, stds=stds, should_limit=True)

    pred, target = _pred_target_with_channel_losses([1, 2, 3, 4, 5, 6, 7, 8])
    loss_fn.update(pred, target)  # raised RuntimeError before the fix

    mse = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) ** 2
    new_weights = (1.0 / mse).reshape(HIST + 1, N_VARS).mean(dim=0)
    limits = 1.0 / stds.pow(2)
    capped = new_weights.min(limits)
    n_window = MseDynamic.N_WINDOW
    expected_scale = (torch.ones(N_VARS) * (n_window - 1) + capped) / n_window

    assert torch.allclose(loss_fn._per_channel_scale, expected_scale)
