"""Structure-function losses for spectral-fidelity training.

This module computes velocity / scalar structure functions on 2D tensor
fields and provides a differentiable loss that matches the structure
functions of a prediction to those of the ground truth, per radial-distance
bin and per order.

A structure function of order ``n`` measures how much a field changes
between two points separated by an integer lattice offset ``d = (di, dj)``:

* scalar field ``f``:  ``S^n = mean( (f(x+d) - f(x))^n )``
* vector field ``(u, v)``: the 2-vector increment is first projected onto
  the unit separation direction ``d_hat = d / ||d||``, then raised to the
  power ``n`` and averaged.

The domain is non-periodic and land-masked, so shifted differences are
computed by *slicing* (never ``torch.roll``, which would wrap across the
boundary), and a pixel pair contributes only if both endpoints are wet
(``mask == 1``).

All operations are pure tensor math and fully differentiable, so the loss
can be used directly in training backprop.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "build_shift_bank",
    "masked_increment",
    "sf_estimate",
    "sf_loss",
    "sf_loss_per_channel",
    "signed_log",
]


def build_shift_bank(max_lag: int = 10, n_bins: int = 5) -> dict[int, list[tuple[int, int]]]:
    """Build integer lattice offsets grouped by radial-distance bin.

    Generates every integer offset ``(di, dj)`` with
    ``1 <= sqrt(di^2 + dj^2) <= max_lag``. To avoid double-counting a
    separation and its exact opposite (which give the same even-order SF
    and a negated odd-order SF), only offsets in the upper half-plane are
    kept: ``dj > 0``, or (``dj == 0`` and ``di > 0``).

    Offsets are grouped into ``n_bins`` radial bins. The bin index for a
    radius ``r`` is ``int((r - 1e-9) / (max_lag / n_bins))``, clamped to
    ``[0, n_bins - 1]``.

    Args:
        max_lag: Maximum separation distance (in grid cells) to include.
        n_bins: Number of radial-distance bins.

    Returns:
        Dict mapping ``bin_idx -> list of (di, dj)`` offset tuples. Bins
        that receive no offsets are omitted.
    """
    bin_width = max_lag / n_bins
    bank: dict[int, list[tuple[int, int]]] = {}
    for di in range(-max_lag, max_lag + 1):
        for dj in range(0, max_lag + 1):
            # Upper half-plane only: dj > 0, or (dj == 0 and di > 0).
            if dj == 0 and di <= 0:
                continue
            r = math.sqrt(di * di + dj * dj)
            if r < 1.0 or r > max_lag:
                continue
            bin_idx = int((r - 1e-9) / bin_width)
            bin_idx = min(max(bin_idx, 0), n_bins - 1)
            bank.setdefault(bin_idx, []).append((di, dj))
    return bank


def masked_increment(
    field: torch.Tensor,
    mask: torch.Tensor,
    di: int,
    dj: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the shifted increment of a field and its validity mask.

    The increment for offset ``(di, dj)`` is computed by slicing (no
    wrap-around), so only the overlap region between the original and the
    shifted field is returned:

    * ``increment[..., y, x] = field[..., y + di, x + dj] - field[..., y, x]``
    * ``valid[..., y, x] = mask(y, x) * mask(y + di, x + dj)``

    ``di`` and ``dj`` may be any integers (positive, negative, or zero).
    The operation is fully differentiable.

    Args:
        field: Tensor of shape ``(B, C, H, W)``.
        mask: Tensor of shape ``(1, 1, H, W)`` or ``(B, 1, H, W)`` with
            1 marking valid (wet) pixels.
        di: Row offset of the second endpoint.
        dj: Column offset of the second endpoint.

    Returns:
        Tuple ``(increment, valid)``. Both have spatial shape
        ``(H - |di|, W - |dj|)``; ``increment`` keeps the field's batch and
        channel dims, ``valid`` keeps the mask's leading dims.
    """
    H, W = field.shape[-2], field.shape[-1]
    # Source (unshifted) endpoint slices.
    src_y = slice(max(0, -di), H - max(0, di))
    src_x = slice(max(0, -dj), W - max(0, dj))
    # Destination (shifted) endpoint slices.
    dst_y = slice(max(0, di), H - max(0, -di))
    dst_x = slice(max(0, dj), W - max(0, -dj))

    increment = field[..., dst_y, dst_x] - field[..., src_y, src_x]
    valid = mask[..., src_y, src_x] * mask[..., dst_y, dst_x]
    return increment, valid


def sf_estimate(
    field: torch.Tensor,
    mask: torch.Tensor,
    shifts: list[tuple[int, int]],
    n: int,
    vector: bool,
) -> torch.Tensor:
    """Estimate the order-``n`` structure function over a set of shifts.

    Averages ``increment^n`` over all valid pixel pairs across all given
    shifts. For ``vector=True``, the field must have exactly 2 channels
    ``(u, v)`` and each 2-vector increment is projected onto the unit
    separation direction before being raised to the power ``n``
    (longitudinal structure function). For ``vector=False``, scalar
    increments are computed per channel.

    Args:
        field: Tensor of shape ``(B, C, H, W)``; ``C == 2`` if ``vector``.
        mask: Validity mask, ``(1, 1, H, W)`` or ``(B, 1, H, W)``.
        shifts: List of ``(di, dj)`` integer offsets.
        n: Structure-function order (e.g. 2 or 3).
        vector: Whether to treat the field as a 2-vector ``(u, v)``.

    Returns:
        A scalar tensor (vector case) or a per-channel tensor of shape
        ``(C,)`` (scalar case). Differentiable w.r.t. ``field``.
    """
    if vector and field.shape[1] != 2:
        raise ValueError(f"vector=True requires 2 channels (u, v), got {field.shape[1]}")

    num: torch.Tensor | None = None
    den: torch.Tensor | None = None
    for di, dj in shifts:
        increment, valid = masked_increment(field, mask, di, dj)
        if vector:
            norm = math.sqrt(di * di + dj * dj)
            # Project (du, dv) onto d_hat = (di, dj) / ||d||.
            projected = (increment[:, 0] * (di / norm) + increment[:, 1] * (dj / norm))
            powed = projected**n  # (B, h, w)
            v = valid[:, 0]  # (1 or B, h, w)
            shift_num = (powed * v).sum()
            shift_den = v.expand_as(powed).sum()
        else:
            powed = increment**n  # (B, C, h, w)
            shift_num = (powed * valid).sum(dim=(0, 2, 3))  # (C,)
            shift_den = valid.expand_as(powed).sum(dim=(0, 2, 3))  # (C,)
        num = shift_num if num is None else num + shift_num
        den = shift_den if den is None else den + shift_den

    if num is None or den is None:
        raise ValueError("sf_estimate requires at least one shift")
    return num / den.clamp(min=1.0)


def signed_log(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving logarithmic compression: ``sign(x) * log1p(|x|)``.

    Monotone, odd, and differentiable; preserves the sign of odd-order
    structure functions while compressing their dynamic range.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def sf_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    shift_bank: dict[int, list[tuple[int, int]]],
    orders: tuple[int, ...] = (2,),
    vector: bool = True,
    weights: dict | None = None,
    log_space: bool = True,
) -> torch.Tensor:
    """Structure-function loss: match pred's SF to truth's SF per (order, bin).

    For each radial bin ``r`` in ``shift_bank`` and each order ``n`` in
    ``orders``, the structure functions of ``pred`` and ``truth`` are
    estimated and compared with an (optionally weighted) absolute
    difference. With ``log_space=True``, both estimates are first mapped
    through :func:`signed_log`, which compresses dynamic range while
    preserving the sign of odd-order SFs.

    Args:
        pred: Predicted field, ``(B, C, H, W)``.
        truth: Ground-truth field, ``(B, C, H, W)``.
        mask: Validity mask, ``(1, 1, H, W)``.
        shift_bank: Output of :func:`build_shift_bank`.
        orders: Structure-function orders to match.
        vector: Treat fields as 2-vectors (longitudinal projection).
        weights: Optional ``{(order, bin_idx): float}`` per-term weights;
            missing keys default to 1.0.
        log_space: Compare SFs in signed-log space.

    Returns:
        Scalar loss tensor (mean over all (order, bin) terms). Fully
        differentiable w.r.t. ``pred`` (and ``truth``).
    """
    terms: list[torch.Tensor] = []
    for bin_idx, shifts in shift_bank.items():
        for n in orders:
            sp = sf_estimate(pred, mask, shifts, n, vector)
            st = sf_estimate(truth, mask, shifts, n, vector)
            if log_space:
                sp = signed_log(sp)
                st = signed_log(st)
            alpha = 1.0 if weights is None else float(weights.get((n, bin_idx), 1.0))
            terms.append(alpha * (sp - st).abs().mean())
    if not terms:
        raise ValueError("sf_loss requires a non-empty shift_bank and orders")
    return torch.stack(terms).mean()


def sf_loss_per_channel(
    pred: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    shift_bank: dict[int, list[tuple[int, int]]],
    orders: tuple[int, ...] = (2,),
    weights: dict | None = None,
    log_space: bool = True,
) -> torch.Tensor:
    """Scalar-mode structure-function loss, returned PER CHANNEL.

    Same math as :func:`sf_loss` with ``vector=False``, but instead of
    reducing to one scalar, the per-channel dimension is kept. For each
    radial bin ``r`` in ``shift_bank`` and each order ``n`` in ``orders``::

        sp = sf_estimate(pred,  mask, shifts_r, n, vector=False)   # (C,)
        st = sf_estimate(truth, mask, shifts_r, n, vector=False)   # (C,)
        if log_space: sp, st = signed_log(sp), signed_log(st)
        term_c = alpha * (sp - st).abs()                           # (C,)

    The terms are accumulated across all ``(n, r)`` and divided by the number
    of terms (mean), giving a ``(C,)`` tensor. Fully differentiable.

    Args:
        pred: Predicted field, ``(B, C, H, W)``.
        truth: Ground-truth field, ``(B, C, H, W)``.
        mask: Validity mask, ``(1, 1, H, W)``.
        shift_bank: Output of :func:`build_shift_bank`.
        orders: Structure-function orders to match.
        weights: Optional ``{(order, bin_idx): float}`` per-term weights;
            missing keys default to 1.0.
        log_space: Compare SFs in signed-log space.

    Returns:
        Per-channel loss tensor of shape ``(C,)`` (mean over all
        ``(order, bin)`` terms). Fully differentiable w.r.t. ``pred``.
    """
    terms: list[torch.Tensor] = []
    for bin_idx, shifts in shift_bank.items():
        for n in orders:
            sp = sf_estimate(pred, mask, shifts, n, vector=False)  # (C,)
            st = sf_estimate(truth, mask, shifts, n, vector=False)  # (C,)
            if log_space:
                sp = signed_log(sp)
                st = signed_log(st)
            alpha = 1.0 if weights is None else float(weights.get((n, bin_idx), 1.0))
            terms.append(alpha * (sp - st).abs())  # (C,)
    if not terms:
        raise ValueError("sf_loss_per_channel requires a non-empty shift_bank and orders")
    return torch.stack(terms, dim=0).mean(dim=0)  # (C,)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. Shift bank: print bin counts.
    bank = build_shift_bank(max_lag=10, n_bins=5)
    for b in sorted(bank):
        print(f"bin {b}: {len(bank[b])} offsets")

    # 2. Identical inputs -> loss ~ 0.
    field = torch.randn(2, 2, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    loss_same = sf_loss(field, field, mask, bank, orders=(2, 3), vector=True)
    print(f"loss(field, field) = {loss_same.item():.3e}")
    assert loss_same.item() < 1e-10, "identical inputs must give ~0 loss"

    # 3. Different inputs -> loss > 0.
    other = torch.randn(2, 2, 32, 32) * 2.0
    loss_diff = sf_loss(field, other, mask, bank, orders=(2, 3), vector=True)
    print(f"loss(field, other) = {loss_diff.item():.3e}")
    assert loss_diff.item() > 0, "different inputs must give positive loss"

    # 4. Gradients flow.
    pred = torch.randn(2, 2, 32, 32, requires_grad=True)
    loss = sf_loss(pred, other, mask, bank, orders=(2, 3), vector=True)
    loss.backward()
    assert pred.grad is not None, "gradients must flow to pred"
    print(f"grad norm = {pred.grad.norm().item():.3e}")

    # Scalar (per-channel) path sanity check.
    loss_scalar = sf_loss(field, other, mask, bank, orders=(2,), vector=False)
    print(f"scalar-path loss = {loss_scalar.item():.3e}")
    assert loss_scalar.item() > 0

    print("All self-tests passed.")
