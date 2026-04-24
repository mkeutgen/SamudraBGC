import torch
import torch.nn.functional as F
from jaxtyping import Float

from ocean_emulators.constants import Grid
from ocean_emulators.utils.distributed import all_reduce_mean, get_world_size


def decomposed_mse(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """Standard MSE loss computed per channel."""
    pred = pred * wet
    target = target * wet
    return F.mse_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))

def decomposed_mae(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """Standard MAE (L1) loss computed per channel."""
    pred = pred * wet
    target = target * wet
    return F.l1_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))



def decomposed_mse_diff_weighted(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """MSE loss with weighted differences."""
    pred = pred * wet
    target = target * wet
    # Compute standard MSE
    mse = F.mse_loss(pred, target, reduction="none")

    # Weight the differences more heavily
    diff_weight = 2.0  # Adjustable weight factor
    diff_mse = (
        F.mse_loss(
            pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], reduction="none"
        )
        * diff_weight
    )

    # Combine losses
    combined_loss = torch.cat([mse[:, :1], diff_mse], dim=1)
    return combined_loss.mean(dim=(0, 2, 3))


def decomposed_mse_cos_weighted(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    """MSE loss weighted by cosine of latitude."""
    pred = pred * wet
    target = target * wet
    weights = cos.view(1, 1, -1, 1)  # Reshape for broadcasting
    mse = F.mse_loss(pred, target, reduction="none")
    weighted_mse = mse * weights
    return weighted_mse.mean(dim=(0, 2, 3))


def decomposed_mse_scaled(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor, scaling: torch.Tensor
) -> torch.Tensor:
    """MSE loss with scaled residuals."""
    pred = pred * wet
    target = target * wet
    scaled_pred = pred * scaling.view(1, -1, 1, 1)
    scaled_target = target * scaling.view(1, -1, 1, 1)
    return F.mse_loss(scaled_pred, scaled_target, reduction="none").mean(dim=(0, 2, 3))


def decomposed_mse_mae(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor
) -> torch.Tensor:
    """Combined MSE and MAE loss."""
    pred = pred * wet
    target = target * wet
    mse = F.mse_loss(pred, target, reduction="none")
    mae = F.l1_loss(pred, target, reduction="none")
    combined = (mse + mae) / 2
    return combined.mean(dim=(0, 2, 3))


class MseDynamic:
    """A loss function that scales each channel to contribute equally to the loss.

    This uses a rolling estimate of the loss of each channel to scale each
    channel's loss, discouraging the model from focusing on only a few channels.

    See: https://openathena.slack.com/archives/C08CYM42DT3/p1752275713570969
    """

    N_WINDOW = 25
    """Rolling window size to average over. (~number of steps)"""

    def __init__(
        self,
        wet: Grid,
        stds: Float[torch.Tensor, " var"],
        *,
        should_limit: bool,
    ):
        self._wet: Grid = wet
        self._per_channel_scale: Float[torch.Tensor, " var"] = torch.ones(
            stds.shape[0], device=wet.device
        )
        if should_limit:
            vars: Float[torch.Tensor, " var"] = stds.pow(2)
            self._limits: Float[torch.Tensor, " var"] | None = 1.0 / vars
        else:
            self._limits = None

    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        loss_with_history_channels: Float[torch.Tensor, " hist*var"] = decomposed_mse(
            pred, target, self._wet
        )
        scaled_loss_including_history_dimension: Float[torch.Tensor, "hist var"] = (
            loss_with_history_channels.reshape(self._per_channel_scale.shape[0], -1)
            * self._per_channel_scale.unsqueeze(1)
        )
        return scaled_loss_including_history_dimension.reshape(-1)

    def update(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> None:
        """Given the prediction & target for this step, update the per-channel scale."""
        mse_loss = decomposed_mse(pred, target, self._wet)
        mse_loss = torch.where(mse_loss == 0, 1e-8, mse_loss)
        new_target_weights_with_history: Float[torch.Tensor, " hist*var"] = (
            1.0 / mse_loss
        )
        # Reshape from channels * history to channels
        # by averaging along the `hist` dimension
        new_target_weights: Float[torch.Tensor, " var"] = (
            new_target_weights_with_history.reshape(
                self._per_channel_scale.shape[0], -1
            ).mean(dim=1)
        )
        if self._limits:
            new_target_weights = new_target_weights.min(self._limits)

        if get_world_size() > 1:
            all_reduce_mean(new_target_weights)

        self._per_channel_scale = (
            self._per_channel_scale * (MseDynamic.N_WINDOW - 1) + new_target_weights
        ) / MseDynamic.N_WINDOW

    def loss_scale_per_channel(self) -> Float[torch.Tensor, " var"]:
        return self._per_channel_scale

    # new methods for saving and loading state
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary for checkpointing."""
        return {"per_channel_scale": self._per_channel_scale.detach().cpu()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state from ``state_dict``."""
        if "per_channel_scale" in state:
            self._per_channel_scale = state["per_channel_scale"].to(self._wet.device)


#############################################
##### Maxime Custom Loss Funs ##################
################################################
def decomposed_mae_gradient_weighted(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    wet: torch.Tensor,
    gradient_weight: float = 0.1,
    second_order_weight: float = 0.0  # NEW: Add second-order term
) -> torch.Tensor:
    """
    MAE loss with WEIGHTED spatial gradient matching penalty.
    
    By controlling gradient_weight,
    we can balance accuracy (MAE term) vs sharpness (gradient term).
    
    Loss = MAE(pred, target) + α * gradient_penalty(pred, target) 
           + β * second_order_penalty(pred, target)
    
    where α = gradient_weight and β = second_order_weight are tunable hyperparameters.
    
    Second-order penalty matches the Laplacian (curvature), which helps preserve:
    - Eddy centers (local extrema)
    - Curvature of fronts
    - Spatial smoothness structure
    
    Recommended starting values:
    - α = 0.05: Very conservative, prioritize accuracy
    - α = 0.1:  Conservative, good balance 
    - α = 0.25: Moderate, more sharpness 
    - α = 0.5:  Aggressive sharpening
    - α = 1.0:  Equal weighting
    
    - β = 0.0:  Disabled (default)
    - β = 0.05: Very conservative second-order
    - β = 0.1:  Moderate second-order penalty
    
    Args:
        pred: Predicted tensor [batch, channels, height, width]
        target: Target tensor [batch, channels, height, width]
        wet: Wet mask [batch, channels, height, width]
        gradient_weight: Scaling factor α for first-order gradient penalty
        second_order_weight: Scaling factor β for second-order (Laplacian) penalty
    
    Returns:
        Loss per channel [channels]
    """
    pred = pred * wet
    target = target * wet
    
    # MAE term (main accuracy objective)
    mae_loss = F.l1_loss(pred, target, reduction="none")
    mae_per_channel = mae_loss.mean(dim=(0, 2, 3))
    
    # First-order gradient penalty: Match spatial gradients
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    
    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction="none")
    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction="none")
    
    # Average gradient losses
    grad_loss = (
        F.pad(grad_loss_y, (0, 0, 0, 1), value=0).mean(dim=(0, 2, 3)) +
        F.pad(grad_loss_x, (0, 1, 0, 0), value=0).mean(dim=(0, 2, 3))
    ) / 2
    
    # NEW: Second-order penalty (Laplacian/curvature matching)
    second_order_loss = 0.0
    if second_order_weight > 0:
        # Compute second derivatives (discrete Laplacian)
        # ∂²/∂x² ≈ f[i,j+1] - 2*f[i,j] + f[i,j-1]
        # ∂²/∂y² ≈ f[i+1,j] - 2*f[i,j] + f[i-1,j]
        
        # Second derivative in y-direction (requires 3 points)
        pred_grad2_y = pred[:, :, 2:, :] - 2*pred[:, :, 1:-1, :] + pred[:, :, :-2, :]
        target_grad2_y = target[:, :, 2:, :] - 2*target[:, :, 1:-1, :] + target[:, :, :-2, :]
        
        # Second derivative in x-direction
        pred_grad2_x = pred[:, :, :, 2:] - 2*pred[:, :, :, 1:-1] + pred[:, :, :, :-2]
        target_grad2_x = target[:, :, :, 2:] - 2*target[:, :, :, 1:-1] + target[:, :, :, :-2]
        
        # L1 loss on second derivatives
        grad2_loss_y = F.l1_loss(pred_grad2_y, target_grad2_y, reduction="none")
        grad2_loss_x = F.l1_loss(pred_grad2_x, target_grad2_x, reduction="none")
        
        # Average and pad to match dimensions
        second_order_loss = (
            F.pad(grad2_loss_y, (0, 0, 0, 2), value=0).mean(dim=(0, 2, 3)) +
            F.pad(grad2_loss_x, (0, 2, 0, 0), value=0).mean(dim=(0, 2, 3))
        ) / 2
    
    # Weighted combination
    total_loss = mae_per_channel + gradient_weight * grad_loss + second_order_weight * second_order_loss
    
    return total_loss






def decomposed_mae_gradient_relative(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    gradient_weight: float = 0.25,
    second_order_weight: float = 0.0,
) -> torch.Tensor:
    """
    MAE loss with RELATIVE spatial gradient matching.

    The key difference from ``decomposed_mae_gradient_weighted`` is how the
    gradient penalty is computed.  Instead of the absolute error
    ``|∇pred - ∇target|`` (dominated by strong fronts), we normalise by
    the local target-gradient magnitude so that a 50 % relative error in
    the subpolar gyre costs the same as a 50 % relative error at the Gulf
    Stream.

        grad_loss = |∇pred - ∇target| / (|∇target| + ε)

    where ε is the per-channel mean |∇target|.  Using the mean (rather
    than the median) guarantees bounded amplification: in perfectly smooth
    regions the relative error is at most |∇pred| / mean(|∇target|),
    keeping loss values on a comparable scale to the absolute variant.

    Loss = MAE(pred, target)
           + α · mean(|∇pred - ∇target| / (|∇target| + ε))
           + β · second_order_penalty(pred, target)

    Args:
        pred: [batch, channels, height, width]
        target: [batch, channels, height, width]
        wet: [batch, channels, height, width]
        gradient_weight: α — weight for relative gradient penalty
        second_order_weight: β — weight for Laplacian penalty (0 = disabled)

    Returns:
        Loss per channel [channels]
    """
    pred = pred * wet
    target = target * wet

    # ── MAE term ──
    mae_per_channel = F.l1_loss(pred, target, reduction="none").mean(dim=(0, 2, 3))

    # ── First-order gradients ──
    pred_gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    tgt_gy = target[:, :, 1:, :] - target[:, :, :-1, :]
    tgt_gx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Absolute gradient error
    err_gy = (pred_gy - tgt_gy).abs()
    err_gx = (pred_gx - tgt_gx).abs()

    # Per-channel adaptive epsilon: mean of |∇target| (bounded amplification).
    # In smooth regions: relative error = |∇pred| / mean(|∇target|)
    # At fronts:         relative error ≈ |error| / |∇target|
    tgt_mag_y = tgt_gy.abs()
    tgt_mag_x = tgt_gx.abs()
    eps_y = tgt_mag_y.permute(1, 0, 2, 3).reshape(tgt_mag_y.shape[1], -1).mean(dim=1)  # [C]
    eps_x = tgt_mag_x.permute(1, 0, 2, 3).reshape(tgt_mag_x.shape[1], -1).mean(dim=1)  # [C]
    eps_y = eps_y.clamp(min=1e-8).view(1, -1, 1, 1)
    eps_x = eps_x.clamp(min=1e-8).view(1, -1, 1, 1)

    # Relative gradient error
    rel_err_gy = err_gy / (tgt_mag_y + eps_y)
    rel_err_gx = err_gx / (tgt_mag_x + eps_x)

    grad_loss = (
        F.pad(rel_err_gy, (0, 0, 0, 1), value=0).mean(dim=(0, 2, 3))
        + F.pad(rel_err_gx, (0, 1, 0, 0), value=0).mean(dim=(0, 2, 3))
    ) / 2

    # ── Second-order penalty (Laplacian) ──
    second_order_loss = 0.0
    if second_order_weight > 0:
        pred_g2y = pred[:, :, 2:, :] - 2 * pred[:, :, 1:-1, :] + pred[:, :, :-2, :]
        tgt_g2y = target[:, :, 2:, :] - 2 * target[:, :, 1:-1, :] + target[:, :, :-2, :]
        pred_g2x = pred[:, :, :, 2:] - 2 * pred[:, :, :, 1:-1] + pred[:, :, :, :-2]
        tgt_g2x = target[:, :, :, 2:] - 2 * target[:, :, :, 1:-1] + target[:, :, :, :-2]

        g2_loss_y = F.l1_loss(pred_g2y, tgt_g2y, reduction="none")
        g2_loss_x = F.l1_loss(pred_g2x, tgt_g2x, reduction="none")

        second_order_loss = (
            F.pad(g2_loss_y, (0, 0, 0, 2), value=0).mean(dim=(0, 2, 3))
            + F.pad(g2_loss_x, (0, 2, 0, 0), value=0).mean(dim=(0, 2, 3))
        ) / 2

    return mae_per_channel + gradient_weight * grad_loss + second_order_weight * second_order_loss


class MaeDynamic:
    """MAE-based dynamic loss with optional relative gradient penalty.

    Analogous to ``MseDynamic`` but uses L1 (MAE) as the base loss so that
    channels with high variance (e.g. surface temperature/salinity) do not
    dominate via squared errors.

    The dynamic scaling maintains a rolling estimate of the per-channel MAE and
    sets each channel's weight to 1/MAE, so every channel contributes equally
    to the total loss regardless of its absolute scale.

    Optionally adds a relative spatial gradient penalty (same as
    ``decomposed_mae_gradient_relative``) to encourage sharpness of fronts and
    eddies, normalised locally so a weak gradient region is not overwhelmed by
    a strong-front region.

    Loss = scale_c * (MAE_c + α * rel_grad_c)   for each channel c
    """

    DEFAULT_N_WINDOW = 25
    """Default rolling window size to average over (~number of steps)."""

    def __init__(
        self,
        wet: Grid,
        n_vars: int,
        *,
        gradient_weight: float = 0.0,
        n_window: int = DEFAULT_N_WINDOW,
    ):
        self._wet: Grid = wet
        self._n_vars = n_vars
        self._gradient_weight = gradient_weight
        self._n_window = n_window
        self._per_channel_scale: Float[torch.Tensor, " var"] = torch.ones(
            n_vars, device=wet.device
        )

    # ------------------------------------------------------------------
    def _mae_per_channel(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        p = pred * self._wet
        t = target * self._wet
        return F.l1_loss(p, t, reduction="none").mean(dim=(0, 2, 3))

    def _rel_grad_per_channel(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        p = pred * self._wet
        t = target * self._wet

        err_gy = (p[:, :, 1:, :] - p[:, :, :-1, :]) - (t[:, :, 1:, :] - t[:, :, :-1, :])
        err_gx = (p[:, :, :, 1:] - p[:, :, :, :-1]) - (t[:, :, :, 1:] - t[:, :, :, :-1])

        tgt_mag_y = (t[:, :, 1:, :] - t[:, :, :-1, :]).abs()
        tgt_mag_x = (t[:, :, :, 1:] - t[:, :, :, :-1]).abs()

        eps_y = tgt_mag_y.permute(1, 0, 2, 3).reshape(t.shape[1], -1).mean(dim=1).clamp(min=1e-8).view(1, -1, 1, 1)
        eps_x = tgt_mag_x.permute(1, 0, 2, 3).reshape(t.shape[1], -1).mean(dim=1).clamp(min=1e-8).view(1, -1, 1, 1)

        rel_gy = err_gy.abs() / (tgt_mag_y + eps_y)
        rel_gx = err_gx.abs() / (tgt_mag_x + eps_x)

        return (
            F.pad(rel_gy, (0, 0, 0, 1), value=0).mean(dim=(0, 2, 3))
            + F.pad(rel_gx, (0, 1, 0, 0), value=0).mean(dim=(0, 2, 3))
        ) / 2

    # ------------------------------------------------------------------
    def __call__(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> Float[torch.Tensor, " hist*var"]:
        mae = self._mae_per_channel(pred, target)

        if self._gradient_weight > 0:
            grad = self._rel_grad_per_channel(pred, target)
            loss_per_ch = mae + self._gradient_weight * grad
        else:
            loss_per_ch = mae

        scaled = loss_per_ch.reshape(self._n_vars, -1) * self._per_channel_scale.unsqueeze(1)
        return scaled.reshape(-1)

    def update(
        self,
        pred: Float[torch.Tensor, "batch hist*var lat lon"],
        target: Float[torch.Tensor, "batch hist*var lat lon"],
    ) -> None:
        """Update the per-channel scale from the current MAE (no gradient term)."""
        mae = self._mae_per_channel(pred, target)
        mae = torch.where(mae == 0, torch.tensor(1e-8, device=mae.device), mae)

        # Average across history dimension to get per-variable weights
        new_weights = (1.0 / mae).reshape(self._n_vars, -1).mean(dim=1)

        if get_world_size() > 1:
            all_reduce_mean(new_weights)

        self._per_channel_scale = (
            self._per_channel_scale * (self._n_window - 1) + new_weights
        ) / self._n_window

    def loss_scale_per_channel(self) -> Float[torch.Tensor, " var"]:
        return self._per_channel_scale

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"per_channel_scale": self._per_channel_scale.detach().cpu()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "per_channel_scale" in state:
            self._per_channel_scale = state["per_channel_scale"].to(self._wet.device)
