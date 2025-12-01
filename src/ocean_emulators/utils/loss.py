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





# TODO CLEAN THIS FILE - REMOVE OLD VERSIONS OF LOSS FUNCS and IMPLEMENT SECOND DERIV 
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
#
#
#


#def decomposed_mae_gradient_weighted(
#    pred: torch.Tensor, 
#    target: torch.Tensor, 
#    wet: torch.Tensor,
#    gradient_weight: float = 0.1
#) -> torch.Tensor:
#    """
#    MAE loss with WEIGHTED spatial gradient matching penalty.
#    
#    By controlling gradient_weight,
#    we can balance accuracy (MAE term) vs sharpness (gradient term).
#    
#    Loss = MAE(pred, target) + α * gradient_penalty(pred, target)
#    
#    where α = gradient_weight is a tunable hyperparameter.
#    
#    Recommended starting values:
#    - α = 0.05: Very conservative, prioritize accuracy
#    - α = 0.1:  Conservative, good balance 
#    - α = 0.25: Moderate, more sharpness 
#    - α = 0.5:  Aggressive sharpening
#    - α = 1.0:  Equal weighting 
#    
#    Args:
#        pred: Predicted tensor [batch, channels, height, width]
#        target: Target tensor [batch, channels, height, width]
#        wet: Wet mask [batch, channels, height, width]
#        gradient_weight: Scaling factor α for gradient penalty
#    
#    Returns:
#        Loss per channel [channels]
#    """
#    pred = pred * wet
#    target = target * wet
#    
#    # MAE term (main accuracy objective)
#    mae_loss = F.l1_loss(pred, target, reduction="none")
#    mae_per_channel = mae_loss.mean(dim=(0, 2, 3))
#    
#    # Gradient penalty: Match spatial gradients
#    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
#    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
#    
#    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
#    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
#    
#    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction="none")
#    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction="none")
#    
#    # Average gradient losses
#    grad_loss = (
#        F.pad(grad_loss_y, (0, 0, 0, 1), value=0).mean(dim=(0, 2, 3)) +
#        F.pad(grad_loss_x, (0, 1, 0, 0), value=0).mean(dim=(0, 2, 3))
#    ) / 2
#    
#    # Weighted combination
#    total_loss = mae_per_channel + gradient_weight * grad_loss
#    
#    return total_loss
#
#
def decomposed_mae_gradient_multiscale(
    pred: torch.Tensor,
    target: torch.Tensor, 
    wet: torch.Tensor,
    gradient_weight: float = 0.15,
    scales: list[int] = None
) -> torch.Tensor:
    """
    MAE loss with MULTI-SCALE gradient matching (Tier 2 experiment).
    
    Instead of matching gradients at a single scale, this matches gradient
    statistics at multiple spatial scales. This is inspired by perceptual loss
    and helps preserve both fine-scale fronts and large-scale patterns.
    
    Loss = MAE + α * Σ_scales gradient_penalty_at_scale
    
    This is more sophisticated than simple gradient matching and may better
    preserve the structure of oceanic fronts while maintaining accuracy.
    
    Args:
        pred: Predicted tensor [batch, channels, height, width]
        target: Target tensor [batch, channels, height, width]
        wet: Wet mask [batch, channels, height, width]
        gradient_weight: Scaling factor for total gradient penalty
        scales: List of pooling sizes for multi-scale (default: [1, 2, 4])
    
    Returns:
        Loss per channel [channels]
    """
    if scales is None:
        scales = [1, 2, 4]
    
    pred = pred * wet
    target = target * wet
    
    # MAE term
    mae_loss = F.l1_loss(pred, target, reduction="none")
    mae_per_channel = mae_loss.mean(dim=(0, 2, 3))
    
    # Multi-scale gradient penalty
    total_grad_loss = 0.0
    
    for scale in scales:
        # Downsample if scale > 1
        if scale > 1:
            pred_scaled = F.avg_pool2d(pred, kernel_size=scale, stride=scale)
            target_scaled = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        else:
            pred_scaled = pred
            target_scaled = target
        
        # Compute gradients at this scale
        pred_grad_y = pred_scaled[:, :, 1:, :] - pred_scaled[:, :, :-1, :]
        pred_grad_x = pred_scaled[:, :, :, 1:] - pred_scaled[:, :, :, :-1]
        
        target_grad_y = target_scaled[:, :, 1:, :] - target_scaled[:, :, :-1, :]
        target_grad_x = target_scaled[:, :, :, 1:] - target_scaled[:, :, :, :-1]
        
        # L1 loss on gradients at this scale
        grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction="none")
        grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction="none")
        
        # Average and accumulate
        scale_grad_loss = (
            grad_loss_y.mean(dim=(0, 2, 3)) + grad_loss_x.mean(dim=(0, 2, 3))
        ) / 2
        
        total_grad_loss = total_grad_loss + scale_grad_loss
    
    # Average over scales
    total_grad_loss = total_grad_loss / len(scales)
    
    # Weighted combination
    total_loss = mae_per_channel + gradient_weight * total_grad_loss
    
    return total_loss


## MK : I still need to figure how to implement this loss function to better penalize lack of coherence of fronts 
#def gradient_structure_loss(field_true, field_pred, patch_size=5):
#	"""
#	Penalize loss of coherent gradient structures (fronts).
#    
#	Idea: In coherent fronts, neighboring gradients are aligned.
#	We normalize gradients to unit vectors, then measure how much
#	they cancel when averaged locally. High coherence = aligned gradients.
#	"""
#	from scipy.ndimage import uniform_filter
#    
#	gy_true, gx_true = np.gradient(field_true)
#	gy_pred, gx_pred = np.gradient(field_pred)
#    
#	# Normalize gradients to unit vectors (direction only, not magnitude)
#	mag_true = np.sqrt(gx_true**2 + gy_true**2) + 1e-10  # avoid division by zero
#	gx_true_norm = gx_true / mag_true
#	gy_true_norm = gy_true / mag_true
#    
#	mag_pred = np.sqrt(gx_pred**2 + gy_pred**2) + 1e-10
#	gx_pred_norm = gx_pred / mag_pred
#	gy_pred_norm = gy_pred / mag_pred
#    
#	# Average the normalized gradient components in local patches
#	# If gradients are aligned, components won't cancel
#	# If gradients are random, components cancel out
#	gx_avg_true = uniform_filter(gx_true_norm, size=patch_size)
#	gy_avg_true = uniform_filter(gy_true_norm, size=patch_size)
#	coherence_true = np.sqrt(gx_avg_true**2 + gy_avg_true**2)
#    
#	gx_avg_pred = uniform_filter(gx_pred_norm, size=patch_size)
#	gy_avg_pred = uniform_filter(gy_pred_norm, size=patch_size)
#	coherence_pred = np.sqrt(gx_avg_pred**2 + gy_avg_pred**2)
#    
#	# Loss: match local gradient coherence
#	# coherence near 1 = organized front, near 0 = incoherent/noisy
#	loss = np.mean((coherence_true - coherence_pred)**2)
#    
#	return loss

