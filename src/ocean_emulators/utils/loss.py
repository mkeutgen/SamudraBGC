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



def gradient_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    gradient_weight: float = 1.0,
) -> torch.Tensor:
    """
    MAE on values + gradients. Stable for long rollouts.
    
    Same principle as gradient_mse, but L1 norm for stability:
    - More robust to outliers
    - Stable gradients (sign-based, not magnitude-based)
    - Better for autoregressive prediction
    
    Use for: Long rollouts (>7 days), autoregressive prediction
    """
    pred = pred * wet
    target = target * wet
    
    # Value MAE (L1)
    value_loss = F.l1_loss(pred, target, reduction='none')
    
    # Gradient MAE (L1 on derivatives)
    gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    gx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    
    gy_loss = F.l1_loss(gy_pred, gy_target, reduction='none')
    gx_loss = F.l1_loss(gx_pred, gx_target, reduction='none')
    
    # Pad and combine
    gy_loss = F.pad(gy_loss, (0, 0, 0, 1), value=0)
    gx_loss = F.pad(gx_loss, (0, 1, 0, 0), value=0)
    gradient_loss = (gy_loss + gx_loss) / 2.0
    
    return (value_loss + gradient_weight * gradient_loss).mean(dim=(0, 2, 3))


# ============================================================================
# LAPLACIAN VERSIONS (still not tested)
# ============================================================================

def laplacian_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    laplacian_weight: float = 0.5,
) -> torch.Tensor:
    """
    MAE on values + Laplacian (second derivatives).
    Matches curvature → maximum sharpness.
    
    Use if gradient_mae still too smooth.
    """
    pred = pred * wet
    target = target * wet
    
    # Value MAE
    value_loss = F.l1_loss(pred, target, reduction='none')
    
    # Laplacian (∇² = ∂²/∂x² + ∂²/∂y²)
    def laplacian(field):
        f_yy = field[:, :, 2:, :] - 2*field[:, :, 1:-1, :] + field[:, :, :-2, :]
        f_xx = field[:, :, :, 2:] - 2*field[:, :, :, 1:-1] + field[:, :, :, :-2]
        h_min = min(f_yy.shape[2], f_xx.shape[2])
        w_min = min(f_yy.shape[3], f_xx.shape[3])
        return f_yy[:, :, :h_min, :w_min] + f_xx[:, :, :h_min, :w_min]
    
    lap_pred = laplacian(pred)
    lap_target = laplacian(target)
    lap_loss = F.l1_loss(lap_pred, lap_target, reduction='none')
    
    # Pad back
    pad_h = (pred.shape[2] - lap_loss.shape[2]) // 2
    pad_w = (pred.shape[3] - lap_loss.shape[3]) // 2
    lap_loss = F.pad(lap_loss, (pad_w, pad_w, pad_h, pad_h), value=0)
    
    return (value_loss + laplacian_weight * lap_loss).mean(dim=(0, 2, 3))


def gradient_laplacian_mae_combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    gradient_weight: float = 1.0,
    laplacian_weight: float = 0.3,
) -> torch.Tensor:
    """
    MAE on values + gradients + Laplacian.
    Most comprehensive: stability + phase coherence + maximum sharpness.
    
    Use for: Long rollouts + need very sharp features
    """
    grad_loss = gradient_mae_loss(pred, target, wet, gradient_weight=1.0)
    lapl_loss = laplacian_mae_loss(pred, target, wet, laplacian_weight=1.0)
    return grad_loss + laplacian_weight * lapl_loss




# MK : I still need to figure how to implement this loss function to better penalize lack of coherence of fronts 
def gradient_structure_loss(field_true, field_pred, patch_size=5):
	"""
	Penalize loss of coherent gradient structures (fronts).
    
	Idea: In coherent fronts, neighboring gradients are aligned.
	We normalize gradients to unit vectors, then measure how much
	they cancel when averaged locally. High coherence = aligned gradients.
	"""
	from scipy.ndimage import uniform_filter
    
	gy_true, gx_true = np.gradient(field_true)
	gy_pred, gx_pred = np.gradient(field_pred)
    
	# Normalize gradients to unit vectors (direction only, not magnitude)
	mag_true = np.sqrt(gx_true**2 + gy_true**2) + 1e-10  # avoid division by zero
	gx_true_norm = gx_true / mag_true
	gy_true_norm = gy_true / mag_true
    
	mag_pred = np.sqrt(gx_pred**2 + gy_pred**2) + 1e-10
	gx_pred_norm = gx_pred / mag_pred
	gy_pred_norm = gy_pred / mag_pred
    
	# Average the normalized gradient components in local patches
	# If gradients are aligned, components won't cancel
	# If gradients are random, components cancel out
	gx_avg_true = uniform_filter(gx_true_norm, size=patch_size)
	gy_avg_true = uniform_filter(gy_true_norm, size=patch_size)
	coherence_true = np.sqrt(gx_avg_true**2 + gy_avg_true**2)
    
	gx_avg_pred = uniform_filter(gx_pred_norm, size=patch_size)
	gy_avg_pred = uniform_filter(gy_pred_norm, size=patch_size)
	coherence_pred = np.sqrt(gx_avg_pred**2 + gy_avg_pred**2)
    
	# Loss: match local gradient coherence
	# coherence near 1 = organized front, near 0 = incoherent/noisy
	loss = np.mean((coherence_true - coherence_pred)**2)
    
	return loss

