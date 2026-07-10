import torch

from ocean_emulators.constants import TensorMap


def get_depth_loss_dict(
    label: str, loss_per_channel: torch.Tensor
) -> dict[str, torch.Tensor]:
    tensor_map = TensorMap.get_instance()
    metrics = {}
    for depth in tensor_map.DEPTH_SET:
        # PCA keys have component suffixes (e.g. 0-19) narrower than the
        # 50-level DEPTH_SET assumed for *_all keys, leaving empty buckets
        # whose .mean() is NaN — skip them instead of logging NaN to W&B
        # (audit finding 8).
        if tensor_map.DP_3D_IDX[depth].numel() == 0:
            continue
        metrics[f"{label}/loss/depth/depth_{depth}_loss"] = loss_per_channel[
            tensor_map.DP_3D_IDX[depth]
        ].mean()
    return metrics


def get_variable_loss_dict(
    label: str, loss_per_channel: torch.Tensor
) -> dict[str, torch.Tensor]:
    tensor_map = TensorMap.get_instance()
    metrics = {}
    for variable in tensor_map.VAR_SET:
        metrics[f"{label}/loss/variable/{variable}_loss"] = loss_per_channel[
            tensor_map.VAR_3D_IDX[variable]
        ].mean()
    return metrics


def get_channel_loss_dict(
    label: str, loss_per_channel: torch.Tensor, loss_name: str = "loss"
) -> dict[str, torch.Tensor]:
    return get_channel_dict(label, loss_name, loss_per_channel)


def get_channel_loss_scale_dict(
    label: str, loss_scale_per_channel: torch.Tensor
) -> dict[str, torch.Tensor]:
    return get_channel_dict(label, "loss_scale", loss_scale_per_channel)


def get_channel_dict(
    prefix: str, measure: str, per_channel: torch.Tensor
) -> dict[str, torch.Tensor]:
    tensor_map = TensorMap.get_instance()
    metrics = {}
    for i, channel in enumerate(tensor_map.prognostic_var_names):
        metrics[f"{prefix}/{measure}/channel/{channel}_{measure}"] = per_channel[i]
    return metrics
