from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_planner.utils.normalizer import StateNormalizer
from diffusion_planner.utils.anchors import load_ego_anchors, normalize_ego_anchor
from diffusion_planner.utils.traj_kinematics import integrate_ego_velocity


# # Keep the original trajectory L2 as the main objective, then add small ego-only
# # heading regularizers to reduce post-turn heading oscillation.
DEFAULT_HEADING_DIM_WEIGHT = 1.0
DEFAULT_EGO_HEADING_LOSS_WEIGHT = 0.2
DEFAULT_UNIT_CIRCLE_LOSS_WEIGHT = 0.05
DEFAULT_KINEMATIC_LOSS_WEIGHT = 0.1
DEFAULT_KINEMATIC_SPEED_THRESHOLD = 0.1


def py_sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    return loss.mean()


def soft_label_cross_entropy(logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=1)
    return -(target_prob * log_prob).sum(dim=1).mean()


def _ego_future_to_target_state(ego_future: torch.Tensor) -> torch.Tensor:
    ego_future_vel = torch.diff(
        torch.cat([torch.zeros_like(ego_future[:, :1, :], device=ego_future.device), ego_future], dim=-2),
        dim=-2
    )
    ego_future_vel[..., 2:] = ego_future[..., 2:]
    return ego_future_vel


def _repeat_tensor_dict(data: Dict[str, torch.Tensor], repeat: int) -> Dict[str, torch.Tensor]:
    repeated = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.ndim > 0:
            repeated[key] = value.repeat_interleave(repeat, dim=0)
        else:
            repeated[key] = value
    return repeated


def _get_ego_anchors(anchor_args, device: torch.device, future_len: int) -> torch.Tensor:
    anchors = getattr(anchor_args, "_ego_anchor_tensor", None)
    if anchors is None:
        anchors = load_ego_anchors(
            getattr(anchor_args, "ego_anchor_path", None),
            future_len=future_len,
            num_anchors=getattr(anchor_args, "num_ego_anchors", 0),
            state_format=getattr(anchor_args, "ego_anchor_state_format", "delta"),
        )
        if anchors is None:
            raise ValueError("--use_ego_anchor requires --ego_anchor_path")
        setattr(anchor_args, "_ego_anchor_tensor", anchors)
    return anchors.to(device)


def _anchor_distances(anchors: torch.Tensor, ego_future: torch.Tensor, w_lat: float = 1.0, w_lon: float = 0.2) -> torch.Tensor:
    anchor_xy = torch.cumsum(anchors[..., :2], dim=1)
    diff_xy = anchor_xy[None] - ego_future[:, None, :, :2]
    
    # ego_future[..., 2] is cos(heading), ego_future[..., 3] is sin(heading)
    gt_cos = ego_future[:, None, :, 2]
    gt_sin = ego_future[:, None, :, 3]
    
    # Lateral error: projection onto normal vector (-sin, cos)
    lat_err = torch.abs(diff_xy[..., 0] * (-gt_sin) + diff_xy[..., 1] * gt_cos)
    
    # Longitudinal error: projection onto heading vector (cos, sin)
    lon_err = torch.abs(diff_xy[..., 0] * gt_cos + diff_xy[..., 1] * gt_sin)
    
    dist = (w_lat * lat_err + w_lon * lon_err).mean(dim=-1)
    return dist


def _closest_anchor_indices(anchors: torch.Tensor, ego_future: torch.Tensor) -> torch.Tensor:
    return _anchor_distances(anchors, ego_future).argmin(dim=1)


# def heading_alignment_loss(
#     pred_future: torch.Tensor,
#     gt_future: torch.Tensor,
#     valid_mask: torch.Tensor = None,
# ) -> torch.Tensor:
#     """
#     Penalize heading mismatch using normalized cos/sin heading vectors.

#     Args:
#         pred_future: Predicted future states [..., 4] in unnormalized coordinates.
#         gt_future: Ground-truth future states [..., 4] in unnormalized coordinates.
#         valid_mask: Optional boolean mask for valid timesteps, matching pred_future[..., 0].
#     """
#     pred_heading = pred_future[..., 2:4]
#     gt_heading = gt_future[..., 2:4]

#     pred_heading = pred_heading / pred_heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
#     gt_heading = gt_heading / gt_heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)

#     cosine = torch.sum(pred_heading * gt_heading, dim=-1).clamp(-1.0, 1.0)
#     loss = 1.0 - cosine

#     if valid_mask is not None:
#         loss = loss[valid_mask]

#     if loss.numel() == 0:
#         return torch.tensor(0.0, device=pred_future.device)

#     return loss.mean()


# def unit_circle_loss(pred_future: torch.Tensor) -> torch.Tensor:
#     """Keep predicted ego heading as a valid cos/sin vector."""
#     ego_heading = pred_future[:, 0, :, 2:4]
#     heading_norm_sq = torch.sum(ego_heading ** 2, dim=-1)
#     return ((heading_norm_sq - 1.0) ** 2).mean()


# def kinematic_heading_loss(
#     pred_future: torch.Tensor,
#     speed_threshold: float = DEFAULT_KINEMATIC_SPEED_THRESHOLD,
# ) -> torch.Tensor:
#     """Align ego heading with the direction of predicted motion when the ego is moving."""
#     ego_future = pred_future[:, 0]
#     dx = ego_future[:, 1:, 0] - ego_future[:, :-1, 0]
#     dy = ego_future[:, 1:, 1] - ego_future[:, :-1, 1]
#     displacement = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)

#     vel_cos = dx / displacement
#     vel_sin = dy / displacement

#     pred_heading = ego_future[:, :-1, 2:4]
#     pred_heading = pred_heading / pred_heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)

#     cos_sim = vel_cos * pred_heading[..., 0] + vel_sin * pred_heading[..., 1]
#     loss = 1.0 - cos_sim.clamp(-1.0, 1.0)

#     moving_mask = displacement > speed_threshold
#     if not moving_mask.any():
#         return torch.tensor(0.0, device=pred_future.device)

#     return loss[moving_mask].mean()


def anchored_diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    sde,
    futures: Tuple[torch.Tensor, torch.Tensor],
    norm: StateNormalizer,
    loss: Dict[str, Any],
    model_type: str,
    supervision_type: str,
    planning_hybrid_loss: float,
    anchor_args,
):
    if supervision_type != "x_start":
        raise ValueError("Ego-anchor training currently expects --diffusion_supervision_type x_start")

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask

    B, Pn, T, _ = neighbors_future.shape
    device = ego_future.device
    anchors = _get_ego_anchors(anchor_args, device, T)
    K = anchors.shape[0]

    ego_current = inputs["ego_current_state"][:, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    ego_future_vel = _ego_future_to_target_state(ego_future)
    gt_future = torch.cat([ego_future_vel[:, None, :, :], neighbors_future], dim=1)
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
    P = gt_future.shape[1]

    w_lat = getattr(anchor_args, "anchor_w_lat", 1.0)
    w_lon = getattr(anchor_args, "anchor_w_lon", 0.2)
    anchor_dist = _anchor_distances(anchors, ego_future, w_lat, w_lon)
    pos_k = anchor_dist.argmin(dim=1)
    top_anchor_dist = torch.topk(anchor_dist, k=min(2, K), dim=1, largest=False).values
    best_anchor_dist = top_anchor_dist[:, 0]
    if K > 1:
        anchor_margin = top_anchor_dist[:, 1] - top_anchor_dist[:, 0]
    else:
        anchor_margin = torch.zeros_like(best_anchor_dist)

    t_min = getattr(anchor_args, "ego_anchor_t_min", 1e-3)
    t_max = getattr(anchor_args, "ego_anchor_t_max", 0.2)
    t = torch.rand(B, device=device) * (t_max - t_min) + t_min

    norm_gt_future = norm(gt_future)
    norm_gt_ego = norm_gt_future[:, :1].expand(B, K, T, 4)
    norm_gt_neighbors = norm_gt_future[:, 1:]
    norm_gt_future_expanded = torch.cat([norm_gt_ego, norm_gt_neighbors], dim=1)

    gt_ego = ego_future_vel[:, None].expand(B, K, T, 4)
    gt_future = torch.cat([gt_ego, neighbors_future], dim=1)
    current_ego = ego_current[:, None].expand(B, K, 4)
    current_states = torch.cat([current_ego, neighbors_current], dim=1)

    all_gt = torch.cat([current_states[:, :, None, :], norm_gt_future_expanded], dim=2)
    all_gt[:, K:][neighbor_mask] = 0.0

    mean, std = sde.marginal_prob(all_gt[..., 1:, :], t)

    anchor_norm = normalize_ego_anchor(anchors, norm)
    anchor_norm = anchor_norm[None].expand(B, K, T, 4)
    ego_anchor_mean, _ = sde.marginal_prob(anchor_norm, t)
    mean[:, :K] = ego_anchor_mean

    z = torch.randn_like(gt_future, device=device)
    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    x_t = xT[:, :, 1:, :]

    with torch.no_grad():
        clean_xT = torch.cat([all_gt[:, :, :1, :], all_gt[:, :, 1:, :]], dim=2)
        clean_xT[:, :K, 1:] = anchor_norm
    clean_t = torch.full((B,), getattr(anchor_args, "ego_anchor_t_min", 1e-3), device=device)

    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
        "anchor_branch_count": K,
        "wta_idx": pos_k,
        "clean_sampled_trajectories": clean_xT,
        "clean_diffusion_time": clean_t,
    }

    encoder_outputs, decoder_output = model(merged_inputs)
    score = decoder_output["score"][:, :, 1:, :]
    pred_x_start = sde.transform(f"{model_type}->x_start", score, t, x_t)

    diff = (pred_x_start - all_gt[:, :, 1:, :]) ** 2
    dim_weight = torch.tensor(
        [1.0, 1.0, DEFAULT_HEADING_DIM_WEIGHT, DEFAULT_HEADING_DIM_WEIGHT],
        device=diff.device,
        dtype=diff.dtype,
    )
    dpm_loss = torch.sum(diff * dim_weight, dim=-1)

    batch_idx = torch.arange(B, device=device)
    pos_ego_loss = dpm_loss[batch_idx, pos_k]

    neighbor_loss_all = dpm_loss[:, K:]
    neighbor_pos_loss = neighbor_loss_all[neighbors_future_valid]
    if neighbor_pos_loss.numel() > 0:
        neighbor_pos_loss = neighbor_pos_loss.mean()
    else:
        neighbor_pos_loss = torch.tensor(0.0, device=device)

    neighbor_all_loss = torch.tensor(0.0, device=device)

    neighbor_all_weight = getattr(anchor_args, "anchor_neighbor_all_loss", 0.1)
    loss["neighbor_prediction_pos_loss"] = neighbor_pos_loss
    loss["neighbor_prediction_all_loss"] = neighbor_all_loss
    loss["neighbor_prediction_loss"] = neighbor_pos_loss + neighbor_all_weight * neighbor_all_loss

    ego_planning_diffusion_loss = pos_ego_loss.mean()

    pred_target_future = norm.inverse(pred_x_start)
    selected_ego = pred_target_future[batch_idx, pos_k]
    selected_future = torch.cat([selected_ego[:, None], pred_target_future[:, K:]], dim=1)
    pred_future = integrate_ego_velocity(selected_future, detach_window_size=10)
    loss["ego_planning_hybrid_loss"] = torch.sum(
        (pred_future[:, 0, :, :2] - ego_future[..., :2]) ** 2,
        dim=-1,
    ).mean()
    loss["ego_planning_diffusion_loss"] = ego_planning_diffusion_loss

    score_loss = torch.tensor(0.0, device=device)
    if "clean_branch_logits" in decoder_output and decoder_output["clean_branch_logits"] is not None:
        branch_logits = decoder_output["clean_branch_logits"].reshape(B, K)
        score_loss_type = getattr(anchor_args, "anchor_score_loss_type", "focal")
        if score_loss_type == "focal":
            target_classes_onehot = torch.zeros_like(branch_logits)
            target_classes_onehot.scatter_(1, pos_k.unsqueeze(1), 1)
            score_loss = py_sigmoid_focal_loss(branch_logits, target_classes_onehot)
        elif score_loss_type == "ce":
            score_loss = F.cross_entropy(branch_logits, pos_k)
        elif score_loss_type == "soft_ce":
            tau = getattr(anchor_args, "anchor_score_soft_label_tau", 1.0)
            if tau <= 0:
                raise ValueError("--anchor_score_soft_label_tau must be > 0 when using soft_ce")
            with torch.no_grad():
                target_prob = torch.softmax(-anchor_dist.detach() / tau, dim=1)
            score_loss = soft_label_cross_entropy(branch_logits, target_prob)
            with torch.no_grad():
                loss["anchor_soft_target_entropy"] = (
                    -(target_prob * torch.log(target_prob.clamp_min(1e-12))).sum(dim=1).mean()
                )
                loss["anchor_soft_positive_prob"] = target_prob[batch_idx, pos_k].mean()
        else:
            raise ValueError(f"Unknown anchor_score_loss_type: {score_loss_type}")
        with torch.no_grad():
            topk = min(3, K)
            predicted_topk = torch.topk(branch_logits, k=topk, dim=1).indices
            loss["anchor_top1_acc"] = (predicted_topk[:, 0] == pos_k).float().mean()
            loss["anchor_top3_acc"] = (predicted_topk == pos_k[:, None]).any(dim=1).float().mean()

            positive_logits = branch_logits[batch_idx, pos_k]
            loss["positive_logit_mean"] = positive_logits.mean()
            if K > 1:
                negative_mask = torch.ones_like(branch_logits, dtype=torch.bool)
                negative_mask[batch_idx, pos_k] = False
                loss["negative_logit_mean"] = branch_logits[negative_mask].mean()
            else:
                loss["negative_logit_mean"] = torch.tensor(0.0, device=device)
    loss["anchor_score_loss"] = score_loss

    loss["ego_planning_loss"] = (
        ego_planning_diffusion_loss
        + planning_hybrid_loss * loss["ego_planning_hybrid_loss"]
        + getattr(anchor_args, "anchor_score_loss", 0.1) * score_loss
    )
    loss["anchor_positive_index"] = pos_k.float().mean()
    loss["anchor_margin"] = anchor_margin.mean()
    loss["anchor_best_distance"] = best_anchor_dist.mean()
    if K > 1:
        loss["anchor_second_best_distance"] = top_anchor_dist[:, 1].mean()
    else:
        loss["anchor_second_best_distance"] = torch.tensor(0.0, device=device)
    with torch.no_grad():
        pos_hist = torch.bincount(pos_k, minlength=K).float() / max(B, 1)
        for anchor_idx, anchor_ratio in enumerate(pos_hist):
            loss[f"anchor_pos_hist_{anchor_idx:02d}"] = anchor_ratio

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    sde,

    futures: Tuple[torch.Tensor, torch.Tensor],
    
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    supervision_type: str = None,
    planning_hybrid_loss: float = 0.01,
    anchor_args=None,
    eps: float = 1e-3,
):   
    supervision_type = supervision_type if supervision_type is not None else model_type
    if anchor_args is not None and getattr(anchor_args, "use_ego_anchor", False):
        return anchored_diffusion_loss_func(
            model,
            inputs,
            sde,
            futures,
            norm,
            loss,
            model_type,
            supervision_type,
            planning_hybrid_loss,
            anchor_args,
        )

    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask # [B, P, V]

    B, Pn, T, _ = neighbors_future.shape
    ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    ego_future_vel = torch.diff(
        torch.cat([torch.zeros_like(ego_future[:, :1, :], device=ego_future.device), ego_future], dim=-2),
        dim=-2
    )
    ego_future_vel[..., 2:] = ego_future[..., 2:]

    gt_future = torch.cat([ego_future_vel[:, None, :, :], neighbors_future[..., :]], dim=1) # [B, P = 1 + neighbor, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1) # [B, P, 4]

    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps # [B,]
    z = torch.randn_like(gt_future, device=gt_future.device) # [B, P, T, 4]
    
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = sde.marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    x_t = xT[:, :, 1:, :]
    
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs) # [B, P, 1 + T, 4]
    score = decoder_output["score"][:, :, 1:, :] # [B, P, T, 4]

    pred = sde.transform(f"{model_type}->{supervision_type}", score, t, x_t)

    if supervision_type == "score":
        dpm_loss = torch.sum((pred * std + z)**2, dim=-1)
    elif supervision_type == "x_start":
        diff = (pred - all_gt[:, :, 1:, :]) ** 2
        dim_weight = torch.tensor(
            [1.0, 1.0, DEFAULT_HEADING_DIM_WEIGHT, DEFAULT_HEADING_DIM_WEIGHT],
            device=diff.device,
            dtype=diff.dtype,
        )
        dpm_loss = torch.sum(diff * dim_weight, dim=-1)
    elif supervision_type == "noise":
        dpm_loss = torch.sum((pred - z)**2, dim=-1)
    elif supervision_type == "v":
        v = sde.transform("noise->v", z, t, x_t)
        dpm_loss = torch.sum((pred - v)**2, dim=-1)
    else:
        raise ValueError(f"Unknown supervision type: {supervision_type}")

    pred_x_start = sde.transform(f"{model_type}->x_start", score, t, x_t)
    
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    ego_planning_diffusion_loss = dpm_loss[:, 0, :].mean()

    pred_target_future = norm.inverse(pred_x_start)
    pred_future = integrate_ego_velocity(pred_target_future, detach_window_size=10)
    loss["ego_planning_hybrid_loss"] = torch.sum(
        (pred_future[:, 0, :, :2] - ego_future[..., :2])**2,
        dim=-1
    ).mean()
    loss["ego_planning_diffusion_loss"] = ego_planning_diffusion_loss
    loss["ego_planning_loss"] = (
        ego_planning_diffusion_loss
        + planning_hybrid_loss * loss["ego_planning_hybrid_loss"]
    )

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output
