import torch


def detached_integral(u, detach_window_size):
    cum_detach = torch.cumsum(u.detach(), dim=-2)
    cum_normal = torch.cumsum(u, dim=-2)

    shifted = torch.roll(cum_normal, shifts=detach_window_size, dims=-2)
    shifted[..., :detach_window_size, :] = 0
    sum_recent = cum_normal - shifted

    cum_detach_shifted = torch.roll(cum_detach, shifts=detach_window_size, dims=-2)
    cum_detach_shifted[..., :detach_window_size, :] = 0

    return cum_detach_shifted + sum_recent


def integrate_ego_velocity(target_future, detach_window_size=None):
    ego_velocity = target_future[:, 0]
    if detach_window_size is None:
        #
        ego_xy = torch.cumsum(ego_velocity[..., :2], dim=-2)
    else:
        ego_xy = detached_integral(ego_velocity[..., :2], detach_window_size)
    ego_future = torch.cat([ego_xy, ego_velocity[..., 2:]], dim=-1)

    return torch.cat([ego_future[:, None], target_future[:, 1:]], dim=1)
