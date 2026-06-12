from typing import Optional

import numpy as np
import torch


def _heading_to_cos_sin(heading: torch.Tensor) -> torch.Tensor:
    return torch.stack([heading.cos(), heading.sin()], dim=-1)


def anchor_to_delta_state(anchor: torch.Tensor, state_format: str) -> torch.Tensor:
    """
    Convert ego trajectory anchors to Diffusion-Planner's ego target state.

    The ego diffusion target in this codebase is [delta_x, delta_y, cos_h, sin_h],
    not absolute [x, y, heading]. Anchor files may be saved in either format.
    """
    if anchor.ndim != 3:
        raise ValueError(f"Expected anchor shape [K, T, D], got {tuple(anchor.shape)}")

    if state_format == "delta":
        if anchor.shape[-1] != 4:
            raise ValueError("Delta anchors must have shape [K, T, 4]")
        return anchor.float()

    if state_format == "absolute":
        if anchor.shape[-1] not in (3, 4):
            raise ValueError("Absolute anchors must have shape [K, T, 3] or [K, T, 4]")
        xy = anchor[..., :2]
        delta_xy = torch.diff(
            torch.cat([torch.zeros_like(xy[:, :1]), xy], dim=1),
            dim=1,
        )
        if anchor.shape[-1] == 3:
            heading = _heading_to_cos_sin(anchor[..., 2])
        else:
            heading = anchor[..., 2:4]
        return torch.cat([delta_xy, heading], dim=-1).float()

    raise ValueError(f"Unknown ego_anchor_state_format: {state_format}")


def load_ego_anchors(
    anchor_path: Optional[str],
    future_len: int,
    num_anchors: int = 0,
    state_format: str = "delta",
) -> Optional[torch.Tensor]:
    if not anchor_path:
        return None

    if anchor_path.endswith(".pt") or anchor_path.endswith(".pth"):
        anchor = torch.load(anchor_path, map_location="cpu")
    else:
        anchor = torch.from_numpy(np.load(anchor_path))

    if isinstance(anchor, dict):
        for key in ("anchors", "ego_anchors", "plan_anchor"):
            if key in anchor:
                anchor = anchor[key]
                break
        else:
            raise ValueError(f"Anchor dict at {anchor_path} does not contain a known anchor key")

    anchor = torch.as_tensor(anchor, dtype=torch.float32)
    anchor = anchor_to_delta_state(anchor, state_format)

    if anchor.shape[1] != future_len:
        raise ValueError(
            f"Anchor horizon mismatch: anchor T={anchor.shape[1]}, expected future_len={future_len}"
        )

    if num_anchors and num_anchors > 0:
        anchor = anchor[:num_anchors]

    if anchor.shape[0] == 0:
        raise ValueError("No ego anchors loaded")

    return anchor.contiguous()


def normalize_ego_anchor(anchor: torch.Tensor, state_normalizer) -> torch.Tensor:
    mean = state_normalizer.mean[0].to(anchor.device)
    std = state_normalizer.std[0].to(anchor.device)
    return (anchor - mean) / std


def normalize_neighbor_future(neighbor_future: torch.Tensor, state_normalizer) -> torch.Tensor:
    mean = state_normalizer.mean[1:1 + neighbor_future.shape[1]].to(neighbor_future.device)
    std = state_normalizer.std[1:1 + neighbor_future.shape[1]].to(neighbor_future.device)
    return (neighbor_future - mean[None]) / std[None]
