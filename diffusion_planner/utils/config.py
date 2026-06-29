import json
import torch

from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer


class Config:
    
    def __init__(
            self,
            args_file,
    ):
        with open(args_file, 'r') as f:
            args_dict = json.load(f)
            
        for key, value in args_dict.items():
            setattr(self, key, value)
        self.diffusion_supervision_type = getattr(self, "diffusion_supervision_type", self.diffusion_model_type)
        self.planning_hybrid_loss = getattr(self, "planning_hybrid_loss", 0.01)
        self.use_ego_anchor = getattr(self, "use_ego_anchor", False)
        self.ego_anchor_path = getattr(self, "ego_anchor_path", None)
        self.ego_anchor_state_format = getattr(self, "ego_anchor_state_format", "delta")
        self.num_ego_anchors = getattr(self, "num_ego_anchors", 0)
        self.ego_anchor_t_min = getattr(self, "ego_anchor_t_min", 1e-3)
        self.ego_anchor_t_max = getattr(self, "ego_anchor_t_max", 0.2)
        self.anchor_neighbor_all_loss = getattr(self, "anchor_neighbor_all_loss", 0.1)
        self.anchor_score_loss = getattr(self, "anchor_score_loss", 0.1)
        self.anchor_score_loss_type = getattr(self, "anchor_score_loss_type", "focal")
        self.anchor_score_soft_label_tau = getattr(self, "anchor_score_soft_label_tau", 1.0)
        self.use_anchor_score = getattr(self, "use_anchor_score", True)
        self.anchor_sampling_t_start = getattr(self, "anchor_sampling_t_start", 0.2)
        self.anchor_sampling_steps = getattr(self, "anchor_sampling_steps", 10)
        self.anchor_neighbor_init = getattr(self, "anchor_neighbor_init", "cv")
        self.state_normalizer = StateNormalizer(self.state_normalizer['mean'], self.state_normalizer['std'])
        self.observation_normalizer = ObservationNormalizer({
            k: {
                'mean': torch.as_tensor(v['mean']),
                'std': torch.as_tensor(v['std'])
            } for k, v in self.observation_normalizer.items()
        })
