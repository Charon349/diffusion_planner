from typing import List
import torch

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.guidance.collision import collision_guidance_fn
from diffusion_planner.utils.traj_kinematics import integrate_ego_velocity

N = 1
sde = VPSDE_linear()

class GuidanceWrapper:
    def __init__(self):
        self._guidance_fns = [
            collision_guidance_fn
        ]

    def __call__(self, x_in, t_input, cond, *args, **kwargs):
        """
        This function is a wrapper for the guidance functions in the model.
        """
        energy = 0
        
        state_normalizer = kwargs["state_normalizer"]
        observation_normalizer = kwargs["observation_normalizer"]
      
        B, P, _ = x_in.shape
        model = kwargs["model"]
        model_condition = kwargs["model_condition"]
        num_ego_slots = int(model_condition.get("num_ego_slots", 1))
        wta_idx = model_condition.get("wta_idx", None)
      
        x_fix = model(x_in, t_input, **model_condition).detach() - x_in.detach()
        x_fix = x_fix.reshape(B, P, -1, 4)
        x_fix[:, :, 0] = 0.0
        x_in = x_in + x_fix.reshape(B, P, -1)
      
        # x_in = torch.repeat_interleave(x_in, N, dim=0) # [B * N, P, T, 4]
        # t_input = torch.repeat_interleave(t_input, N, dim=0) # [B * N]        
        # kwargs["inputs"] = {k: torch.repeat_interleave(v, N, dim=0) for k, v in kwargs["inputs"].items()}
      
        # sigma_t = sde.marginal_prob_std(t_input)
        # sigma_t = sigma_t / torch.sqrt(1 + sigma_t ** 2)
        # x_in = torch.cat([x_in[:, :1] + sigma_t[:, None, None] * torch.randn_like(x_in[:, :1]), x_in[:, 1:]], dim=1)
      
        raw_inputs = observation_normalizer.inverse(kwargs["inputs"])
        x_in = state_normalizer.inverse(x_in.reshape(B, P, -1, 4))
        if num_ego_slots > 1:
            if wta_idx is None:
                wta_idx = torch.zeros(B, dtype=torch.long, device=x_in.device)
            batch_idx = torch.arange(B, device=x_in.device)
            selected_ego = x_in[batch_idx, wta_idx.to(x_in.device)]
            x_in = torch.cat([selected_ego[:, None], x_in[:, num_ego_slots:]], dim=1)
            P = x_in.shape[1]
        x_in = torch.cat([x_in[:, :, :1], integrate_ego_velocity(x_in[:, :, 1:])], dim=2)
        x_in[:, 0, 0, :4] = raw_inputs["ego_current_state"][:, :4]
        x_in[:, 1:, 0, :4] = raw_inputs["neighbor_agents_past"][:, :P - 1, -1, :4]
      
        for guidance_fn in self._guidance_fns:
            energy += guidance_fn(x_in, t_input, cond, **kwargs)
        # energy1 = self._guidance_fns[0](x_in, t_input, cond, **kwargs)
        # energy2 = self._guidance_fns[1](x_in, t_input, cond, **kwargs)
        
        # energy = energy1 if energy2 < 1 else energy2
        
        assert not torch.isnan(energy).any()
          
        return energy
