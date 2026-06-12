import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer
from diffusion_planner.utils.anchors import load_ego_anchors, normalize_ego_anchor, normalize_neighbor_future
from diffusion_planner.utils.traj_kinematics import integrate_ego_velocity


def repeat_tensor_dict(data, repeat):
    repeated = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.ndim > 0:
            repeated[key] = value.repeat_interleave(repeat, dim=0)
        else:
            repeated[key] = value
    return repeated


def integrate_ego_modes(target_future, num_ego_slots):
    if num_ego_slots <= 1:
        return integrate_ego_velocity(target_future)

    B, P, T, D = target_future.shape
    ego_future = target_future[:, :num_ego_slots].reshape(B * num_ego_slots, 1, T, D)
    ego_future = integrate_ego_velocity(ego_future).reshape(B, num_ego_slots, T, D)
    return torch.cat([ego_future, target_future[:, num_ego_slots:]], dim=1)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()
        self._use_ego_anchor = getattr(config, "use_ego_anchor", False)
        self._anchor_sampling_t_start = getattr(config, "anchor_sampling_t_start", 0.2)
        self._anchor_sampling_steps = getattr(config, "anchor_sampling_steps", 10)
        self._anchor_neighbor_init = getattr(config, "anchor_neighbor_init", "cv")

        self.dit = DiT(
            sde=self._sde, 
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth, 
            output_dim= (config.future_len + 1) * 4, # x, y, cos, sin
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type,
            use_anchor_score=self._use_ego_anchor and getattr(config, "use_anchor_score", True),
        )
        
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        
        self._guidance_fn = config.guidance_fn

        anchors = None
        if self._use_ego_anchor:
            anchors = load_ego_anchors(
                getattr(config, "ego_anchor_path", None),
                future_len=config.future_len,
                num_anchors=getattr(config, "num_ego_anchors", 0),
                state_format=getattr(config, "ego_anchor_state_format", "delta"),
            )
        if anchors is None:
            anchors = torch.empty(0, config.future_len, 4)
        self.register_buffer("_ego_anchors", anchors, persistent=False)
        
    @property
    def sde(self):
        return self._sde

    def _neighbor_cv_prior(self, inputs, batch_size, neighbor_count, device):
        raw_inputs = self._observation_normalizer.inverse(inputs)
        neighbor_current = raw_inputs["neighbor_agents_past"][:, :neighbor_count, -1]
        dt = torch.arange(1, self._future_len + 1, device=device, dtype=neighbor_current.dtype) * 0.1
        xy = neighbor_current[..., None, :2] + neighbor_current[..., None, 4:6] * dt[None, None, :, None]
        heading = neighbor_current[..., None, 2:4].expand(batch_size, neighbor_count, self._future_len, 2)
        prior = torch.cat([xy, heading], dim=-1)
        current_mask = torch.sum(torch.ne(neighbor_current[..., :4], 0), dim=-1) == 0
        prior[current_mask] = 0.0
        return normalize_neighbor_future(prior, self._state_normalizer)

    def _anchored_inference(
        self,
        inputs,
        current_states,
        ego_neighbor_encoding,
        route_lanes,
        neighbor_current_mask,
    ):
        B, P, _ = current_states.shape
        device = current_states.device
        if self._ego_anchors.numel() == 0:
            raise ValueError("Anchored inference requires ego anchors. Set ego_anchor_path in args/config.")

        K = self._ego_anchors.shape[0]
        t_start = torch.full((B,), self._anchor_sampling_t_start, device=device)
        ego_anchor = normalize_ego_anchor(self._ego_anchors.to(device), self._state_normalizer)
        ego_anchor = ego_anchor[None].expand(B, K, self._future_len, 4)
        ego_mean, ego_std = self._sde.marginal_prob(ego_anchor, t_start)
        ego_noisy = ego_mean + ego_std * torch.randn_like(ego_anchor)

        if self._anchor_neighbor_init == "cv":
            neighbor_prior = self._neighbor_cv_prior(inputs, B, P - 1, device)
            neighbor_mean, neighbor_std = self._sde.marginal_prob(neighbor_prior, t_start)
            neighbor_noisy = neighbor_mean + neighbor_std * torch.randn_like(neighbor_prior)
        else:
            neighbor_prior = torch.zeros(B, P - 1, self._future_len, 4, device=device)
            neighbor_noisy = torch.randn_like(neighbor_prior) * 0.5

        current_states = torch.cat(
            [current_states[:, :1].expand(B, K, 4), current_states[:, 1:]],
            dim=1,
        )
        xT = torch.cat([ego_noisy, neighbor_noisy], dim=1)
        xT = torch.cat([current_states[:, :, None], xT], dim=2).reshape(B, K + P - 1, -1)

        branch_logits = None
        best_k = None
        if getattr(self.dit, "use_anchor_score", False):
            score_t = torch.full((B,), 1e-3, device=device)
            score_future = torch.cat([ego_anchor, neighbor_prior], dim=1)
            score_state = torch.cat([current_states[:, :, None], score_future], dim=2).reshape(B, K + P - 1, -1)
            _, branch_logits = self.dit.forward_with_logits(
                score_state,
                score_t,
                ego_neighbor_encoding,
                route_lanes,
                neighbor_current_mask,
                num_ego_slots=K,
            )
            best_k = branch_logits.argmax(dim=1)

        def initial_state_constraint(xt, t, step):
            xt = xt.reshape(B, K + P - 1, -1, 4)
            xt[:, :, 0, :] = current_states
            return xt.reshape(B, K + P - 1, -1)

        guidance_inputs = dict(inputs)
        guidance_inputs["neighbor_current_mask"] = neighbor_current_mask

        x0 = dpm_sampler(
            self.dit,
            xT,
            diffusion_steps=self._anchor_sampling_steps,
            other_model_params={
                "cross_c": ego_neighbor_encoding,
                "route_lanes": route_lanes,
                "neighbor_current_mask": neighbor_current_mask,
                "num_ego_slots": K,
                "wta_idx": best_k,
            },
            dpm_solver_params={
                "correcting_xt_fn": initial_state_constraint,
            },
            model_wrapper_params={
                "classifier_fn": self._guidance_fn,
                "classifier_kwargs": {
                    "model": self.dit,
                    "model_condition": {
                        "cross_c": ego_neighbor_encoding,
                        "route_lanes": route_lanes,
                        "neighbor_current_mask": neighbor_current_mask,
                        "num_ego_slots": K,
                        "wta_idx": best_k,
                    },
                    "inputs": guidance_inputs,
                    "observation_normalizer": self._observation_normalizer,
                    "state_normalizer": self._state_normalizer,
                },
                "guidance_scale": 0.5,
                "guidance_type": "classifier" if self._guidance_fn is not None else "uncond",
            },
            sample_params={
                "t_start": self._anchor_sampling_t_start,
            },
        )

        x0 = self._state_normalizer.inverse(x0.reshape(B, K + P - 1, -1, 4))[:, :, 1:]
        x0 = integrate_ego_modes(x0, K)

        if best_k is None:
            best_k = torch.zeros(B, dtype=torch.long, device=device)
        batch_idx = torch.arange(B, device=device)
        selected_ego = x0[batch_idx, best_k]
        prediction = torch.cat([selected_ego[:, None], x0[:, K:]], dim=1)

        output = {
            "prediction": prediction,
            "anchor_predictions": x0,
            "anchor_selected_index": best_k,
        }
        if branch_logits is not None:
            output["anchor_scores"] = branch_logits
        return output
    
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """
        # Extract ego & neighbor current states
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        if self.training:
            num_ego_slots = int(inputs.get("anchor_branch_count", 1))
            if num_ego_slots > 1:
                current_states = torch.cat(
                    [ego_current.expand(B, num_ego_slots, 4), neighbors_current],
                    dim=1,
                )
                P = current_states.shape[1]
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1) # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            diffusion_time = inputs['diffusion_time']

            if getattr(self.dit, "use_anchor_score", False) and "anchor_branch_count" in inputs:
                score, branch_logits = self.dit.forward_with_logits(
                    sampled_trajectories,
                    diffusion_time,
                    ego_neighbor_encoding,
                    route_lanes,
                    neighbor_current_mask,
                    num_ego_slots=num_ego_slots,
                    wta_idx=inputs.get("wta_idx", None),
                )
                return {
                    "score": score.reshape(B, P, -1, 4),
                    "branch_logits": branch_logits,
                }

            return {
                    "score": self.dit(
                        sampled_trajectories, 
                        diffusion_time,
                        ego_neighbor_encoding,
                        route_lanes,
                        neighbor_current_mask,
                        num_ego_slots=num_ego_slots,
                        wta_idx=inputs.get("wta_idx", None),
                    ).reshape(B, P, -1, 4)
                }
        else:
            if self._use_ego_anchor:
                return self._anchored_inference(
                    inputs,
                    current_states,
                    ego_neighbor_encoding,
                    route_lanes,
                    neighbor_current_mask,
                )

            # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            xT = torch.cat([current_states[:, :, None], torch.randn(B, P, self._future_len, 4).to(current_states.device) * 0.5], dim=2).reshape(B, P, -1)

            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, P, -1, 4)
                xt[:, :, 0, :] = current_states
                return xt.reshape(B, P, -1)
            
            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        other_model_params={
                            "cross_c": ego_neighbor_encoding, 
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask                            
                        },
                        dpm_solver_params={
                            "correcting_xt_fn":initial_state_constraint,
                        },
                        model_wrapper_params={
                            "classifier_fn": self._guidance_fn,
                            "classifier_kwargs": {
                                "model": self.dit,
                                "model_condition": {
                                    "cross_c": ego_neighbor_encoding, 
                                    "route_lanes": route_lanes,
                                    "neighbor_current_mask": neighbor_current_mask                            
                                },
                                "inputs": inputs,
                                "observation_normalizer": self._observation_normalizer,
                                "state_normalizer": self._state_normalizer
                            },
                            "guidance_scale": 0.5,
                            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                        },
                )
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]
            x0 = integrate_ego_velocity(x0)

            return {
                    "prediction": x0
                }

        
class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, -1)


class DiT(nn.Module):
    def __init__(
        self,
        sde: SDE,
        route_encoder: nn.Module,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
        model_type="x_start",
        use_anchor_score=False,
    ):
        super().__init__()
        
        assert model_type in ["noise", "score", "x_start", "v"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.use_anchor_score = use_anchor_score
        self._num_heads = heads
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        if self.use_anchor_score:
            self.score_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    @property
    def model_type(self):
        return self._model_type

    def _build_mode_attn_mask(self, batch_size, total_slots, num_ego_slots, wta_idx, device):
        if num_ego_slots <= 1:
            return None

        mask = torch.zeros((batch_size, total_slots, total_slots), dtype=torch.bool, device=device)
        ego = num_ego_slots

        ego_block = torch.ones((ego, ego), dtype=torch.bool, device=device)
        ego_block.fill_diagonal_(False)
        mask[:, :ego, :ego] = ego_block

        mask[:, ego:, :ego] = True
        if wta_idx is not None:
            selected = wta_idx.to(device).view(batch_size, 1, 1).expand(-1, total_slots - ego, 1)
            mask[:, ego:, :ego].scatter_(2, selected, False)

        return mask.repeat_interleave(self._num_heads, dim=0)

    def _forward_features(
        self,
        x,
        t,
        cross_c,
        route_lanes,
        neighbor_current_mask,
        num_ego_slots=1,
        wta_idx=None,
    ):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        B, P, _ = x.shape
        
        x = self.preproj(x)

        num_ego_slots = int(num_ego_slots)
        if not 1 <= num_ego_slots <= P:
            raise ValueError(f"num_ego_slots must be in [1, {P}], got {num_ego_slots}")
        neighbor_slots = P - num_ego_slots

        x_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :].expand(num_ego_slots, -1),
                self.agent_embedding.weight[1][None, :].expand(neighbor_slots, -1),
            ],
            dim=0,
        )  # (P, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B, P, D)
        x = x + x_embedding     

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)      

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        if neighbor_slots > 0:
            attn_mask[:, num_ego_slots:] = neighbor_current_mask
        mode_attn_mask = self._build_mode_attn_mask(B, P, num_ego_slots, wta_idx, x.device)
        
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask, mode_attn_mask)  
        return x, y

    def forward_with_logits(
        self,
        x,
        t,
        cross_c,
        route_lanes,
        neighbor_current_mask,
        num_ego_slots=1,
        wta_idx=None,
    ):
        x, y = self._forward_features(
            x,
            t,
            cross_c,
            route_lanes,
            neighbor_current_mask,
            num_ego_slots=num_ego_slots,
            wta_idx=wta_idx,
        )
        branch_logits = self.score_head(x[:, :num_ego_slots]).squeeze(-1)
        x = self.final_layer(x, y)
        x = self._format_output(x, t)
        return x, branch_logits

    def _format_output(self, x, t):
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start" or self._model_type == "noise" or self._model_type == "v":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def forward(
        self,
        x,
        t,
        cross_c,
        route_lanes,
        neighbor_current_mask,
        num_ego_slots=1,
        wta_idx=None,
    ):
        x, y = self._forward_features(
            x,
            t,
            cross_c,
            route_lanes,
            neighbor_current_mask,
            num_ego_slots=num_ego_slots,
            wta_idx=wta_idx,
        )
        x = self.final_layer(x, y)
        
        return self._format_output(x, t)
