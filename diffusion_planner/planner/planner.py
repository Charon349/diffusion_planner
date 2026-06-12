import json
import os
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config

def identity(ego_state, predictions):
    return predictions


class DiffusionPlanner(AbstractPlanner):
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer
        self._anchor_log_dir = os.getenv("ANCHOR_LOG_DIR", "")
        self._anchor_log_topk = int(os.getenv("ANCHOR_LOG_TOPK", "5"))
        self._anchor_log_include_traj = os.getenv("ANCHOR_LOG_INCLUDE_TRAJ", "1") != "0"
        self._anchor_log_path = None
        self._anchor_log_failed = False
        self._anchor_log_scenario_id = None

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            state_dict:Dict = torch.load(self._ckpt_path, map_location=self._device)
            
            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # use for ddp
            model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization
        self._anchor_log_scenario_id = None

    def __getstate__(self):
        """
        Keep planner picklable when nuPlan serializes SimulationLog.
        """
        state = self.__dict__.copy()
        state.pop("_anchor_log_handle", None)
        return state

    def _get_anchor_log_path(self):
        if not self._anchor_log_dir or self._anchor_log_failed:
            return None

        if self._anchor_log_path is None:
            try:
                os.makedirs(self._anchor_log_dir, exist_ok=True)
                self._anchor_log_path = os.path.join(
                    self._anchor_log_dir,
                    f"anchor_selection_pid{os.getpid()}_{id(self)}.jsonl",
                )
            except Exception as exc:
                self._anchor_log_failed = True
                print(f"[anchor-log] Failed to open anchor log dir {self._anchor_log_dir}: {exc}")
                return None

        return self._anchor_log_path

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _anchor_template_xy(self, selected_anchor_index: int):
        if not self._anchor_log_include_traj:
            return None

        try:
            anchors = self._planner.decoder.decoder._ego_anchors
            if anchors.numel() == 0:
                return None
            anchor = anchors[selected_anchor_index].detach().cpu()
            xy = torch.cumsum(anchor[:, :2], dim=0).numpy().astype(np.float64)
            return xy.tolist()
        except Exception:
            return None

    def _anchor_templates_xy(self, anchor_score_topk: List[Dict[str, float]]):
        if not self._anchor_log_include_traj:
            return None

        templates = []
        for rank, item in enumerate(anchor_score_topk, start=1):
            anchor_index = self._safe_int(item.get("index"))
            if anchor_index is None:
                continue
            templates.append(
                {
                    "rank": rank,
                    "index": anchor_index,
                    "score": self._safe_float(item.get("score")),
                    "xy": self._anchor_template_xy(anchor_index),
                }
            )
        return templates

    def _prediction_xy(self, outputs: Dict[str, torch.Tensor]):
        if not self._anchor_log_include_traj:
            return None

        try:
            prediction = outputs["prediction"][0, 0, :, :2].detach().cpu().numpy().astype(np.float64)
            return prediction.tolist()
        except Exception:
            return None

    def _anchor_state_format(self) -> str:
        configured_format = getattr(self._config, "ego_anchor_state_format", None)
        if configured_format:
            return configured_format

        try:
            anchor_path = getattr(self._config, "ego_anchor_path", None)
            if anchor_path and anchor_path.endswith(".npy"):
                anchor = np.load(anchor_path, mmap_mode="r")
                if anchor.ndim == 3 and anchor.shape[-1] == 3:
                    return "absolute"
        except Exception:
            pass

        return "delta"

    def _log_anchor_selection(self, current_input: PlannerInput, outputs: Dict[str, torch.Tensor]) -> None:
        if "anchor_selected_index" not in outputs:
            return

        log_path = self._get_anchor_log_path()
        if log_path is None:
            return

        try:
            selected_anchor_index = int(outputs["anchor_selected_index"][0].detach().cpu().item())
            iteration = getattr(current_input, "iteration", None)
            iteration_index = self._safe_int(getattr(iteration, "index", None))
            iteration_time_point = getattr(iteration, "time_point", None)
            iteration_time_us = self._safe_int(getattr(iteration_time_point, "time_us", None))

            ego_state = current_input.history.ego_states[-1]
            ego_time_us = self._safe_int(getattr(getattr(ego_state, "time_point", None), "time_us", None))
            ego_rear_axle = getattr(ego_state, "rear_axle", None)
            ego_x = self._safe_float(getattr(ego_rear_axle, "x", None))
            ego_y = self._safe_float(getattr(ego_rear_axle, "y", None))
            ego_heading = self._safe_float(getattr(ego_rear_axle, "heading", None))
            map_name = getattr(getattr(self, "_map_api", None), "map_name", None)

            if self._anchor_log_scenario_id is None or iteration_index == 0:
                self._anchor_log_scenario_id = (
                    f"{map_name or 'unknown_map'}_"
                    f"{ego_time_us if ego_time_us is not None else iteration_time_us}_"
                    f"{ego_x:.2f}_{ego_y:.2f}"
                    if ego_x is not None and ego_y is not None
                    else f"{map_name or 'unknown_map'}_{ego_time_us if ego_time_us is not None else iteration_time_us}"
                )

            record = {
                "scenario_id": self._anchor_log_scenario_id,
                "iteration_index": iteration_index,
                "iteration_time_us": iteration_time_us,
                "ego_time_us": ego_time_us,
                "ego": {
                    "x": ego_x,
                    "y": ego_y,
                    "heading": ego_heading,
                },
                "map_name": map_name,
                "ckpt_path": self._ckpt_path,
                "anchor_path": getattr(self._config, "ego_anchor_path", None),
                "anchor_state_format": self._anchor_state_format(),
                "selected_anchor_index": selected_anchor_index,
            }

            if "anchor_scores" in outputs:
                scores = outputs["anchor_scores"][0].detach().cpu()
                topk = min(max(self._anchor_log_topk, 1), scores.shape[0])
                top_values, top_indices = torch.topk(scores, k=topk)
                record["anchor_score_selected"] = float(scores[selected_anchor_index].item())
                record["anchor_score_topk"] = [
                    {"index": int(idx.item()), "score": float(val.item())}
                    for idx, val in zip(top_indices, top_values)
                ]
                record["topk_anchor_template_xy"] = self._anchor_templates_xy(record["anchor_score_topk"])

            record["selected_anchor_template_xy"] = self._anchor_template_xy(selected_anchor_index)
            record["selected_prediction_xy"] = self._prediction_xy(outputs)

            with open(log_path, "a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._anchor_log_failed = True
            print(f"[anchor-log] Failed to write anchor selection record: {exc}")

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        inputs = self.observation_normalizer(inputs)        
        _, outputs = self._planner(inputs)
        self._log_anchor_selection(current_input, outputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
    
