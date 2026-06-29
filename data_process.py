import os
import argparse
import json
from typing import Any, Dict

from diffusion_planner.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except ImportError:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _scenario_filter_path(name: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diffusion_planner",
        "config",
        "scenario_filter",
        f"{name}.yaml",
    )


def _make_filter_from_dict(cfg: Dict[str, Any]) -> ScenarioFilter:
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return ScenarioFilter(
        cfg.get("scenario_types"),
        cfg.get("scenario_tokens"),
        cfg.get("log_names"),
        cfg.get("map_names"),
        cfg.get("num_scenarios_per_type"),
        cfg.get("limit_total_scenarios"),
        cfg.get("timestamp_threshold_s"),
        cfg.get("ego_displacement_minimum_m"),
        cfg.get("expand_scenarios", True),
        cfg.get("remove_invalid_goals", False),
        cfg.get("shuffle", True),
        cfg.get("ego_start_speed_threshold"),
        cfg.get("ego_stop_speed_threshold"),
        cfg.get("speed_noise_tolerance"),
    )


def _make_train_filter(args) -> ScenarioFilter:
    with open(args.log_names_json, "r", encoding="utf-8") as file:
        log_names = json.load(file)

    limit_total_scenarios = 10 if args.total_scenarios is None else args.total_scenarios
    shuffle = True if args.shuffle_scenarios is None else args.shuffle_scenarios

    return ScenarioFilter(
        None,                         # scenario_types
        None,                         # scenario_tokens
        log_names,                    # log_names
        None,                         # map_names
        args.scenarios_per_type,
        limit_total_scenarios,
        None,                         # timestamp_threshold_s
        None,                         # ego_displacement_minimum_m
        True,                         # expand_scenarios
        False,                        # remove_invalid_goals
        shuffle,
        None,                         # ego_start_speed_threshold
        None,                         # ego_stop_speed_threshold
        None,                         # speed_noise_tolerance
    )


def build_scenario_filter(args) -> ScenarioFilter:
    if args.scenario_filter == "train" and args.scenario_filter_yaml is None:
        return _make_train_filter(args)

    filter_path = args.scenario_filter_yaml or _scenario_filter_path(args.scenario_filter)
    if not os.path.exists(filter_path):
        raise FileNotFoundError(f"Scenario filter yaml not found: {filter_path}")

    cfg = _load_yaml(filter_path)
    if args.scenarios_per_type is not None:
        cfg["num_scenarios_per_type"] = args.scenarios_per_type
    if args.total_scenarios is not None:
        cfg["limit_total_scenarios"] = args.total_scenarios
    if args.shuffle_scenarios is not None:
        cfg["shuffle"] = args.shuffle_scenarios
    return _make_filter_from_dict(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')
    parser.add_argument('--save_list', default=None, type=str, help='json list path for processed .npz files')
    parser.add_argument('--scenario_filter', default='train', type=str,
                        help='train, val14, val14-collision, test14-random, test14-hard, or a yaml basename under config/scenario_filter')
    parser.add_argument('--scenario_filter_yaml', default=None, type=str,
                        help='explicit ScenarioFilter yaml path; overrides --scenario_filter')
    parser.add_argument('--log_names_json', default='./nuplan_train.json', type=str,
                        help='training log-name json used only when --scenario_filter train')
    parser.add_argument('--map_version', default='nuplan-maps-v1.0', type=str)
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=None, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=_str_to_bool, default=None, help='override yaml shuffle')

    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    sensor_root = None
    db_files = None

    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, args.map_version)
    scenario_filter = build_scenario_filter(args)

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Scenario filter: {args.scenario_filter_yaml or args.scenario_filter}")
    print(f"Total number of scenarios: {len(scenarios)}")
    if len(scenarios) == 0:
        raise RuntimeError(
            "No scenarios found. Check --data_path for nuPlan .db files and verify "
            "that the selected scenario_filter matches this split."
        )

    # process data
    del worker, builder, scenario_filter
    processor = DataProcessor(args)
    processor.work(scenarios)

    npz_files = sorted(f for f in os.listdir(args.save_path) if f.endswith('.npz'))

    # Save the list to a JSON file
    save_list = args.save_list
    if save_list is None:
        save_list = "./diffusion_planner_training.json" if args.scenario_filter == "train" else f"./diffusion_planner_{args.scenario_filter}.json"
    with open(save_list, 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names to {save_list}")
