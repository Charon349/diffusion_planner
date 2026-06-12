#!/usr/bin/env python3
"""Export nuPlan expert ego trajectory for one scenario.

Example:
    python gt_visualization.py \
        --log-name 2021.05.12.19.36.12_veh-35_00005_00204 \
        --scenario-name fa20c34947a8558d \
        --data-root /data/nuplan-v1.1/trainval \
        --map-root /data/nuplan-v1.1/maps
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class FixedScenarioMapping:
    """Return the same extraction window for every nuPlan scenario type."""

    def __init__(self, scenario_duration: float, extraction_offset: float, subsample_ratio: float):
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo

        self._extraction_info = ScenarioExtractionInfo(
            scenario_duration=scenario_duration,
            extraction_offset=extraction_offset,
            subsample_ratio=subsample_ratio,
        )

    def get_extraction_info(self, scenario_type: str) -> Any:
        return self._extraction_info


def _add_nuplan_to_pythonpath(nuplan_devkit_root: Optional[str]) -> None:
    """Make sibling nuplan-devkit imports work when the package is not installed."""
    candidates: List[Path] = []

    if nuplan_devkit_root:
        candidates.append(Path(nuplan_devkit_root))

    env_root = os.environ.get("NUPLAN_DEVKIT_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(Path(__file__).resolve().parents[1] / "nuplan-devkit")

    for candidate in candidates:
        if (candidate / "nuplan").is_dir():
            sys.path.insert(0, str(candidate))
            return


def _resolve_data_root(data_root: Optional[str], split: str) -> str:
    """Resolve a nuPlan db directory from explicit arg or NUPLAN_DATA_ROOT."""
    if data_root:
        return data_root

    env_root = os.environ.get("NUPLAN_DATA_ROOT")
    if not env_root:
        raise ValueError("Please provide --data-root or set NUPLAN_DATA_ROOT.")

    env_path = Path(env_root)
    split_path = env_path / "nuplan-v1.1" / split
    if split_path.exists():
        return str(split_path)

    return str(env_path)


def _resolve_map_root(map_root: Optional[str]) -> str:
    """Resolve nuPlan maps directory from explicit arg or environment."""
    if map_root:
        return map_root

    env_maps = os.environ.get("NUPLAN_MAPS_ROOT")
    if env_maps:
        return env_maps

    env_data = os.environ.get("NUPLAN_DATA_ROOT")
    if env_data:
        candidate = Path(env_data) / "nuplan-v1.1" / "maps"
        if candidate.exists():
            return str(candidate)

    raise ValueError("Please provide --map-root or set NUPLAN_MAPS_ROOT.")


def _resolve_sensor_root(sensor_root: Optional[str]) -> Optional[str]:
    """Resolve sensor root if available; ego trajectory export does not need it."""
    if sensor_root:
        return sensor_root

    env_sensor = os.environ.get("NUPLAN_SENSOR_ROOT")
    if env_sensor:
        return env_sensor

    env_data = os.environ.get("NUPLAN_DATA_ROOT")
    if env_data:
        candidate = Path(env_data) / "nuplan-v1.1" / "sensor_blobs"
        if candidate.exists():
            return str(candidate)

    return None


def _make_filter(log_name: str, scenario_name: str) -> Any:
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    return ScenarioFilter(
        scenario_types=None,
        scenario_tokens=[scenario_name],
        log_names=[log_name],
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
    )


def _build_scenarios(
    data_root: str,
    map_root: str,
    sensor_root: Optional[str],
    log_name: str,
    scenario_name: str,
    map_version: str,
    scenario_duration: float,
    extraction_offset: float,
    subsample_ratio: float,
) -> List[Any]:
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

    # The filtered token is the anchor lidarpc. With expand_scenarios=False,
    # nuPlan uses ScenarioExtractionInfo to collect the full time window.
    scenario_mapping = FixedScenarioMapping(
        scenario_duration=scenario_duration,
        extraction_offset=extraction_offset,
        subsample_ratio=subsample_ratio,
    )

    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        sensor_root=sensor_root,
        db_files=None,
        map_version=map_version,
        include_cameras=False,
        verbose=True,
        scenario_mapping=scenario_mapping,
    )
    scenario_filter = _make_filter(log_name, scenario_name)
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    return builder.get_scenarios(scenario_filter, worker)


def _state_to_row(state: Any, iteration: int, start_time_us: int) -> Dict[str, Any]:
    dynamic = state.dynamic_car_state
    rear_v = dynamic.rear_axle_velocity_2d
    rear_a = dynamic.rear_axle_acceleration_2d
    center_v = dynamic.center_velocity_2d
    center_a = dynamic.center_acceleration_2d

    return {
        "iteration": iteration,
        "time_us": state.time_us,
        "relative_time_s": (state.time_us - start_time_us) * 1e-6,
        "x": state.rear_axle.x,
        "y": state.rear_axle.y,
        "heading": state.rear_axle.heading,
        "center_x": state.center.x,
        "center_y": state.center.y,
        "center_heading": state.center.heading,
        "vx": rear_v.x,
        "vy": rear_v.y,
        "ax": rear_a.x,
        "ay": rear_a.y,
        "center_vx": center_v.x,
        "center_vy": center_v.y,
        "center_ax": center_a.x,
        "center_ay": center_a.y,
        "speed": dynamic.speed,
        "acceleration": dynamic.acceleration,
        "angular_velocity": dynamic.angular_velocity,
        "angular_acceleration": dynamic.angular_acceleration,
        "tire_steering_angle": state.tire_steering_angle,
        "is_in_auto_mode": state.is_in_auto_mode,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "trajectory": rows}, f, indent=2)


def _save_plot(path: Path, rows: List[Dict[str, Any]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot") from exc

    frames = [row["iteration"] for row in rows]
    xs = [row["x"] for row in rows]
    ys = [row["y"] for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    fig.suptitle(title)

    axes[0].plot(frames, xs, linewidth=2)
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("x [m]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, ys, linewidth=2)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("y [m]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _default_output_path(log_name: str, scenario_name: str, output_format: str) -> Path:
    safe_log = log_name.replace("/", "_").replace(".db", "")
    return Path(f"gt_{safe_log}_{scenario_name}.{output_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-name", required=True, help="nuPlan log name, with or without .db suffix.")
    parser.add_argument(
        "--scenario-name",
        "--scenario-token",
        dest="scenario_name",
        required=True,
        help="nuPlan scenario name/token. In this devkit version scenario_name == lidarpc token.",
    )
    parser.add_argument("--data-root", default=None, help="Directory containing nuPlan .db files, e.g. /data/nuplan-v1.1/trainval.")
    parser.add_argument("--map-root", default=None, help="nuPlan maps directory, e.g. /data/nuplan-v1.1/maps.")
    parser.add_argument("--sensor-root", default=None, help="Optional nuPlan sensor_blobs directory.")
    parser.add_argument("--nuplan-devkit-root", default=None, help="Optional path to nuplan-devkit if it is not installed.")
    parser.add_argument("--split", default="trainval", choices=["trainval", "test", "mini"], help="Used only when resolving NUPLAN_DATA_ROOT.")
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--scenario-duration", type=float, default=20.0, help="Seconds to export from the anchor frame.")
    parser.add_argument("--extraction-offset", type=float, default=0.0, help="Start offset in seconds relative to the anchor frame.")
    parser.add_argument(
        "--subsample-ratio",
        type=float,
        default=1.0,
        help="1.0 keeps every DB frame at 20Hz; 0.5 keeps every other frame at 10Hz.",
    )
    parser.add_argument("--output", default=None, help="Output file path. Defaults to gt_<log>_<scenario>.csv/json.")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output file format.")
    parser.add_argument("--plot", default=None, help="Optional PNG path for a quick xy trajectory plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_nuplan_to_pythonpath(args.nuplan_devkit_root)

    log_name = args.log_name[:-3] if args.log_name.endswith(".db") else args.log_name
    data_root = _resolve_data_root(args.data_root, args.split)
    map_root = _resolve_map_root(args.map_root)
    sensor_root = _resolve_sensor_root(args.sensor_root)

    scenarios = _build_scenarios(
        data_root=data_root,
        map_root=map_root,
        sensor_root=sensor_root,
        log_name=log_name,
        scenario_name=args.scenario_name,
        map_version=args.map_version,
        scenario_duration=args.scenario_duration,
        extraction_offset=args.extraction_offset,
        subsample_ratio=args.subsample_ratio,
    )

    if not scenarios:
        raise RuntimeError(
            "No scenario found. Check --log-name, --scenario-name/token, --data-root, and --map-root."
        )
    if len(scenarios) > 1:
        print(f"Warning: found {len(scenarios)} scenarios; exporting the first one.", file=sys.stderr)

    scenario = scenarios[0]
    start_time_us = scenario.get_time_point(0).time_us
    rows = [
        _state_to_row(scenario.get_ego_state_at_iteration(i), i, start_time_us)
        for i in range(scenario.get_number_of_iterations())
    ]

    metadata = {
        "log_name": scenario.log_name,
        "scenario_name": scenario.scenario_name,
        "scenario_token": scenario.token,
        "scenario_type": scenario.scenario_type,
        "map_name": getattr(scenario, "_map_name", None),
        "database_interval_s": scenario.database_interval,
        "num_iterations": scenario.get_number_of_iterations(),
        "duration_s": scenario.duration_s.time_s,
        "data_root": data_root,
        "map_root": map_root,
    }

    output_path = Path(args.output) if args.output else _default_output_path(log_name, args.scenario_name, args.format)
    if args.format == "csv":
        _write_csv(output_path, rows)
        metadata_path = output_path.with_suffix(".metadata.json")
        _write_json(metadata_path, metadata, [])
    else:
        _write_json(output_path, metadata, rows)
        metadata_path = None

    if args.plot:
        _save_plot(Path(args.plot), rows, f"{scenario.log_name} / {scenario.scenario_name}")

    print(f"Exported {len(rows)} expert ego states to {output_path}")
    if metadata_path:
        print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
