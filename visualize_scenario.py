"""
Scenario-level simulation visualization tool.
Replaces nuboard with matplotlib-based static plots.

Usage examples:
  # List all scenario types
  python visualize_scenario.py --sim_root <SIM_DIR> --list_types

  # List scenarios of a specific type
  python visualize_scenario.py --sim_root <SIM_DIR> --scenario_type near_multiple_vehicles --list_scenarios

  # Visualize a specific scenario by token
  python visualize_scenario.py --sim_root <SIM_DIR> --token 005c7b67ab8c5eb7

  # Visualize a specific scenario by type + index
  python visualize_scenario.py --sim_root <SIM_DIR> --scenario_type near_multiple_vehicles --index 0

  # Visualize with specific timestep planning overlay
  python visualize_scenario.py --sim_root <SIM_DIR> --token 005c7b67ab8c5eb7 --show_plan_at 0,50,100

  # Sort by score and visualize worst N
  python visualize_scenario.py --sim_root <SIM_DIR> --worst 5
"""

import argparse
import lzma
import math
import msgpack
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


NUPLAN_DEVKIT_ROOT = os.environ.get(
    "NUPLAN_DEVKIT_ROOT",
    "/mnt/workspace/users/wangchenggang/nuplan-devkit",
)
if NUPLAN_DEVKIT_ROOT not in sys.path:
    sys.path.insert(0, NUPLAN_DEVKIT_ROOT)

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def load_sim_log(msgpack_xz_path: str):
    with lzma.open(msgpack_xz_path, "rb") as f:
        return pickle.loads(msgpack.unpackb(f.read()))


def find_scenario_file(sim_root: str, token: str = None, scenario_type: str = None, index: int = 0):
    """Locate the .msgpack.xz file for a scenario."""
    sim_log_dir = os.path.join(sim_root, "simulation_log", "diffusion_planner")

    if token:
        for stype in os.listdir(sim_log_dir):
            stype_dir = os.path.join(sim_log_dir, stype)
            if not os.path.isdir(stype_dir):
                continue
            for log_name in os.listdir(stype_dir):
                token_dir = os.path.join(stype_dir, log_name, token)
                if os.path.isdir(token_dir):
                    xz = os.path.join(token_dir, f"{token}.msgpack.xz")
                    if os.path.exists(xz):
                        return xz, stype
        raise FileNotFoundError(f"Token {token} not found under {sim_log_dir}")

    if scenario_type:
        stype_dir = os.path.join(sim_log_dir, scenario_type)
        log_names = sorted(os.listdir(stype_dir))
        if index >= len(log_names):
            raise IndexError(f"Index {index} out of range (total {len(log_names)} for {scenario_type})")
        log_dir = os.path.join(stype_dir, log_names[index])
        tokens = os.listdir(log_dir)
        if not tokens:
            raise FileNotFoundError(f"No token dirs in {log_dir}")
        tok = tokens[0]
        xz = os.path.join(log_dir, tok, f"{tok}.msgpack.xz")
        return xz, scenario_type

    raise ValueError("Must specify --token or --scenario_type + --index")


def get_metrics_df(sim_root: str):
    agg_dir = os.path.join(sim_root, "aggregator_metric")
    parquets = list(Path(agg_dir).glob("*.parquet"))
    if not parquets:
        return None
    return pd.read_parquet(parquets[0])


def draw_vehicle_box(ax, x, y, heading, length=4.5, width=2.0, color="blue", alpha=0.6, label=None):
    """Draw a rotated rectangle representing a vehicle."""
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    corners = np.array([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2],
    ])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners @ rot.T + np.array([x, y])
    rect = patches.Polygon(corners, closed=True, facecolor=color, edgecolor="black",
                           alpha=alpha, linewidth=0.5, label=label)
    ax.add_patch(rect)

    arrow_len = length * 0.4
    ax.arrow(x, y, arrow_len * cos_h, arrow_len * sin_h,
             head_width=0.3, head_length=0.2, fc=color, ec="black", linewidth=0.3)


def draw_lanes(ax, map_api, center_x, center_y, radius=80.0):
    """Draw nearby lane center-lines and lane connectors."""
    point = Point2D(center_x, center_y)
    try:
        nearby = map_api.get_proximal_map_objects(
            point, radius,
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )
    except Exception:
        return

    for layer, objs in nearby.items():
        color = "#cccccc" if layer == SemanticMapLayer.LANE else "#aaddaa"
        lw = 1.0 if layer == SemanticMapLayer.LANE else 0.8
        ls = "-" if layer == SemanticMapLayer.LANE else "--"
        for obj in objs:
            try:
                pts = obj.baseline_path.discrete_path
                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, alpha=0.7, zorder=0)
            except Exception:
                continue


def visualize_scenario(sim_root, xz_path, scenario_type, output_dir, show_plan_at=None, score=None):
    """Main visualization for one scenario."""
    print(f"Loading: {os.path.basename(xz_path)} ...")
    data = load_sim_log(xz_path)

    sim_hist = data.simulation_history
    scenario = data.scenario
    token = scenario.token
    log_name = scenario.log_name
    map_api = sim_hist.map_api
    mission_goal = sim_hist.mission_goal
    num_steps = len(sim_hist)

    sim_ego_x = [s.ego_state.rear_axle.x for s in sim_hist.data]
    sim_ego_y = [s.ego_state.rear_axle.y for s in sim_hist.data]
    sim_ego_h = [s.ego_state.rear_axle.heading for s in sim_hist.data]

    gt_ego_x, gt_ego_y, gt_ego_h = [], [], []
    for i in range(scenario.get_number_of_iterations()):
        st = scenario.get_ego_state_at_iteration(i)
        gt_ego_x.append(st.rear_axle.x)
        gt_ego_y.append(st.rear_axle.y)
        gt_ego_h.append(st.rear_axle.heading)

    # ---- Figure 1: Full trajectory overview ----
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    cx = np.mean(sim_ego_x)
    cy = np.mean(sim_ego_y)

    draw_lanes(ax, map_api, cx, cy, radius=100.0)

    ax.plot(gt_ego_x, gt_ego_y, "g-", linewidth=2.5, label="GT (Expert)", zorder=2, alpha=0.8)
    ax.plot(sim_ego_x, sim_ego_y, "b-", linewidth=2.5, label="Model (Sim)", zorder=3, alpha=0.9)

    ax.plot(sim_ego_x[0], sim_ego_y[0], "ko", markersize=10, zorder=5, label="Start")
    ax.plot(mission_goal.x, mission_goal.y, "r*", markersize=15, zorder=5, label="Goal")

    draw_vehicle_box(ax, sim_ego_x[0], sim_ego_y[0], sim_ego_h[0],
                     color="#4488ff", alpha=0.8, label="Ego (start)")
    draw_vehicle_box(ax, sim_ego_x[-1], sim_ego_y[-1], sim_ego_h[-1],
                     color="#ff6644", alpha=0.8, label="Ego (end)")

    all_x = sim_ego_x + gt_ego_x
    all_y = sim_ego_y + gt_ego_y
    margin = 20
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    title = f"[{scenario_type}] token={token}\nlog={log_name}"
    if score is not None:
        title += f"  |  score={score:.4f}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    fname1 = os.path.join(output_dir, f"{token}_overview.png")
    fig.savefig(fname1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname1}")

    # ---- Figure 2: Per-timestep snapshots with planning ----
    if show_plan_at is None:
        step_count = min(6, num_steps)
        show_plan_at = np.linspace(0, num_steps - 1, step_count, dtype=int).tolist()

    ncols = min(3, len(show_plan_at))
    nrows = math.ceil(len(show_plan_at) / ncols)
    fig2, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, step in enumerate(show_plan_at):
        step = min(step, num_steps - 1)
        row, col = divmod(idx, ncols)
        ax2 = axes[row, col]

        sample = sim_hist.data[step]
        ego = sample.ego_state
        ex, ey, eh = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading

        draw_lanes(ax2, map_api, ex, ey, radius=60.0)

        ax2.plot(gt_ego_x, gt_ego_y, "g-", linewidth=1.5, alpha=0.4, label="GT")
        ax2.plot(sim_ego_x[:step + 1], sim_ego_y[:step + 1], "b-", linewidth=1.5, alpha=0.5, label="Sim (past)")

        traj = sample.trajectory
        try:
            traj_states = traj.get_sampled_trajectory()
            plan_x = [s.rear_axle.x for s in traj_states]
            plan_y = [s.rear_axle.y for s in traj_states]
            ax2.plot(plan_x, plan_y, "r-", linewidth=2.0, alpha=0.8, label="Planned traj")
            ax2.plot(plan_x[-1], plan_y[-1], "rv", markersize=6)
        except Exception:
            pass

        obs = sample.observation
        if hasattr(obs, "tracked_objects"):
            agents = obs.tracked_objects.tracked_objects
            for ai, agent in enumerate(agents):
                c = agent.center
                draw_vehicle_box(ax2, c.x, c.y, c.heading,
                                 length=agent.box.length if hasattr(agent, 'box') else 4.5,
                                 width=agent.box.width if hasattr(agent, 'box') else 2.0,
                                 color="#ffaa44", alpha=0.4,
                                 label="Other agents" if ai == 0 else None)

        draw_vehicle_box(ax2, ex, ey, eh, color="#4488ff", alpha=0.9, label="Ego")

        view_r = 40
        ax2.set_xlim(ex - view_r, ex + view_r)
        ax2.set_ylim(ey - view_r, ey + view_r)
        ax2.set_aspect("equal")
        ax2.set_title(f"Step {step}/{num_steps - 1}", fontsize=11)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, alpha=0.3)

    for idx in range(len(show_plan_at), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig2.suptitle(f"[{scenario_type}] token={token} — Planning at each step", fontsize=13)
    fig2.tight_layout()
    fname2 = os.path.join(output_dir, f"{token}_steps.png")
    fig2.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fname2}")

    # ---- Figure 3: Trajectory error plot ----
    min_len = min(len(sim_ego_x), len(gt_ego_x))
    dx = np.array(sim_ego_x[:min_len]) - np.array(gt_ego_x[:min_len])
    dy = np.array(sim_ego_y[:min_len]) - np.array(gt_ego_y[:min_len])
    pos_err = np.sqrt(dx ** 2 + dy ** 2)

    dh = np.array(sim_ego_h[:min_len]) - np.array(gt_ego_h[:min_len])
    heading_err = np.abs(np.arctan2(np.sin(dh), np.cos(dh)))

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    timesteps = np.arange(min_len) * 0.1

    ax3a.plot(timesteps, pos_err, "b-", linewidth=1.5)
    ax3a.set_ylabel("Position Error (m)")
    ax3a.set_title(f"[{scenario_type}] token={token} — Trajectory Error vs GT")
    ax3a.grid(True, alpha=0.3)
    ax3a.axhline(y=np.mean(pos_err), color="r", linestyle="--", alpha=0.5, label=f"mean={np.mean(pos_err):.2f}m")
    ax3a.legend()

    ax3b.plot(timesteps, np.degrees(heading_err), "r-", linewidth=1.5)
    ax3b.set_ylabel("Heading Error (deg)")
    ax3b.set_xlabel("Time (s)")
    ax3b.grid(True, alpha=0.3)
    ax3b.axhline(y=np.mean(np.degrees(heading_err)), color="b", linestyle="--", alpha=0.5,
                 label=f"mean={np.mean(np.degrees(heading_err)):.2f}°")
    ax3b.legend()

    fname3 = os.path.join(output_dir, f"{token}_error.png")
    fig3.savefig(fname3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {fname3}")


def main():
    parser = argparse.ArgumentParser(description="Visualize nuPlan simulation scenarios (replaces nuboard)")
    parser.add_argument("--sim_root", required=True, help="Simulation result directory (contains simulation_log/, metrics/, etc.)")
    parser.add_argument("--output_dir", default=None, help="Output image directory (default: <sim_root>/vis/)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list_types", action="store_true", help="List all scenario types")
    group.add_argument("--list_scenarios", action="store_true", help="List scenarios (use with --scenario_type)")

    parser.add_argument("--token", default=None, help="Scenario token to visualize")
    parser.add_argument("--scenario_type", default=None, help="Scenario type filter")
    parser.add_argument("--index", type=int, default=0, help="Scenario index within a type (default: 0)")
    parser.add_argument("--show_plan_at", default=None, help="Comma-separated timestep indices to show planning, e.g. '0,30,60,90'")
    parser.add_argument("--worst", type=int, default=None, help="Visualize N worst-scoring scenarios")
    parser.add_argument("--best", type=int, default=None, help="Visualize N best-scoring scenarios")

    args = parser.parse_args()

    sim_log_dir = os.path.join(args.sim_root, "simulation_log", "diffusion_planner")
    if not os.path.isdir(sim_log_dir):
        print(f"Error: simulation_log dir not found at {sim_log_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.sim_root, "vis")
    os.makedirs(output_dir, exist_ok=True)

    if args.list_types:
        types = sorted(os.listdir(sim_log_dir))
        df = get_metrics_df(args.sim_root)
        print(f"\n{'Scenario Type':<60s}  {'Count':>6s}  {'Avg Score':>10s}")
        print("-" * 80)
        for t in types:
            t_dir = os.path.join(sim_log_dir, t)
            if not os.path.isdir(t_dir):
                continue
            count = len(os.listdir(t_dir))
            score_str = ""
            if df is not None:
                type_df = df[(df["scenario_type"] == t) & (df["scenario"] != "final_score")]
                if len(type_df) > 0:
                    score_str = f"{pd.to_numeric(type_df['score'], errors='coerce').mean():.4f}"
            print(f"{t:<60s}  {count:>6d}  {score_str:>10s}")
        return

    if args.list_scenarios:
        if not args.scenario_type:
            print("Error: --list_scenarios requires --scenario_type")
            sys.exit(1)
        stype_dir = os.path.join(sim_log_dir, args.scenario_type)
        log_names = sorted(os.listdir(stype_dir))
        df = get_metrics_df(args.sim_root)
        scores = {}
        if df is not None:
            for _, row in df[df["scenario_type"] == args.scenario_type].iterrows():
                scores[row["scenario"]] = float(row["score"])

        print(f"\n{'Idx':>4s}  {'Token':<20s}  {'Log Name':<50s}  {'Score':>8s}")
        print("-" * 90)
        for i, log_name in enumerate(log_names):
            token_dirs = os.listdir(os.path.join(stype_dir, log_name))
            tok = token_dirs[0] if token_dirs else "?"
            sc_str = f"{scores[tok]:.4f}" if tok in scores else ""
            print(f"{i:>4d}  {tok:<20s}  {log_name:<50s}  {sc_str:>8s}")
        return

    df = get_metrics_df(args.sim_root)
    show_plan_at = None
    if args.show_plan_at:
        show_plan_at = [int(x.strip()) for x in args.show_plan_at.split(",")]

    if args.worst or args.best:
        if df is None:
            print("Error: No aggregator_metric parquet found")
            sys.exit(1)
        scenario_df = df[df["scenario"] != "final_score"].copy()
        scenario_df["score_num"] = pd.to_numeric(scenario_df["score"], errors="coerce")

        if args.worst:
            selected = scenario_df.nsmallest(args.worst, "score_num")
        else:
            selected = scenario_df.nlargest(args.best, "score_num")

        for _, row in selected.iterrows():
            tok = row["scenario"]
            try:
                xz_path, stype = find_scenario_file(args.sim_root, token=tok)
                visualize_scenario(args.sim_root, xz_path, stype, output_dir,
                                   show_plan_at=show_plan_at, score=float(row["score_num"]))
            except Exception as e:
                print(f"  Error with {tok}: {e}")
        return

    if args.token:
        xz_path, stype = find_scenario_file(args.sim_root, token=args.token)
    elif args.scenario_type:
        xz_path, stype = find_scenario_file(args.sim_root, scenario_type=args.scenario_type, index=args.index)
    else:
        parser.print_help()
        sys.exit(1)

    score_val = None
    if df is not None:
        tok = os.path.basename(os.path.dirname(xz_path))
        score_row = df[df["scenario"] == tok]
        if len(score_row) > 0:
            score_val = float(score_row["score"].iloc[0])

    visualize_scenario(args.sim_root, xz_path, stype, output_dir,
                       show_plan_at=show_plan_at, score=score_val)


if __name__ == "__main__":
    main()
