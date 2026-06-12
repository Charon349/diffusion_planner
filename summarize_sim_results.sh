#!/bin/bash
# ============================================
# nuPlan 仿真结果总结脚本
# 用法1 (单结果): bash summarize_sim_results.sh <仿真结果目录路径>
# 用法2 (对比):   bash summarize_sim_results.sh --compare <基准目录> <对比目录1> [<对比目录2> ...] [--output <html输出路径>]
# 示例1: bash summarize_sim_results.sh /path/to/model_epoch_480_...
# 示例2: bash summarize_sim_results.sh --compare /path/to/baseline /path/to/exp1 --output compare.html
# 示例3: bash summarize_sim_results.sh --compare /path/to/baseline /path/to/exp1 /path/to/exp2
# ============================================

resolve_result_dir() {
    local d="$1"
    if [ -d "$d/aggregator_metric" ]; then
        echo "$d"
        return 0
    fi
    local resolved
    resolved=$(find "$d" -mindepth 2 -maxdepth 4 -type d -name aggregator_metric -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [ -n "$resolved" ] && [ -d "$resolved/aggregator_metric" ]; then
        echo "$resolved"
        return 0
    fi
    return 1
}

# --- Compare mode ---
if [ "$1" = "--compare" ]; then
    shift
    COMPARE_DIRS=()
    OUTPUT_HTML=""
    while [ $# -gt 0 ]; do
        if [ "$1" = "--output" ]; then
            if [ -z "$2" ]; then
                echo "错误: --output 需要指定路径"
                exit 1
            fi
            OUTPUT_HTML="$2"
            shift 2
            break
        fi
        COMPARE_DIRS+=("$1")
        shift
    done

    if [ ${#COMPARE_DIRS[@]} -lt 2 ]; then
        echo "用法: bash $0 --compare <基准目录> <对比目录1> [<对比目录2> ...] [--output <html输出路径>]"
        exit 1
    fi

    RESOLVED_COMPARE_DIRS=()
    COMPARE_LABELS=()
    for d in "${COMPARE_DIRS[@]}"; do
        if [ ! -d "$d" ]; then
            echo "错误: 目录不存在: $d"; exit 1
        fi
        if ! resolved_d=$(resolve_result_dir "$d"); then
            echo "错误: 未找到 aggregator_metric 目录: $d"; exit 1
        fi
        if [ "$resolved_d" != "$d" ]; then
            echo "提示: 使用最新子结果目录: $resolved_d"
        fi
        RESOLVED_COMPARE_DIRS+=("$resolved_d")
        COMPARE_LABELS+=("$(basename "${d%/}")")
    done

    COMPARE_DIRS_JOINED=$(IFS=:; echo "${RESOLVED_COMPARE_DIRS[*]}")
    COMPARE_LABELS_JOINED=$(IFS=:; echo "${COMPARE_LABELS[*]}")
    export COMPARE_DIRS_JOINED COMPARE_LABELS_JOINED OUTPUT_HTML

    python3 << 'COMPARE_SCRIPT'
import pandas as pd
import glob, os, sys, html
from datetime import datetime

COMPARE_DIRS = os.environ["COMPARE_DIRS_JOINED"].split(":")
COMPARE_LABELS = os.environ.get("COMPARE_LABELS_JOINED", "").split(":")
OUTPUT_HTML = os.environ.get("OUTPUT_HTML", "")
BASELINE_DIR = COMPARE_DIRS[0]
TARGET_DIRS = COMPARE_DIRS[1:]

def load_result(result_dir):
    agg_files = glob.glob(os.path.join(result_dir, "aggregator_metric", "*.parquet"))
    if not agg_files:
        print(f"错误: {result_dir}/aggregator_metric 下未找到 parquet 文件")
        sys.exit(1)
    df = pd.read_parquet(agg_files[0])
    scenario_rows = df[df["scenario"] != "final_score"]
    final_rows = df[df["scenario"] == "final_score"]
    runner_file = os.path.join(result_dir, "runner_report.parquet")
    runner = pd.read_parquet(runner_file) if os.path.exists(runner_file) else None
    return df, scenario_rows, final_rows, runner

metric_cols = [
    "drivable_area_compliance", "driving_direction_compliance",
    "ego_is_comfortable", "ego_is_making_progress",
    "ego_progress_along_expert_route", "no_ego_at_fault_collisions",
    "speed_limit_compliance", "time_to_collision_within_bound", "score"
]
metric_names = {
    "drivable_area_compliance":        "可行驶区域合规",
    "driving_direction_compliance":    "行驶方向合规",
    "ego_is_comfortable":              "舒适性",
    "ego_is_making_progress":          "前进性",
    "ego_progress_along_expert_route": "沿专家路线前进",
    "no_ego_at_fault_collisions":      "无碰撞",
    "speed_limit_compliance":          "限速合规",
    "time_to_collision_within_bound":  "TTC",
    "score":                           "Overall Score"
}

higher_is_better = {col: True for col in metric_cols}

def get_final_val(final_rows, scenario_rows, col):
    if len(final_rows) > 0:
        v = final_rows[col].iloc[0]
        return float(v) if v is not None else None
    vals = pd.to_numeric(scenario_rows[col], errors="coerce")
    return float(vals.mean())

results = []
for i, d in enumerate(COMPARE_DIRS):
    _, scen, final, runner = load_result(d)
    scores = pd.to_numeric(scen["score"], errors="coerce")
    scen_c = scen.copy()
    scen_c["score_num"] = scores
    type_grp = scen_c.groupby("scenario_type")["score_num"].agg(["mean", "count"])
    results.append({
        "dir": d,
        "name": COMPARE_LABELS[i] if i < len(COMPARE_LABELS) and COMPARE_LABELS[i] else os.path.basename(d.rstrip("/")),
        "scen": scen,
        "final": final,
        "runner": runner,
        "scores": scores,
        "type_grp": type_grp,
    })

baseline = results[0]
targets = results[1:]

def delta_cell(base_val, tgt_val, hib=True, fmt=".4f", is_int=False):
    if base_val is None or tgt_val is None:
        return ("N/A", "", "neutral")
    diff = tgt_val - base_val
    if is_int:
        tgt_s = f"{int(tgt_val)}"
        sign = "+" if diff > 0 else ""
        d_s = f"{sign}{int(diff)}"
    else:
        tgt_s = f"{tgt_val:{fmt}}"
        sign = "+" if diff > 0 else ""
        d_s = f"{sign}{diff:{fmt}}"
    if abs(diff) < 1e-9:
        return (tgt_s, d_s, "neutral")
    improved = (diff > 0) == hib
    cls = "better" if improved else "worse"
    return (tgt_s, d_s, cls)

def target_header_cells():
    cells = []
    for t in targets:
        cells.append(f'<th>{html.escape(t["name"])}</th><th>Δ</th>')
    return "".join(cells)

def target_value_cells(base_val, tgt_vals, hib=True, fmt=".4f", is_int=False):
    cells = []
    for tv in tgt_vals:
        tgt_s, d_s, cls = delta_cell(base_val, tv, hib=hib, fmt=fmt, is_int=is_int)
        cells.append(f"<td>{tgt_s}</td><td><span class=\"tag {cls}\">{d_s}</span></td>")
    return "".join(cells)

all_types = sorted(set().union(*(set(r["type_grp"].index) for r in results)))

css = """
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 24px; background: #f5f7fa; color: #1a1a2e; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .subtitle { color: #666; font-size: 13px; margin-bottom: 20px; }
  .card { background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
          padding: 20px 24px; margin-bottom: 18px; overflow-x: auto; }
  .card h2 { font-size: 16px; margin: 0 0 12px 0; border-bottom: 2px solid #eee; padding-bottom: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; min-width: 480px; }
  th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }
  th { background: #fafbfc; font-weight: 600; color: #555; }
  td:first-child, th:first-child { text-align: left; white-space: normal; }
  .better { color: #0a8a3e; font-weight: 600; }
  .worse  { color: #d32f2f; font-weight: 600; }
  .neutral { color: #888; }
  .tag { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .tag.better { background: #e6f9ed; }
  .tag.worse  { background: #fde8e8; }
  .tag.neutral { background: #f0f0f0; }
  .dir-label { font-size: 12px; word-break: break-all; color: #888; }
  .legend { font-size: 12px; color: #777; margin-bottom: 12px; }
  .legend span { margin-right: 16px; }
</style>
"""

th_targets = target_header_cells()

rows_final = []
for col in metric_cols:
    b_val = get_final_val(baseline["final"], baseline["scen"], col)
    tgt_vals = [get_final_val(t["final"], t["scen"], col) for t in targets]
    hib = higher_is_better.get(col, True)
    b_s = f"{b_val:.4f}" if b_val is not None else "N/A"
    cn = html.escape(metric_names.get(col, col))
    rows_final.append(f"""<tr>
      <td>{html.escape(col)}<br><span style="color:#999;font-size:11px">{cn}</span></td>
      <td>{b_s}</td>
      {target_value_cells(b_val, tgt_vals, hib=hib)}
    </tr>""")

stats_items = [
    ("Mean",   lambda r: r["scores"].mean(),   True),
    ("Median", lambda r: r["scores"].median(), True),
    ("Std",    lambda r: r["scores"].std(),    False),
]
rows_stats = []
for name, fn, hib in stats_items:
    b_val = fn(baseline)
    tgt_vals = [fn(t) for t in targets]
    b_s = f"{b_val:.4f}"
    rows_stats.append(f"""<tr><td>{name}</td><td>{b_s}</td>
      {target_value_cells(b_val, tgt_vals, hib=hib)}</tr>""")

def count_ratio_cells(base_val, tgt_vals, tgt_totals, hib=True):
    cells = []
    for tv, tt in zip(tgt_vals, tgt_totals):
        tgt_s, d_s, cls = delta_cell(base_val, tv, hib=hib, is_int=True)
        cells.append(
            f"<td>{int(tv)} / {tt}</td><td><span class=\"tag {cls}\">{d_s}</span></td>"
        )
    return "".join(cells)

b_perfect = int((baseline["scores"] == 1.0).sum())
b_failed = int((baseline["scores"] == 0.0).sum())
tgt_perfect = [int((t["scores"] == 1.0).sum()) for t in targets]
tgt_failed = [int((t["scores"] == 0.0).sum()) for t in targets]
tgt_lens = [len(t["scores"]) for t in targets]
rows_stats.append(f"""<tr><td>Perfect (1.0)</td>
  <td>{b_perfect} / {len(baseline["scores"])}</td>
  {count_ratio_cells(b_perfect, tgt_perfect, tgt_lens, hib=True)}</tr>""")
rows_stats.append(f"""<tr><td>Failed (0.0)</td>
  <td>{b_failed} / {len(baseline["scores"])}</td>
  {count_ratio_cells(b_failed, tgt_failed, tgt_lens, hib=False)}</tr>""")

rows_type = []
for st in all_types:
    bm = baseline["type_grp"].loc[st, "mean"] if st in baseline["type_grp"].index else None
    bc = int(baseline["type_grp"].loc[st, "count"]) if st in baseline["type_grp"].index else 0
    tgt_means = []
    tgt_counts = []
    for t in targets:
        if st in t["type_grp"].index:
            tgt_means.append(t["type_grp"].loc[st, "mean"])
            tgt_counts.append(int(t["type_grp"].loc[st, "count"]))
        else:
            tgt_means.append(None)
            tgt_counts.append(0)
    b_s = f"{bm:.4f}" if bm is not None else "N/A"
    type_cells = []
    for tm, tc in zip(tgt_means, tgt_counts):
        tgt_s, d_s, cls = delta_cell(bm, tm, hib=True)
        n_s = f" <span style=\"color:#aaa\">(n={tc})</span>" if tc else ""
        type_cells.append(f"<td>{tgt_s}{n_s}</td><td><span class=\"tag {cls}\">{d_s}</span></td>")
    rows_type.append(f"""<tr><td>{html.escape(st)}</td>
      <td>{b_s} <span style="color:#aaa">(n={bc})</span></td>
      {"".join(type_cells)}</tr>""")

runner_html = ""
if all(r["runner"] is not None for r in results):
    items = [
        ("Succeeded", lambda r: int(r["runner"]["succeeded"].sum()), True, True),
        ("Failed", lambda r: int((~r["runner"]["succeeded"]).sum()), False, True),
        ("Avg time (mean)", lambda r: r["runner"]["compute_trajectory_runtimes_mean"].mean(), False, False),
        ("Avg time (median)", lambda r: r["runner"]["compute_trajectory_runtimes_median"].mean(), False, False),
    ]
    rr = []
    for name, fn, hib, is_int in items:
        b_val = fn(baseline)
        tgt_vals = [fn(t) for t in targets]
        b_s = f"{int(b_val)}" if is_int else f"{b_val:.4f}"
        rr.append(f"<tr><td>{name}</td><td>{b_s}</td>{target_value_cells(b_val, tgt_vals, hib=hib, is_int=is_int)}</tr>")
    runner_html = f"""
    <div class="card">
      <h2>Runner Report</h2>
      <table><tr><th>Item</th><th>{html.escape(baseline["name"])} (Baseline)</th>{th_targets}</tr>
      {"".join(rr)}</table>
    </div>"""

dir_rows = []
for i, r in enumerate(results):
    label = "Baseline" if i == 0 else f"Compare {i}"
    dir_rows.append(
        f'<tr><td style="width:100px;font-weight:600">{label} ({html.escape(r["name"])})</td>'
        f'<td class="dir-label">{html.escape(r["dir"])}</td></tr>'
    )
scenario_counts = " &nbsp;|&nbsp; ".join(
    f'{r["name"]}: {len(r["scen"])}' for r in results
)

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
n_compare = len(TARGET_DIRS)

page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nuPlan Simulation Comparison ({1 + n_compare} runs)</title>
{css}
</head><body>
<h1>nuPlan Simulation Comparison Report</h1>
<p class="subtitle">Generated: {now_str} &nbsp;|&nbsp; {1 + n_compare} runs (1 baseline + {n_compare} compare)</p>
<div class="legend">
  <span class="better">■ Green = improved vs baseline</span>
  <span class="worse">■ Red = degraded vs baseline</span>
  <span class="neutral">■ Gray = no change</span>
</div>

<div class="card">
  <h2>Directories</h2>
  <table>
    {"".join(dir_rows)}
    <tr><td style="font-weight:600">Scenarios</td><td>{scenario_counts}</td></tr>
  </table>
</div>

<div class="card">
  <h2>Final Aggregated Scores</h2>
  <table>
    <tr><th style="text-align:left">Metric</th><th>{html.escape(baseline["name"])} (Baseline)</th>{th_targets}</tr>
    {"".join(rows_final)}
  </table>
</div>

<div class="card">
  <h2>Score Statistics</h2>
  <table>
    <tr><th style="text-align:left">Item</th><th>{html.escape(baseline["name"])} (Baseline)</th>{th_targets}</tr>
    {"".join(rows_stats)}
  </table>
</div>

<div class="card">
  <h2>Score by Scenario Type</h2>
  <table>
    <tr><th style="text-align:left">Scenario Type</th><th>{html.escape(baseline["name"])} (Baseline)</th>{th_targets}</tr>
    {"".join(rows_type)}
  </table>
</div>

{runner_html}

</body></html>
"""

if not OUTPUT_HTML:
    OUTPUT_HTML = os.path.join(os.path.dirname(BASELINE_DIR.rstrip("/")), "sim_compare.html")
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(page)
print(f"对比报告已生成 ({1 + len(TARGET_DIRS)} 个结果): {OUTPUT_HTML}")

COMPARE_SCRIPT

    exit 0
fi

# --- Single result mode (original) ---
if [ -z "$1" ]; then
    echo "用法: bash $0 <仿真结果目录路径>"
    echo "      bash $0 --compare <基准目录> <对比目录1> [<对比目录2> ...] [--output <html输出路径>]"
    exit 1
fi

RESULT_DIR="$1"

if [ ! -d "$RESULT_DIR" ]; then
    echo "错误: 目录不存在: $RESULT_DIR"
    exit 1
fi

if [ ! -d "$RESULT_DIR/aggregator_metric" ]; then
    echo "错误: 未找到 aggregator_metric 目录，请确认仿真已完成"
    exit 1
fi

export RESULT_DIR="$RESULT_DIR"

python3 << 'PYTHON_SCRIPT'
import pandas as pd
import glob, os, sys

RESULT_DIR = os.environ["RESULT_DIR"]

agg_files = glob.glob(os.path.join(RESULT_DIR, "aggregator_metric", "*.parquet"))
if not agg_files:
    print("错误: aggregator_metric 目录下未找到 parquet 文件")
    sys.exit(1)

df = pd.read_parquet(agg_files[0])

scenario_rows = df[df["scenario"] != "final_score"]
final_rows = df[df["scenario"] == "final_score"]

metric_cols = [
    "drivable_area_compliance", "driving_direction_compliance",
    "ego_is_comfortable", "ego_is_making_progress",
    "ego_progress_along_expert_route", "no_ego_at_fault_collisions",
    "speed_limit_compliance", "time_to_collision_within_bound", "score"
]

metric_names = {
    "drivable_area_compliance":      "可行驶区域合规",
    "driving_direction_compliance":  "行驶方向合规",
    "ego_is_comfortable":            "舒适性",
    "ego_is_making_progress":        "前进性",
    "ego_progress_along_expert_route": "沿专家路线前进",
    "no_ego_at_fault_collisions":    "无碰撞",
    "speed_limit_compliance":        "限速合规",
    "time_to_collision_within_bound": "TTC",
    "score":                         "Overall Score"
}

print("=" * 65)
print("  nuPlan Simulation Results Summary")
print(f"  Dir: {os.path.basename(RESULT_DIR)}")
print("=" * 65)
print(f"  Total scenarios: {len(scenario_rows)}")
print()

print("--- Final Aggregated Scores ---")
for col in metric_cols:
    label = f"{col} ({metric_names.get(col, '')})"
    if len(final_rows) > 0:
        val = final_rows[col].iloc[0]
        val_str = f"{float(val):.4f}" if val is not None else "N/A"
    else:
        vals = pd.to_numeric(scenario_rows[col], errors="coerce")
        val_str = f"{vals.mean():.4f}"
    print(f"  {label:60s}: {val_str}")

scores = pd.to_numeric(scenario_rows["score"], errors="coerce")
print()
print("--- Score Statistics ---")
print(f"  Mean:   {scores.mean():.4f}")
print(f"  Median: {scores.median():.4f}")
print(f"  Std:    {scores.std():.4f}")
print(f"  Perfect (1.0): {(scores == 1.0).sum():>4d} / {len(scores)}")
print(f"  Failed  (0.0): {(scores == 0.0).sum():>4d} / {len(scores)}")

print()
print("--- Score by Scenario Type ---")
scenario_rows = scenario_rows.copy()
scenario_rows["score_num"] = pd.to_numeric(scenario_rows["score"], errors="coerce")
type_scores = scenario_rows.groupby("scenario_type")["score_num"].agg(["mean", "count"]).sort_values("mean", ascending=False)
for idx, row in type_scores.iterrows():
    print(f"  {idx:55s}: {row['mean']:.4f}  (n={int(row['count'])})")

print()
print("--- Runner Report ---")
runner_file = os.path.join(RESULT_DIR, "runner_report.parquet")
if os.path.exists(runner_file):
    rr = pd.read_parquet(runner_file)
    print(f"  Succeeded: {rr['succeeded'].sum()} / {len(rr)}")
    print(f"  Failed:    {(~rr['succeeded']).sum()} / {len(rr)}")
    print(f"  Avg inference time (mean):   {rr['compute_trajectory_runtimes_mean'].mean():.4f} s")
    print(f"  Avg inference time (median): {rr['compute_trajectory_runtimes_median'].mean():.4f} s")
else:
    print("  runner_report.parquet not found")

print()
print("=" * 65)
PYTHON_SCRIPT
