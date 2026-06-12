#!/usr/bin/env bash
set -euo pipefail

# Batch simulation runner for locally swept checkpoints.
#
# Examples:
#   bash sim_diffusion_planner_runner_batch.sh
#   MAX_RUNS=3 bash sim_diffusion_planner_runner_batch.sh
#   START_INDEX=9 CONTINUE_ON_FAILURE=1 bash sim_diffusion_planner_runner_batch.sh
#   TRAINING_LOG_ROOT=/path/to/training_log CKPT_GLOB='model_epoch_150*.pth' bash sim_diffusion_planner_runner_batch.sh
#   RUN_NAME_GLOBS='anchor_score_grid1_soft_ce_tau0p25* anchor_score_grid1_soft_ce_tau0p1*' bash sim_diffusion_planner_runner_batch.sh

TRAINING_LOG_ROOT="${TRAINING_LOG_ROOT:-/mnt/workspace/users/wangchenggang/Diffusion-Planner/results/training_log}"
CKPT_GLOB="${CKPT_GLOB:-auto}"
RUN_NAME_GLOBS="${RUN_NAME_GLOBS:-anchor_score_grid1_soft_ce_tau0p25* anchor_score_grid1_soft_ce_tau0p1*}"
START_INDEX="${START_INDEX:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

compact_tag() {
  local value="$1"
  value="${value//:/-}"
  value="${value// /_}"
  value="${value//\//_}"
  echo "$value"
}

pick_checkpoint() {
  local ckpt_dir="$1"
  local ckpt_glob="$2"
  local -a ckpt_files

  if [[ "$ckpt_glob" == "auto" ]]; then
    if [[ -f "$ckpt_dir/latest.pth" ]]; then
      echo "$ckpt_dir/latest.pth"
      return 0
    fi

    mapfile -t ckpt_files < <(
      find "$ckpt_dir" -maxdepth 1 -type f -name 'model_epoch_*_trainloss_*.pth' \
      | sed -E 's#^.*/model_epoch_([0-9]+)_trainloss_.*#\1 & #' \
      | sort -n \
      | awk '{$1=""; sub(/^ /, ""); print}'
    )
  else
    mapfile -t ckpt_files < <(find "$ckpt_dir" -maxdepth 1 -type f -name "$ckpt_glob" | sort)
  fi

  if (( ${#ckpt_files[@]} == 0 )); then
    return 1
  fi

  echo "${ckpt_files[$((${#ckpt_files[@]} - 1))]}"
}

read -r -a RUN_NAME_GLOB_LIST <<< "$RUN_NAME_GLOBS"

mapfile -t ARGS_FILES < <(
  for run_name_glob in "${RUN_NAME_GLOB_LIST[@]}"; do
    find "$TRAINING_LOG_ROOT" \
      -mindepth 3 \
      -maxdepth 3 \
      -path "$TRAINING_LOG_ROOT/$run_name_glob/*/args.json" \
      -type f
  done | sort -u
)

if (( ${#ARGS_FILES[@]} == 0 )); then
  echo "No args.json files found under $TRAINING_LOG_ROOT matching run globs: $RUN_NAME_GLOBS"
  exit 1
fi

trial_index=0
launched=0

for args_file in "${ARGS_FILES[@]}"; do
  ckpt_dir="$(dirname "$args_file")"
  run_timestamp="$(basename "$ckpt_dir")"
  run_name="$(basename "$(dirname "$ckpt_dir")")"

  if ! ckpt_file="$(pick_checkpoint "$ckpt_dir" "$CKPT_GLOB")"; then
    echo "Skipping ${ckpt_dir}: no checkpoint matching ${CKPT_GLOB}"
    continue
  fi

  if (( trial_index < START_INDEX )); then
    trial_index=$((trial_index + 1))
    continue
  fi

  if (( MAX_RUNS > 0 && launched >= MAX_RUNS )); then
    echo "Reached MAX_RUNS=${MAX_RUNS}; stopping."
    exit 0
  fi

  run_tag="$(compact_tag "${run_name}_${run_timestamp}")"

  echo "============================================================"
  echo "Simulation ${trial_index}: ${run_tag}"
  echo "ARGS_FILE=${args_file}"
  echo "CKPT_FILE=${ckpt_file}"
  echo "============================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    :
  elif [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
    ARGS_FILE="$args_file" CKPT_FILE="$ckpt_file" RUN_TAG="$run_tag" \
      bash sim_diffusion_planner_runner.sh \
      || echo "Simulation ${trial_index} failed: ${run_tag}"
  else
    ARGS_FILE="$args_file" CKPT_FILE="$ckpt_file" RUN_TAG="$run_tag" \
      bash sim_diffusion_planner_runner.sh
  fi

  launched=$((launched + 1))
  trial_index=$((trial_index + 1))
done

echo "Finished ${launched} simulation(s)."
