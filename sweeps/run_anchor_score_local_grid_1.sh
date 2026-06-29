#!/usr/bin/env bash
set -euo pipefail

# Local/offline grid 1:
# 8 specific hyperparameter combinations (anchor_w_lon|soft_label_tau|anchor_score_loss|anchor_t_sync)

LOSS_TYPE="soft_ce"
export MASTER_PORT="${MASTER_PORT:-29561}"

NUM_EGO_ANCHORS="128"
TARGET_EFFECTIVE_BATCH_SIZE="1024"
LEARNING_RATE="5e-4"
WARM_UP_EPOCH="10"

# Format: anchor_w_lon|soft_label_tau|anchor_score_loss|anchor_t_sync
COMBOS=(
  "0.05|0.15|5|0.2"
  "0.05|0.1|10|0.15"
  "0.05|0.25|10|0.2"
  "0.1|0.15|10|0.15"
  "0.1|0.15|10|0.2"
  "0.1|0.15|5|0.15"
  "0.1|0.15|5|0.1"
  "0.025|0.15|10|0.2"
)

START_INDEX="${START_INDEX:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

compact_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//- /m}"
  value="${value//-/m}"
  value="${value//\\//_}"
  value="${value// /}"
  echo "$value"
}

trial_index=0
launched=0

for combo in "${COMBOS[@]}"; do
  IFS='|' read -r anchor_w_lon tau score_loss anchor_t <<< "$combo"

  if (( trial_index < START_INDEX )); then
    trial_index=$((trial_index + 1))
    continue
  fi

  if (( MAX_RUNS > 0 && launched >= MAX_RUNS )); then
    echo "Reached MAX_RUNS=${MAX_RUNS}; stopping."
    exit 0
  fi

  name="anchor_score_grid1_${LOSS_TYPE}_wlon$(compact_value "$anchor_w_lon")_tau$(compact_value "$tau")_sw$(compact_value "$score_loss")_lr$(compact_value "$LEARNING_RATE")_warm$(compact_value "$WARM_UP_EPOCH")_t$(compact_value "$anchor_t")_K${NUM_EGO_ANCHORS}_ebs${TARGET_EFFECTIVE_BATCH_SIZE}"

  cmd=(
    bash sweeps/run_anchor_score_sweep_trial.sh
    --name "$name"
    --wandb_mode offline
    --wandb_group "anchor_score_local_grid_1"
    --wandb_tags "local_grid,offline,anchor_score,grid1"
    --anchor_score_loss_type "$LOSS_TYPE"
    --anchor_score_soft_label_tau "$tau"
    --anchor_score_loss "$score_loss"
    --anchor_w_lon "$anchor_w_lon"
    --learning_rate "$LEARNING_RATE"
    --warm_up_epoch "$WARM_UP_EPOCH"
    --num_ego_anchors "$NUM_EGO_ANCHORS"
    --anchor_t_sync "$anchor_t"
    --target_effective_batch_size "$TARGET_EFFECTIVE_BATCH_SIZE"
  )

  echo "============================================================"
  echo "Grid 1 Trial ${trial_index}: ${name}"
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    :
  elif [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
    "${cmd[@]}" || echo "Grid 1 trial ${trial_index} failed: ${name}"
  else
    "${cmd[@]}"
  fi

  launched=$((launched + 1))
  trial_index=$((trial_index + 1))
done

echo "Finished ${launched} grid 1 trial(s)."
