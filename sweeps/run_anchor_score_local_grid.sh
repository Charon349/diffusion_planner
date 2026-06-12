#!/usr/bin/env bash
set -euo pipefail

# Local/offline hyperparameter grid search.
# This does not use wandb agent, so it can run on servers without internet access.
#
# Useful controls:
#   START_INDEX=0 MAX_RUNS=6 bash sweeps/run_anchor_score_local_grid.sh
#   DRY_RUN=1 bash sweeps/run_anchor_score_local_grid.sh
#   CONTINUE_ON_FAILURE=1 bash sweeps/run_anchor_score_local_grid.sh

LOSS_TYPES=("focal" "ce" "soft_ce")
SOFT_LABEL_TAUS=("0.5" "1.0" "2.0")
ANCHOR_SCORE_LOSSES=("0.05" "0.1" "0.2")
NUM_EGO_ANCHORS=("128")
ANCHOR_T_SYNCS=("0.2" "0.3")
TARGET_EFFECTIVE_BATCH_SIZE="2048"

# Fixed by default. Add more values here if you also want to sweep them.
LEARNING_RATES=("5e-4")
WARM_UP_EPOCHS=("10")

START_INDEX="${START_INDEX:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

compact_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//- /m}"
  value="${value//-/m}"
  value="${value//\//_}"
  value="${value// /}"
  echo "$value"
}

trial_index=0
launched=0

for loss_type in "${LOSS_TYPES[@]}"; do
  for tau in "${SOFT_LABEL_TAUS[@]}"; do
    for score_loss in "${ANCHOR_SCORE_LOSSES[@]}"; do
      for num_anchors in "${NUM_EGO_ANCHORS[@]}"; do
        for anchor_t in "${ANCHOR_T_SYNCS[@]}"; do
          for lr in "${LEARNING_RATES[@]}"; do
            for warmup in "${WARM_UP_EPOCHS[@]}"; do
                if (( trial_index < START_INDEX )); then
                  trial_index=$((trial_index + 1))
                  continue
                fi

                if (( MAX_RUNS > 0 && launched >= MAX_RUNS )); then
                  echo "Reached MAX_RUNS=${MAX_RUNS}; stopping."
                  exit 0
                fi

                name="anchor_score_local_${loss_type}_tau$(compact_value "$tau")_sw$(compact_value "$score_loss")_lr$(compact_value "$lr")_warm$(compact_value "$warmup")_t$(compact_value "$anchor_t")_K${num_anchors}_ebs${TARGET_EFFECTIVE_BATCH_SIZE}"

                cmd=(
                  bash sweeps/run_anchor_score_sweep_trial.sh
                  --name "$name"
                  --wandb_mode offline
                  --wandb_group "anchor_score_local_grid"
                  --wandb_tags "local_grid,offline,anchor_score"
                  --anchor_score_loss_type "$loss_type"
                  --anchor_score_soft_label_tau "$tau"
                  --anchor_score_loss "$score_loss"
                  --learning_rate "$lr"
                  --warm_up_epoch "$warmup"
                  --num_ego_anchors "$num_anchors"
                  --anchor_t_sync "$anchor_t"
                  --target_effective_batch_size "$TARGET_EFFECTIVE_BATCH_SIZE"
                )

                echo "============================================================"
                echo "Trial ${trial_index}: ${name}"
                echo "Command: ${cmd[*]}"
                echo "============================================================"

                if [[ "$DRY_RUN" == "1" ]]; then
                  :
                elif [[ "$CONTINUE_ON_FAILURE" == "1" ]]; then
                  "${cmd[@]}" || echo "Trial ${trial_index} failed: ${name}"
                else
                  "${cmd[@]}"
                fi

                launched=$((launched + 1))
                trial_index=$((trial_index + 1))
            done
          done
        done
      done
    done
  done
done

echo "Finished ${launched} trial(s)."
