export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/mnt/workspace/users/wangchenggang/nuplan-devkit}"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/mnt/workspace/shared/pnc/data/nuplan/dataset}"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/workspace/shared/pnc/data/nuplan/dataset/maps}" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/mnt/workspace/users/wangchenggang/Diffusion-Planner_4/val_simulation}" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="${SPLIT:-val14}"  # e.g., "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="${CHALLENGE:-closed_loop_reactive_agents}"  # e.g., "closed_loop_nonreactive_agents"
###################################


BRANCH_NAME="${BRANCH_NAME:-diffusion_planner_release}"
ARGS_FILE="${ARGS_FILE:-/mnt/workspace/users/wangchenggang/Diffusion-Planner_3/results/training_log/l2_loss_decouple/2026-06-15-09:41:26/args.json}"
CKPT_FILE="${CKPT_FILE:-/mnt/workspace/users/wangchenggang/Diffusion-Planner_3/results/training_log/l2_loss_decouple/2026-06-15-09:41:26/latest.pth}"

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
    DATA_ROOT="/mnt/workspace/shared/pnc/data/nuplan/dataset/nuplan-v1.1/splits/trainval"
else
    SCENARIO_BUILDER="nuplan_challenge"
    DATA_ROOT="/mnt/workspace/shared/pnc/data/nuplan/dataset/nuplan-v1.1/splits/test"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER="${PLANNER:-diffusion_planner}"
if [ -z "${RUN_TAG:-}" ]; then
    CKPT_DIR="$(dirname "$CKPT_FILE")"
    RUN_TIMESTAMP="$(basename "$CKPT_DIR")"
    RUN_NAME="$(basename "$(dirname "$CKPT_DIR")")"
    RUN_TAG="${RUN_NAME}_${RUN_TIMESTAMP}"
fi
EXPERIMENT_UID="${PLANNER}/${SPLIT}/${BRANCH_NAME}/${RUN_TAG}_${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S")"
export ANCHOR_LOG_DIR="${ANCHOR_LOG_DIR:-$NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/$EXPERIMENT_UID/anchor_selection_logs}"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_builder.data_root=$DATA_ROOT \
    scenario_filter=$SPLIT \
    experiment_uid=$EXPERIMENT_UID \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=80 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.025 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"
