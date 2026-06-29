#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

RUN_PYTHON_PATH="${RUN_PYTHON_PATH:-/mnt/workspace/miniconda3/envs/wcg-difplnr/bin/python}"
TRAIN_SET_PATH="${TRAIN_SET_PATH:-/mnt/workspace/shared/pnc/data/nuplan/data_for_diffusion}"
TRAIN_SET_LIST_PATH="${TRAIN_SET_LIST_PATH:-/mnt/workspace/users/wangchenggang/Diffusion-Planner/nuplan_train_sampled.json}"
SAVE_DIR="${SAVE_DIR:-/mnt/workspace/users/wangchenggang/Diffusion-Planner_4/results}"
MASTER_PORT="${MASTER_PORT:-29588}"

ANCHOR_PATH="${ANCHOR_PATH:-/mnt/workspace/users/wangchenggang/Diffusion-Planner/anchors/ego_anchors_128.npy}"

"$RUN_PYTHON_PATH" -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  --standalone \
  train_predictor.py \
  --train_set "$TRAIN_SET_PATH" \
  --train_set_list "$TRAIN_SET_LIST_PATH" \
  --learning_rate 5e-4 \
  --warm_up_epoch 10 \
  --num_workers 60 \
  --train_epochs 150 \
  --save_utd 5 \
  --save_dir "$SAVE_DIR" \
  --name "anchor_score_sweep" \
  --use_wandb True \
  --wandb_project "Diffusion-Planner_4" \
  --wandb_group "anchor_score_sweep" \
  --wandb_tags "sweep,anchor_score" \
  --batch_size 384 \
  --target_effective_batch_size 1024 \
  --use_ego_anchor True \
  --ego_anchor_path "$ANCHOR_PATH" \
  --ego_anchor_state_format absolute \
  --num_ego_anchors 128 \
  --ego_anchor_t_max 0.2 \
  --anchor_sampling_steps 10 \
  --use_anchor_score True \
  --diffusion_model_type x_start \
  --diffusion_supervision_type x_start \
  "$@"
