export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/mnt/workspace/miniconda3/envs/wcg-difplnr/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="/mnt/workspace/shared/pnc/data/nuplan/data_for_diffusion" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/mnt/workspace/users/wangchenggang/Diffusion-Planner/nuplan_train_sampled.json"
###################################

$RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node $NPROC_PER_NODE --master_port 29566 --standalone train_predictor.py \
--name "score_head_2" \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--use_ego_anchor true \
--ego_anchor_path /mnt/workspace/users/wangchenggang/Diffusion-Planner/anchors/ego_anchors_128.npy \
--ego_anchor_state_format absolute \
--num_ego_anchors 128 \
--learning_rate 5e-4 \
--warm_up_epoch 10 \
--ego_anchor_t_max 0.2 \
--anchor_sampling_steps 10 \
--num_workers 60 \
--batch_size 384 \
--train_epochs 150 \
--target_effective_batch_size 1024 \
--anchor_score_loss_type 'soft_ce' \
--anchor_score_soft_label_tau 0.1 \
--anchor_score_loss 5.0 \
--anchor_w_lat 1.0 \
--anchor_w_lon 0.05 \
--save_utd 5 \
--save_dir "/mnt/workspace/users/wangchenggang/Diffusion-Planner_2/results"


