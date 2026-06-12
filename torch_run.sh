export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/mnt/workspace/miniconda3/envs/wcg-difplnr/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="/mnt/workspace/shared/pnc/data/nuplan/data_for_diffusion" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/mnt/workspace/users/yangjianding/Diffusion-Planner/diffusion_planner_training.json"
###################################

$RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node $NPROC_PER_NODE --master_port 29500 --standalone train_predictor.py \
--name "ego_anchor_128" \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--use_ego_anchor true \
--ego_anchor_path /mnt/workspace/users/wangchenggang/Diffusion-Planner/anchors/ego_anchors_128.npy \
--ego_anchor_state_format absolute \
--num_ego_anchors 128 \
--learning_rate 3e-4 \
--warm_up_epoch 10 \
--ego_anchor_t_max 0.2 \
--anchor_sampling_steps 10 \
--num_workers 60 \
--batch_size 384 \
--train_epochs 150 \
--save_utd 5 \
--save_dir "/mnt/workspace/users/wangchenggang/Diffusion-Planner/results"


