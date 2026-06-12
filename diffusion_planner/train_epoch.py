from tqdm import tqdm
from contextlib import nullcontext
import torch
from torch import nn

from diffusion_planner.utils.data_augmentation import StatePerturbation   
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func


def _slice_batch(batch, start, end):
    sliced = []
    for value in batch:
        if torch.is_tensor(value) and value.ndim > 0:
            sliced.append(value[start:end])
        else:
            sliced.append(value)
    return tuple(sliced)


def _rescale_grads(model, scale):
    if scale == 1.0:
        return
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation=None):
    epoch_loss = []

    model.train()
    optimizer.zero_grad(set_to_none=True)

    target_effective_batch_size = getattr(args, "target_effective_batch_size", 0)
    world_size = ddp.get_world_size()
    if target_effective_batch_size > 0:
        if target_effective_batch_size % world_size != 0:
            raise ValueError("--target_effective_batch_size must be divisible by the DDP world size")
        local_target_batch_size = target_effective_batch_size // world_size
    else:
        local_target_batch_size = 0
    accumulated_samples = 0

    if args.ddp:
        torch.cuda.synchronize()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for raw_batch_idx, raw_batch in enumerate(data_epoch):
            raw_batch_size = raw_batch[0].shape[0]
            batch_start = 0
            while batch_start < raw_batch_size:
                if local_target_batch_size > 0:
                    take = min(raw_batch_size - batch_start, local_target_batch_size - accumulated_samples)
                    batch = _slice_batch(raw_batch, batch_start, batch_start + take)
                else:
                    take = raw_batch_size
                    batch = raw_batch

                batch_start += take
                is_last_raw_batch = raw_batch_idx == len(data_epoch) - 1
                is_last_chunk = batch_start >= raw_batch_size
                will_step = local_target_batch_size == 0 or accumulated_samples + take >= local_target_batch_size
                is_epoch_tail = is_last_raw_batch and is_last_chunk

                if args.ddp and not will_step and not is_epoch_tail:
                    sync_context = model.no_sync()
                else:
                    sync_context = nullcontext()

                '''
                data structure in batch: Tuple(Tensor)

                ego_current_state,
                ego_future_gt,

                neighbor_agents_past,
                neighbors_future_gt,

                lanes,
                lanes_speed_limit,
                lanes_has_speed_limit,

                route_lanes,
                route_lanes_speed_limit,
                route_lanes_has_speed_limit,

                static_objects,

                '''

                # prepare data
                inputs = {
                    'ego_current_state': batch[0].to(args.device),

                    'neighbor_agents_past': batch[2].to(args.device),

                    'lanes': batch[4].to(args.device),
                    'lanes_speed_limit': batch[5].to(args.device),
                    'lanes_has_speed_limit': batch[6].to(args.device),

                    'route_lanes': batch[7].to(args.device),
                    'route_lanes_speed_limit': batch[8].to(args.device),
                    'route_lanes_has_speed_limit': batch[9].to(args.device),

                    'static_objects': batch[10].to(args.device)

                }

                ego_future = batch[1].to(args.device)
                neighbors_future = batch[3].to(args.device)
                # Normalize to ego-centric
                if aug is not None:
                    inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

                # heading to cos sin
                ego_future = torch.cat(
                [
                    ego_future[..., :2],
                    torch.stack(
                        [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                    ),
                ],
                dim=-1,
                )

                mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
                neighbors_future = torch.cat(
                [
                    neighbors_future[..., :2],
                    torch.stack(
                        [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                    ),
                ],
                dim=-1,
                )
                neighbors_future[mask] = 0.
                inputs = args.observation_normalizer(inputs)
                      
                # call the mdoel
                with sync_context:
                    loss = {}

                    loss, _ = diffusion_loss_func(
                        model,
                        inputs,
                        ddp.get_model(model, args.ddp).sde,
                        (ego_future, neighbors_future, mask),
                        args.state_normalizer,
                        loss,
                        args.diffusion_model_type,
                        args.diffusion_supervision_type,
                        args.planning_hybrid_loss,
                        args
                    )

                    loss['loss'] = loss['neighbor_prediction_loss'] + args.alpha_planning_loss * loss['ego_planning_loss']

                    total_loss = loss['loss'].item()
                    if local_target_batch_size > 0:
                        backward_loss = loss['loss'] * (take / local_target_batch_size)
                        accumulated_samples += take
                    else:
                        backward_loss = loss['loss']
                        accumulated_samples = take

                    backward_loss.backward()

                if will_step or is_epoch_tail:
                    if local_target_batch_size > 0 and accumulated_samples != local_target_batch_size:
                        _rescale_grads(model, local_target_batch_size / accumulated_samples)
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    ema.update(model)
                    accumulated_samples = 0

                if args.ddp:
                    torch.cuda.synchronize()
                
                data_epoch.set_postfix(loss='{:.4f}'.format(total_loss))
                epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")
        
    return epoch_mean_loss, epoch_mean_loss['loss']
