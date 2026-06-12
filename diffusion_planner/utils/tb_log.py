import os
from torch.utils.tensorboard import SummaryWriter

import wandb

from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def _wandb_config_from_args(args):
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, (StateNormalizer, ObservationNormalizer)):
            config[key] = value.to_dict()
        elif key == "guidance_fn":
            config[key] = None
        else:
            config[key] = value
    return config

class TensorBoardLogger():
    def __init__(self, run_name, notes, args, wandb_resume_id, save_path, rank=0):
        """
        project_name (str): wandb project name
        config: dict or argparser
        """              
        self.args = args
        self.writer = None
        self.id = None
        self.wandb_run = None
        
        if rank == 0:
            wandb_mode = args.wandb_mode
            if wandb_mode is None:
                wandb_mode = "online" if args.use_wandb else "offline"
            os.environ["WANDB_MODE"] = wandb_mode

            tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
            display_name = run_name

            wandb_writer = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=display_name,
                group=args.wandb_group,
                tags=tags,
                notes=notes,
                resume="allow",
                id = wandb_resume_id,
                sync_tensorboard=True,
                dir=f'{save_path}')
            wandb.config.update(_wandb_config_from_args(args), allow_val_change=True)
            self.id = wandb_writer.id
            self.wandb_run = wandb_writer
            
            self.writer = SummaryWriter(log_dir=f'{save_path}/tb')
    
    def log_metrics(self, metrics: dict, step: int):
       """
       metrics (dict):
       step (int, optional): epoch or step
       """
       if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def finish(self):
       if self.writer is not None:
            self.writer.close()
            wandb.finish()
