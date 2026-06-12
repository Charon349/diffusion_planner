import sys
import json
import torch
import torch.nn as nn

from diffusion_planner.model.diffusion_planner import Diffusion_Planner

class DummyConfig:
    pass

with open('training_log/args.json') as f:
    args_dict = json.load(f)

config = DummyConfig()
for k, v in args_dict.items():
    setattr(config, k, v)

model = Diffusion_Planner(config)
for i, (name, param) in enumerate(model.named_parameters()):
    if i == 58:
        print(f"Parameter 58 is: {name}")
        break
