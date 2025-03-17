import torch
import os
import numpy as np
import torch
# Take the checkpoint file on imagenet-1k to get pretrained weights that can be used for downstream tasks

weights_path = "path/best_ckpt.pth"
state_dict = torch.load(weights_path)
torch.save(state_dict['model'], 'path/DAMamba.pth')