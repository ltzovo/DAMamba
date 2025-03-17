from torchprofile import profile_macs
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings
warnings.filterwarnings("ignore")
from DAMamba import *
model=DAMamba_T()
model=model.to(torch.device("cuda:0"))
input_size = [1, 3, 224, 224]
input = torch.randn(input_size)
device = torch.device('cuda:0')
input = input.to(device)
output=model(input)
model.eval()
macs = profile_macs(model, input)
model.train()
print('model flops:', macs/1e9, 'input_size:', input_size)
print('model params:', sum([m.numel() for m in model.parameters()])/1e6)
