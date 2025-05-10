import sys
sys.path.insert(0, 'facebookresearch_dinov2_main')
from dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14

import torch

device = 'cuda:0'
ckpt = '/workspace/work/local/IMC2025/ckpts/dinov2_vitb14_pretrain.pth'
model = dinov2_vitb14(ckpt=ckpt).to(device).eval()

im = torch.rand([1,3,322,322]).to(device)
out = model(im)
print(out.shape)