
import sys
import os

import torch
from backbones import ResNet, DinoV2
from boq.src.boq import BoQ
import cv2
import glob
import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, 'facebookresearch_dinov2_main')

class VPRModel(torch.nn.Module):
    def __init__(self, 
                 backbone,
                 aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        
    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x, attns


AVAILABLE_BACKBONES = {
    # this list will be extended
    # "resnet18": [8192 , 4096],
    "resnet50": [16384],
    "dinov2": [12288],
}

MODEL_URLS = {
    "resnet50_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    "dinov2_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
    # "resnet50_4096": "",
}

def get_trained_boq(backbone_name="resnet50", output_dim=16384, ckpt=None):
    
    if backbone_name not in AVAILABLE_BACKBONES:
        raise ValueError(f"backbone_name should be one of {list(AVAILABLE_BACKBONES.keys())}")
    try:
        output_dim = int(output_dim)
    except:
        raise ValueError(f"output_dim should be an integer, not a {type(output_dim)}")
    if output_dim not in AVAILABLE_BACKBONES[backbone_name]:
        raise ValueError(f"output_dim should be one of {AVAILABLE_BACKBONES[backbone_name]}")
    
    if "dinov2" in backbone_name:
        # load the backbone
        backbone = DinoV2()
        # load the aggregator
        aggregator = BoQ(
            in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=384,
            num_queries=64,
            num_layers=2,
            row_dim=output_dim//384, # 32 for dinov2
        )
        
    elif "resnet" in backbone_name:
        backbone = ResNet(
                backbone_name=backbone_name,
                crop_last_block=True,
            )
        aggregator = BoQ(
                in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
                proj_channels=512,
                num_queries=64,
                num_layers=2,
                row_dim=output_dim//512, # 32 for resnet
            )

    vpr_model = VPRModel(
            backbone=backbone,
            aggregator=aggregator
        )
    
    vpr_model.load_state_dict(
        torch.load(ckpt)
    )
    return vpr_model


def input_transform(image_size):
    return T.Compose([
        T.ToTensor(),
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    


def find_topk_similar(features, k):
    """
    features: Tensor of shape [N, D], must be L2-normalized
    return: topk indices per row, excluding self
    """
    sim_matrix = features @ features.T  # shape [N, N]
    
    # 排除自身（对角置为 -inf）
    sim_matrix.fill_diagonal_(-float('inf'))

    # 取每一行 top-k 相似的索引
    topk_sim, topk_idx = torch.topk(sim_matrix, k=k, dim=1)

    return topk_idx, topk_sim

@torch.no_grad()
def boq_sort_topk(image_list, model, device, topk=20, vis=False,  vis_save_dir='vis_show_boq', half=False):
    # NOTE: when using our models, use the following transform for best results.
    im_size = (322, 322) # to be used with DinoV2 backbone
    trans = input_transform(im_size)
    features = []
    for fn in tqdm(image_list):
        img = Image.open(fn).convert("RGB")   # 转换为 RGB 模式

        img = trans(img)
        if half:
            img = img.half()
        img = img.to(device)
        g_feature = model(img.unsqueeze(0))[0].detach()  # shape [1, D]
        features.append(g_feature)

    k = min(len(image_list) - 1, topk)
    features = torch.cat(features, dim=0)  # shape [N, D]
    start = time.time()
    topk_idx, topk_sim = find_topk_similar(features, k)
    end = time.time()
    print(f'find topk time: {end -start} s')
    res = {}
    for i in range(topk_idx.shape[0]):
        res[image_list[i]] = [[topk_sim[i][j].item(), image_list[topk_idx[i][j]]] for j in range(k)]
        if vis:
            os.makedirs(vis_save_dir, exist_ok=True)
            img_paths = [f"{image_list[i]}"]
            img_paths += [f"{image_list[topk_idx[i][j]]}" for j in range(k)]

            images = [cv2.imread(p) for p in img_paths if os.path.exists(p)]
            images = [cv2.resize(img, (400, 300)) for img in images]

            concat = cv2.hconcat(images)
            save_path = os.path.join(vis_save_dir, f"{image_list[i]}_top{k}.jpg")
            cv2.imwrite(save_path, concat)

    return res    
