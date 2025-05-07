import torch

from hloc.networks.roma.roma import RoMa as get_roma

from ..utils.base_model import BaseModel

class RoMa(BaseModel):
    def _init(self, conf):
        # state_dict = torch.load(checkpoints_path, map_location='cpu')
        ckpt = 'gim_roma_100h.ckpt'
        model = get_roma(img_size=[672])
    
    def _forward(self, data):
        pass