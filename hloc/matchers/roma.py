import sys
from pathlib import Path
import subprocess
import logging
import torch
from PIL import Image
from hloc.utils.base_model import BaseModel

from hloc.networks.roma.roma import RoMa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class Roma(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "max_keypoints": 6000,
        'max_num_matches': None,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        self.net = RoMa(img_size=[672])
        checkpoints_path = conf["checkpoints_path"]
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)

        self.net.load_state_dict(state_dict)
        logger.info(f"Load Roma model done.")

    def _forward(self, data):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        # Match
        warp, certainty = self.net.match(img0, img1, device=device)
        # Sample matches for estimation
        matches, certainty = self.net.sample(
            warp, certainty, num=self.conf["max_keypoints"]
        )
        kpts1, kpts2 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        pred = {}
        pred["keypoints0"], pred["keypoints1"] = kpts1, kpts2
        pred['scores'] = certainty

        return pred
