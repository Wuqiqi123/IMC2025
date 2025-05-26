import sys
from pathlib import Path
import tempfile
import torch
from PIL import Image

import logging
from hloc.utils.base_model import BaseModel
from hloc.networks.roma.roma import RoMa
from hloc.networks.dad import load_DaD

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dad(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "roma_outdoor.pth",
        "model_utils_name": "dinov2_vitl14_pretrain.pth",
        "max_keypoints": 3000,
        "coarse_res": (560, 560),
        "upsample_res": (864, 1152),
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        self.matcher = RoMa(img_size=[672])
        checkpoints_path = conf["checkpoints_path"]
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)

        self.matcher.load_state_dict(state_dict)
        logger.info(f"Load Roma model done.")

        self.matcher.symmetric = False


        self.detector = load_DaD(weights_path = conf["checkpoints_path"])
        logger.info("Load Dad + Roma model done.")

    def _forward(self, data):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        # hack: bad way to save then match
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img0,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img1,
        ):
            img0_path = temp_img0.name
            img1_path = temp_img1.name
            img0.save(img0_path)
            img1.save(img1_path)

        # Match
        warp, certainty = self.matcher.match(img0_path, img1_path, device=device)
        # Detect
        keypoints_A = self.detector.detect_from_path(
            img0_path,
            num_keypoints=self.conf["max_keypoints"],
        )["keypoints"][0]
        keypoints_B = self.detector.detect_from_path(
            img1_path,
            num_keypoints=self.conf["max_keypoints"],
        )["keypoints"][0]
        matches = self.matcher.match_keypoints(
            keypoints_A,
            keypoints_B,
            warp,
            certainty,
            return_tuple=False,
        )

        # Sample matches for estimation
        kpts1, kpts2 = self.matcher.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        offset = self.detector.topleft - 0
        kpts1, kpts2 = kpts1 - offset, kpts2 - offset
        pred = {
            "keypoints0": self.matcher._to_pixel_coordinates(keypoints_A, H_A, W_A),
            "keypoints1": self.matcher._to_pixel_coordinates(keypoints_B, H_B, W_B),
            "mkeypoints0": kpts1,
            "mkeypoints1": kpts2,
            "mconf": torch.ones_like(kpts1[:, 0]),
        }
        return pred
