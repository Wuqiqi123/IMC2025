import os
import torch
import numpy as np
import argparse
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from evaluation.ba import run_vggt_with_ba

use_fp16 = True
def load_model(model_path, device):
    model = VGGT(include_point_head=False, return_intermediate=True)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    return model.to(device)

def infer_with_vggtba(image_folder, model_path, output_file, use_fp16=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = load_model(model_path, device)
    if use_fp16:
        model = model.half()

    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    if len(image_files) < 2:
        raise ValueError("Need at least 2 images for VGGTBA")

    print(f"[INFO] Found {len(image_files)} images. Running inference...")

    images = load_and_preprocess_images(image_files)
    if use_fp16:
        images = images.half()
    
    images = images.to(device)

    with torch.no_grad():
        # with torch.cuda.amp.autocast(dtype=dtype):
            extrinsics = run_vggt_with_ba(model, images, image_names=image_files, dtype=dtype, use_fp16=use_fp16)

    np.save(output_file, extrinsics.cpu().numpy())
    print(f"[INFO] Saved extrinsics to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to VGGT .pt model")
    parser.add_argument("--output", type=str, default="extrinsics.npy", help="Path to save output .npy")
    args = parser.parse_args()

    infer_with_vggtba(args.image_folder, args.model_path, args.output)
