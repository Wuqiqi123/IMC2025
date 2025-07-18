from mast3r.model import AsymmetricMASt3R
import os
import numpy as np
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import pandas as pd
import dataclasses
from pathlib import Path
import gc
import shutil
from matplotlib import pyplot as plt
import torchvision.transforms as tvf
import h5py
import scipy.cluster.hierarchy as sch
from copy import deepcopy
import PIL
import PIL.Image
from tqdm import tqdm
import torch
import subprocess
import glob
import matplotlib.pyplot as pl
import imageio.v2 as iio
import time
from transformers import AutoImageProcessor, AutoModel
from boq.boq_infer import get_trained_boq, boq_sort_topk
import json
import torchvision
from scipy.cluster.hierarchy import DisjointSet
from dust3r.dust3r.utils.geometry import geotrf 
from dust3r.dust3r.utils.device import to_cpu, to_numpy, todevice
from mast3r.fast_nn import extract_correspondences_nonsym
from mast3r.cloud_opt.sparse_ga import mkdir_for, extract_correspondences
from dust3r.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)

from hloc import gim
from hloc import match_dense

from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.utils.parsers import names_to_pair, parse_retrieval
from hloc.utils.io import get_keypoints, get_matches
from hloc.utils.database import COLMAPDatabase
import pycolmap
import random


def set_seed(seed_value=1177):
    """Sets the seed for random number generators in torch, numpy, and random."""
    torch.manual_seed(seed_value)
    
    # Sets the seed for all GPUs, if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        #Ensures that CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    np.random.seed(seed_value)
    random.seed(seed_value)

set_seed(1177)

half = True
device = "cuda:0"

# Set is_train=True to run the notebook on the training data.
# Set is_train=False if submitting an entry to the competition (test data is hidden, and different from what you see on the "test" folder).
is_train = True
data_dir = 'data/image-matching-challenge-2025'
workdir = 'result/'
os.makedirs(workdir, exist_ok=True)
workdir = Path(workdir)
max_images = None  # Used For debugging only. Set to None to disable.
datasets_to_process = None  # Not the best convention, but None means all datasets.
image_size = 224
patch_size = 16


if is_train:
    sample_submission_csv = os.path.join(data_dir, 'train_labels.csv')
else:
    sample_submission_csv = os.path.join(data_dir, 'sample_submission.csv')


mast_model = AsymmetricMASt3R.from_pretrained("ckpts/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
if half:
    mast_model.half().to(device)
else:
    mast_model.to(device)

mast_model = torch.compile(mast_model)

boq_model = get_trained_boq(backbone_name="dinov2", output_dim=12288, ckpt='ckpts/dinov2_12288.pth')
if half:
    boq_model.half().to(device)
else:
    boq_model.to(device)
    
boq_model.eval()



def create_ori_model(activation: Optional[str] = "softmax"):
    from timm import create_model as timm_create_model
    check_ori_model = timm_create_model("swsl_resnext50_32x4d", pretrained=False, num_classes=4)
    check_ori_state_dict = torch.load("/kaggle/input/check-orientation-weight/2020-11-16_resnext50_32x4d.pth")["state_dict"]
    check_ori_state_dict = rename_layers(check_ori_state_dict, {"model.": ""})
    check_ori_model.load_state_dict(check_ori_state_dict)

    if activation == "softmax":
        return torch.nn.Sequential(check_ori_model, torch.nn.Softmax(dim=1))

    return check_ori_model

check_ori_model = create_ori_model()

check_ori_model.eval()
check_ori_model.to(device)


@dataclasses.dataclass
class Prediction:
    image_id: str | None  # A unique identifier for the row -- unused otherwise. Used only on the hidden test set.
    dataset: str
    filename: str
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None

samples = {}
competition_data = pd.read_csv(sample_submission_csv)
for _, row in competition_data.iterrows():
    # Note: For the test data, the "scene" column has no meaning, and the rotation_matrix and translation_vector columns are random.
    if row.dataset not in samples:
        samples[row.dataset] = []
    samples[row.dataset].append(
        Prediction(
            image_id=None if is_train else row.image_id,
            dataset=row.dataset,
            filename=row.image
        )
    )

for dataset in samples:
    print(f'Dataset "{dataset}" -> num_images={len(samples[dataset])}')


if is_train:
    # max_images = 5

    # Note: When running on the training dataset, the notebook will hit the time limit and die. Use this filter to run on a few specific datasets.
    datasets_to_process = [
    	# New data.
    	# 'amy_gardens',
    	# 'ETs',
    	# 'fbk_vineyard',
    	# 'stairs',
    	# Data from IMC 2023 and 2024.
    	# 'imc2024_dioscuri_baalshamin',
    	# 'imc2023_theather_imc2024_church',
    	# 'imc2023_heritage',
    	# 'imc2023_haiper',
    	# 'imc2024_lizard_pond',
    	# 'pt_stpeters_stpauls',
    	'pt_brandenburg_british_buckingham',
    	# 'pt_piazzasanmarco_grandplace',
    	# 'pt_sacrecoeur_trevi_tajmahal',
    ]


ratios_resolutions = {
    224: {1.0: [224, 224]},
    512: {4 / 3: [512, 384], 32 / 21: [512, 336], 16 / 9: [512, 288], 2 / 1: [512, 256], 16 / 5: [512, 160]}
}

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result

def load_rgb(image_path: Union[Path, str], lib: str = "cv2") -> np.array:
    """Load RGB image from path.

    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2` and `jpeg4py`

    Returns: 3 channel array with RGB image

    """
    import cv2 
    if Path(image_path).is_file():
        if lib == "cv2":
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError("Only cv2 are supported.")
        return image

    raise FileNotFoundError(f"File not found {image_path}")

def get_HW_resolution(H, W, maxdim, patchsize=16):
    assert maxdim in ratios_resolutions, "Error, maxdim can only be 224 or 512 for now. Other maxdims not implemented yet."
    ratios_resolutions_maxdim = ratios_resolutions[maxdim]
    mindims = set([min(res) for res in ratios_resolutions_maxdim.values()])
    ratio = W / H
    ref_ratios = np.array([*(ratios_resolutions_maxdim.keys())])
    islandscape = (W >= H)
    if islandscape:
        diff = np.abs(ratio - ref_ratios)
    else:
        diff = np.abs(ratio - (1 / ref_ratios))
    selkey = ref_ratios[np.argmin(diff)]
    res = ratios_resolutions_maxdim[selkey]
    # check patchsize and make sure output resolution is a multiple of patchsize
    if isinstance(patchsize, tuple):
        assert len(patchsize) == 2 and isinstance(patchsize[0], int) and isinstance(
            patchsize[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
        assert patchsize[0] == patchsize[1], "Error, non square patches not managed"
        patchsize = patchsize[0]
    assert max(res) == maxdim
    assert min(res) in mindims
    return res[::-1] if islandscape else res  # return HW


def get_resize_function(maxdim, patch_size, H, W, is_mask=False):
    if [max(H, W), min(H, W)] in ratios_resolutions[maxdim].values():
        return lambda x: x, np.eye(3), np.eye(3)
    else:
        target_HW = get_HW_resolution(H, W, maxdim=maxdim, patchsize=patch_size)

        ratio = W / H
        target_ratio = target_HW[1] / target_HW[0]
        to_orig_crop = np.eye(3)
        to_rescaled_crop = np.eye(3)
        if abs(ratio - target_ratio) < np.finfo(np.float32).eps:
            crop_W = W
            crop_H = H
        elif ratio - target_ratio < 0:
            crop_W = W
            crop_H = int(W / target_ratio)
            to_orig_crop[1, 2] = (H - crop_H) / 2.0
            to_rescaled_crop[1, 2] = -(H - crop_H) / 2.0
        else:
            crop_W = int(H * target_ratio)
            crop_H = H
            to_orig_crop[0, 2] = (W - crop_W) / 2.0
            to_rescaled_crop[0, 2] = - (W - crop_W) / 2.0

        crop_op = tvf.CenterCrop([crop_H, crop_W])

        if is_mask:
            resize_op = tvf.Resize(size=target_HW, interpolation=tvf.InterpolationMode.NEAREST_EXACT)
        else:
            resize_op = tvf.Resize(size=target_HW)
        to_orig_resize = np.array([[crop_W / target_HW[1], 0, 0],
                                   [0, crop_H / target_HW[0], 0],
                                   [0, 0, 1]])
        to_rescaled_resize = np.array([[target_HW[1] / crop_W, 0, 0],
                                       [0, target_HW[0] / crop_H, 0],
                                       [0, 0, 1]])

        op = tvf.Compose([crop_op, resize_op])

        return op, to_rescaled_resize @ to_rescaled_crop, to_orig_crop @ to_orig_resize

def boq_make_pairs(sfm_pairs_path, boq_topks, image_list):
    pairs_name = []
    for query, topk in boq_topks.items():
        for top in topk:
            database = top[1] ## image name
            if (query, database) not in pairs_name and (database, query) not in pairs_name:
                pairs_name.append((query, database))

    print(f"Found {len(pairs_name)} pairs.")
    with open(sfm_pairs_path, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs_name))


def make_cluster_paris(sfm_pairs_path, image_list, distance_matrix, image_name_dict):
    pairs_name = []
    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            if distance_matrix[image_name_dict[image_list[i]], image_name_dict[image_list[j]]] < 2.2:
                pairs_name.append((image_list[i], image_list[j]))
                print(f"Found {len(pairs_name)} pairs.")
    with open(sfm_pairs_path, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs_name))

    return len(pairs_name)


@torch.no_grad()
def mast_inference(cache_path, model, img1, img2, device, half=False):
    infer_path = cache_path / f"forward_{img1['instance']}_{img2['instance']}.pth"
    if infer_path.exists():
        pred1, pred2 = torch.load(infer_path)
        return pred1, pred2
    
    infer_path_rev = cache_path / f"forward_{img2['instance']}_{img1['instance']}.pth"
    if infer_path_rev.exists():
        pred2, pred1 = torch.load(infer_path_rev)
        return pred1, pred2


    shape1 = torch.from_numpy(img1['true_shape']).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2['true_shape']).to(device, non_blocking=True)
    if half:
        img1 = img1['img'].half().to(device, non_blocking=True)
        img2 = img2['img'].half().to(device, non_blocking=True)
    else:
        img1 = img1['img'].to(device, non_blocking=True)
        img2 = img2['img'].to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2, half=False):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float().half() if half else tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float().half() if half else tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    pred1, pred2 = decoder(feat1, feat2, pos1, pos2, shape1, shape2, half=half)
    pred1 = to_cpu(pred1)
    pred2 = to_cpu(pred2)
    
    pred1["pts3d"] = pred1["pts3d"].squeeze(0)
    pred2["pts3d"] = pred2["pts3d"].squeeze(0)

    pred1["conf"] = pred1["conf"].squeeze(0)
    pred2["conf"] = pred2["conf"].squeeze(0)

    pred1["desc"] = pred1["desc"].squeeze(0)
    pred2["desc"] = pred2["desc"].squeeze(0)

    pred1["desc_conf"] = pred1["desc_conf"].squeeze(0)
    pred2["desc_conf"] = pred2["desc_conf"].squeeze(0)

    torch.save((pred1, pred2), infer_path)

    return pred1, pred2

def symmetric_inference(model, img1, img2, device, half=False):
    shape1 = torch.from_numpy(img1['true_shape']).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2['true_shape']).to(device, non_blocking=True)
    if half:
        img1 = img1['img'].half().to(device, non_blocking=True)
        img2 = img2['img'].half().to(device, non_blocking=True)
    else:
        img1 = img1['img'].to(device, non_blocking=True)
        img2 = img2['img'].to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2, half=False):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float().half() if half else tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float().half() if half else tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2, half=half)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1, half=half)

    return (res11, res21, res22, res12)

@torch.no_grad()
def forward_mast3r(cache_path, model, img1, img2, desc_conf='desc_conf',
                   device='cuda', subsample=8, half=False, **matching_kw):

    idx1 = img1['instance']
    idx2 = img2['instance']

    path_corres = cache_path + f'/corres_={desc_conf}_{subsample=}/{idx1}-{idx2}.pth'
    path_corres2 = cache_path + f'/corres_={desc_conf}_{subsample=}/{idx2}-{idx1}.pth'

    if os.path.isfile(path_corres2) and not os.path.isfile(path_corres):
        score, (xy1, xy2, confs) = torch.load(path_corres2)
        return score, (xy2, xy1, confs)
        

    res = symmetric_inference(model, img1, img2, device=device, half=half)
    # X11, X21, X22, X12 = [r['pts3d'][0] for r in res]
    C11, C21, C22, C12 = [r['conf'][0] for r in res]
    descs = [r['desc'][0] for r in res]
    qonfs = [r[desc_conf][0] for r in res]

    # perform reciprocal matching
    corres = extract_correspondences(descs, qonfs, device=device, subsample=subsample)

    conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
    matching_score = (float(conf_score), float(corres[2].sum()), len(corres[2]))

    if cache_path is not None:
        torch.save((matching_score, corres), mkdir_for(path_corres))

    return corres

def convert_im_matches_pairs(img0, img1, image_to_colmap, im_keypoints, matches_im0, matches_im1, viz):
    if viz:
        from matplotlib import pyplot as pl

        image_mean = torch.as_tensor(
            [0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor(
            [0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        rgb0 = img0['img'] * image_std + image_mean
        rgb0 = torchvision.transforms.functional.to_pil_image(rgb0[0])
        rgb0 = np.array(rgb0)

        rgb1 = img1['img'] * image_std + image_mean
        rgb1 = torchvision.transforms.functional.to_pil_image(rgb1[0])
        rgb1 = np.array(rgb1)

        imgs = [rgb0, rgb1]
        # visualize a few matches
        n_viz = 100
        num_matches = matches_im0.shape[0]
        match_idx_to_viz = np.round(np.linspace(
            0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        rgb0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)),
                                (0, 0), (0, 0)), 'constant', constant_values=0)
        rgb1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)),
                                (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((rgb0, rgb1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for ii in range(n_viz):
            (x0, y0), (x1,
                       y1) = viz_matches_im0[ii].T, viz_matches_im1[ii].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(ii /
                    (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)

    matches = [matches_im0.astype(np.float64), matches_im1.astype(np.float64)]
    imgs = [img0, img1]
    imidx0 = img0['idx']
    imidx1 = img1['idx']
    ravel_matches = []
    for j in range(2):
        H, W = imgs[j]['true_shape'][0]
        with np.errstate(invalid='ignore'):
            qx, qy = matches[j].round().astype(np.int32).T
        ravel_matches_j = qx.clip(min=0, max=W - 1, out=qx) + W * qy.clip(min=0, max=H - 1, out=qy)
        ravel_matches.append(ravel_matches_j)
        imidxj = imgs[j]['idx']
        for m in ravel_matches_j:
            if m not in im_keypoints[imidxj]:
                im_keypoints[imidxj][m] = 0
            im_keypoints[imidxj][m] += 1
    imid0 = copy.deepcopy(image_to_colmap[imidx0]['colmap_imid'])
    imid1 = copy.deepcopy(image_to_colmap[imidx1]['colmap_imid'])
    if imid0 > imid1:
        colmap_matches = np.stack([ravel_matches[1], ravel_matches[0]], axis=-1)
        imid0, imid1 = imid1, imid0
        imidx0, imidx1 = imidx1, imidx0
    else:
        colmap_matches = np.stack([ravel_matches[0], ravel_matches[1]], axis=-1)
    colmap_matches = np.unique(colmap_matches, axis=0)
    return imidx0, imidx1, colmap_matches


def get_im_matches_conf(cache_path, pred1, pred2, pairs, conf_thr=1.001,
                   subsample=8, pixel_tol=0, viz=False, device='cuda'):
    corres = load_corres(cache_path, pred1, pred2, pairs, conf_thr=conf_thr,
                        subsample=subsample, pixel_tol=pixel_tol, device=device)

    conf = corres[2]
    mask = conf >= conf_thr
    conf = conf[mask].cpu().numpy()
    return conf


def load_corres(cache_path, pred1, pred2, pairs, conf_thr=1.001, 
                subsample=8, pixel_tol=0, device='cuda'):
    corres_path = cache_path / f'corres_{pairs[0]["instance"]}_{pairs[1]["instance"]}.pth'
    corres_path_rev = cache_path / f'corres_{pairs[1]["instance"]}_{pairs[0]["instance"]}.pth'
    if corres_path.exists():
        corres = torch.load(corres_path)
        corres = todevice(corres, device)
    elif corres_path_rev.exists():
        corres = torch.load(corres_path_rev)
        corres = todevice(corres, device)
        return corres[1], corres[0], corres[2]
    else:
        corres = extract_correspondences_nonsym(pred1['desc'], pred2['desc'], pred1['desc_conf'], pred2['desc_conf'],
                                                device=device, subsample=subsample, pixel_tol=pixel_tol)
        torch.save(corres, corres_path)

    return corres


def get_im_matches(cache_path, pred1, pred2, pairs, image_to_colmap, im_keypoints, conf_thr=1.001,
                   subsample=8, pixel_tol=0, viz=False, device='cuda'):
    im_match = {}
    corres = load_corres(cache_path, pred1, pred2, pairs, conf_thr=conf_thr,
                            subsample=subsample, pixel_tol=pixel_tol, device=device)
    conf = corres[2]
    mask = conf >= conf_thr
    matches_im0 = corres[0][mask].cpu().numpy()
    matches_im1 = corres[1][mask].cpu().numpy()
    if len(matches_im0) == 0:
        return im_match
    
    imidx0, imidx1, colmap_matches = convert_im_matches_pairs(pairs[0], pairs[1], image_to_colmap, im_keypoints,
                                                              matches_im0, matches_im1, viz=viz)
    im_match[(imidx0, imidx1)] = colmap_matches
    return im_match


def scene_prepare_images(root, maxdim, patch_size, image_paths):
    images = []
    image_name_dict = {}
    # image loading
    for idx in tqdm(range(len(image_paths))):
        rgb_image = PIL.Image.open(os.path.join(root, image_paths[idx])).convert('RGB')

        # resize images
        W, H = rgb_image.size
        resize_func, _, to_orig = get_resize_function(maxdim, patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))

        # image dictionary
        images.append({'img': rgb_tensor.unsqueeze(0),
                       'true_shape': np.int32([rgb_tensor.shape[1:]]),
                       'to_orig': to_orig,
                       'idx': idx,
                       'instance': image_paths[idx],
                       'orig_shape': np.int32([H, W])})
        image_name_dict[image_paths[idx]] = idx
    return images, image_name_dict


def can_unoin(a, b, distance_matrix, distance_th = 3.1, min_simular_num = 4, verbose=False):
    simular_num = 0
    for i in a:
        for j in b:
            if verbose:
                print(f"Checking distance between {i} and {j}: {distance_matrix[i, j]}")
            if distance_matrix[i, j] < 1.1:
                simular_num += 4
            elif distance_matrix[i, j] < 2.1:
                simular_num += 2
            elif distance_matrix[i, j] < distance_th:
                simular_num += 1

    if simular_num >= min_simular_num:
        return True
    else:
        return False


def try_union(clusters_list, distance_matrix):
    for i in range(len(clusters_list)):
        for j in range(i+1, len(clusters_list)):
            if can_unoin(clusters_list[i], clusters_list[j], distance_matrix):
                return (i, j)

    return ()
    

def find_cluster(distance_matrix, name_list, show_dendrogram = False):
    # Compute the condensed distance matrix
    condensed_distance_matrix = sch.distance.squareform(distance_matrix)

    if show_dendrogram:
        plt.matshow(distance_matrix)
        plt.colorbar()
        plt.show()

        plt.figure(figsize=(50, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')

    # Perform hierarchical clustering using the linkage method
    Z = sch.linkage(condensed_distance_matrix, method="average")
    names = [name.split('.')[0] for name in name_list] 
    if show_dendrogram:
        sch.dendrogram(Z, leaf_rotation=90., leaf_font_size=5, labels=names)
        plt.show()

    clusters = sch.fcluster(Z, t=5, criterion='distance')
    clusters_dict = {}

    clusters_bucket = {}
    
    print(f'clusters = {clusters}')
    for i, cluster in enumerate(clusters):
        if cluster not in clusters_bucket:
            clusters_bucket[cluster] = set()
        clusters_bucket[cluster].add(i)

    clusters_list = list(clusters_bucket.values())
    clusters_list = sorted(clusters_list, key=len, reverse=True)

    while True:
        union = try_union(clusters_list, distance_matrix)
        if union == ():
            break
        clusters_list[union[0]].update(clusters_list[union[1]])
        clusters_list.pop(union[1])

    clusters_list = filter(lambda x: len(x) > 8, clusters_list)     

    for i, cluster in enumerate(clusters_list):
        clusters_dict[i] = dict(names=[])
        for idx in cluster:
            clusters_dict[i]["names"].append(name_list[idx])
    
    return clusters_dict

def mast_find_cluster(cache_path, mast_model, images, image_name_dict, 
                      device, pairs_path, subsample = 8, conf_thr = 1.001, half=True, pixel_tol=5):

    im_keypoints = {idx: {} for idx in range(len(image_names))}
    pairs = []
    assert pairs_path.exists(), pairs_path
    pairs_names = parse_retrieval(pairs_path)
    pairs_names = [(q, r) for q, rs in pairs_names.items() for r in rs]
    for i, j in pairs_names:
        pairs.append((images[image_name_dict[i]], images[image_name_dict[j]]))

    pairwise_scores = torch.zeros((len(images), len(images)), device=device)
    for img1, img2 in tqdm(pairs,  desc='Mast inference'):
        pred1, pred2 = mast_inference(cache_path, mast_model, img1, img2, device, half=half)
        conf = get_im_matches_conf(cache_path, pred1=pred1, pred2=pred2, pairs=(img1, img2), 
                                   subsample=subsample, conf_thr=conf_thr,
                                    pixel_tol=pixel_tol, viz=False)
        pairwise_scores[img1['idx'], img2['idx']] = conf.size * conf.mean() if conf.size > 0 else 0
        pairwise_scores[img2['idx'], img1['idx']] = conf.size * conf.mean() if conf.size > 0 else 0
        

    imsizes = [torch.from_numpy(img['true_shape']) for img in images]
    imsizes = torch.concat(imsizes, dim=0).to(device)

    # Convert the affinity matrix to a distance matrix (if needed)
    n_patches = (imsizes // subsample).prod(dim=1)
    max_n_corres = 3 * torch.minimum(n_patches[:,None], n_patches[None,:])
    pws = (pairwise_scores.clone() / max_n_corres).clip(min=np.exp(-20), max=1)
    pws.fill_diagonal_(1)
    pws = to_numpy(pws)

    distance_matrix = np.where(pws <= 1.0, -np.log(pws), 20).clip(max=20)
    clusters_dict = find_cluster(distance_matrix, image_names, show_dendrogram=True)

    print("-------------------------mean of distance_matrix: ", distance_matrix.mean(), "-------------------------")

    return clusters_dict, distance_matrix


def export_matches(colmap_db_path, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification):
    db = COLMAPDatabase.connect(colmap_db_path)
    colmap_image_pairs = []
    # 2D-2D are quite dense
    # we want to remove the very small tracks
    # and export only kpt for which we have values
    # build tracks
    print("building tracks")
    keypoints_to_track_id = {}
    track_id_to_kpt_list = []
    to_merge = []
    for (imidx0, imidx1), colmap_matches in tqdm(im_matches.items()):
        if imidx0 not in keypoints_to_track_id:
            keypoints_to_track_id[imidx0] = {}
        if imidx1 not in keypoints_to_track_id:
            keypoints_to_track_id[imidx1] = {}

        for m in colmap_matches:
            if m[0] not in keypoints_to_track_id[imidx0] and m[1] not in keypoints_to_track_id[imidx1]:
                # new pair of kpts never seen before
                track_idx = len(track_id_to_kpt_list)
                keypoints_to_track_id[imidx0][m[0]] = track_idx
                keypoints_to_track_id[imidx1][m[1]] = track_idx
                track_id_to_kpt_list.append(
                    [(imidx0, m[0]), (imidx1, m[1])])
            elif m[1] not in keypoints_to_track_id[imidx1]:
                # 0 has a track, not 1
                track_idx = keypoints_to_track_id[imidx0][m[0]]
                keypoints_to_track_id[imidx1][m[1]] = track_idx
                track_id_to_kpt_list[track_idx].append((imidx1, m[1]))
            elif m[0] not in keypoints_to_track_id[imidx0]:
                # 1 has a track, not 0
                track_idx = keypoints_to_track_id[imidx1][m[1]]
                keypoints_to_track_id[imidx0][m[0]] = track_idx
                track_id_to_kpt_list[track_idx].append((imidx0, m[0]))
            else:
                # both have tracks, merge them
                track_idx0 = keypoints_to_track_id[imidx0][m[0]]
                track_idx1 = keypoints_to_track_id[imidx1][m[1]]
                if track_idx0 != track_idx1:
                    # let's deal with them later
                    to_merge.append((track_idx0, track_idx1))

    # regroup merge targets
    print("merging tracks")
    unique = np.unique(to_merge)
    tree = DisjointSet(unique)
    for track_idx0, track_idx1 in tqdm(to_merge):
        tree.merge(track_idx0, track_idx1)

    subsets = tree.subsets()
    print("applying merge")
    for setvals in tqdm(subsets):
        new_trackid = len(track_id_to_kpt_list)
        kpt_list = []
        for track_idx in setvals:
            kpt_list.extend(track_id_to_kpt_list[track_idx])
            for imidx, kpid in track_id_to_kpt_list[track_idx]:
                keypoints_to_track_id[imidx][kpid] = new_trackid
        track_id_to_kpt_list.append(kpt_list)

    # binc = np.bincount([len(v) for v in track_id_to_kpt_list])
    # nonzero = np.nonzero(binc)
    # nonzerobinc = binc[nonzero[0]]
    # print(nonzero[0].tolist())
    # print(nonzerobinc)
    num_valid_tracks = sum(
        [1 for v in track_id_to_kpt_list if len(v) >= min_len_track])

    keypoints_to_idx = {}
    print(f"squashing keypoints - {num_valid_tracks} valid tracks")
    for imidx, keypoints_imid in tqdm(im_keypoints.items()):
        imid = image_to_colmap[imidx]['colmap_imid']
        keypoints_kept = []
        keypoints_to_idx[imidx] = {}
        for kp in keypoints_imid.keys():
            if kp not in keypoints_to_track_id[imidx]:
                continue
            track_idx = keypoints_to_track_id[imidx][kp]
            track_length = len(track_id_to_kpt_list[track_idx])
            if track_length < min_len_track:
                continue
            keypoints_to_idx[imidx][kp] = len(keypoints_kept)
            keypoints_kept.append(kp)
        if len(keypoints_kept) == 0:
            continue
        keypoints_kept = np.array(keypoints_kept)
        keypoints_kept = np.unravel_index(keypoints_kept, images[imidx]['true_shape'][0])[
            0].base[:, ::-1].copy().astype(np.float32)
        # rescale coordinates
        keypoints_kept[:, 0] += 0.5
        keypoints_kept[:, 1] += 0.5
        keypoints_kept = geotrf(images[imidx]['to_orig'], keypoints_kept, norm=True)

        H, W = images[imidx]['orig_shape']
        keypoints_kept[:, 0] = keypoints_kept[:, 0].clip(min=0, max=W - 0.01)
        keypoints_kept[:, 1] = keypoints_kept[:, 1].clip(min=0, max=H - 0.01)

        db.add_keypoints(imid, keypoints_kept)

    print("exporting im_matches")
    for (imidx0, imidx1), colmap_matches in im_matches.items():
        imid0, imid1 = image_to_colmap[imidx0]['colmap_imid'], image_to_colmap[imidx1]['colmap_imid']
        assert imid0 < imid1
        final_matches = np.array([[keypoints_to_idx[imidx0][m[0]], keypoints_to_idx[imidx1][m[1]]]
                                  for m in colmap_matches
                                  if m[0] in keypoints_to_idx[imidx0] and m[1] in keypoints_to_idx[imidx1]])
        if len(final_matches) > 0:
            colmap_image_pairs.append(
                (images[imidx0]['instance'], images[imidx1]['instance']))
            db.add_matches(imid0, imid1, final_matches)
            if skip_geometric_verification:
                db.add_two_view_geometry(imid0, imid1, final_matches)

    db.commit()
    db.close()
    return colmap_image_pairs

def create_empty_db(database_path):
    if database_path.exists():
        print("The database already exists, deleting it.")
        database_path.unlink()
    print("Creating an empty database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(
    image_dir,
    database_path,
    camera_mode,
    image_list,
    options = None,
):
    print("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_list=image_list or [],
            options=options,
        )


def get_colmap_image_ids(database_path):
    db = COLMAPDatabase.connect(database_path)
    image_ids = {}
    camera_ids = {}
    for name, image_id, camera_id in db.execute("SELECT name, image_id, camera_id FROM images;"):
        image_ids[name] = image_id
        camera_ids[name] = camera_id

    db.close()
    return image_ids, camera_ids

def import_images_and_cameras(image_dir, database_path, camera_mode, image_list, image_path_to_idx):
    import_images(image_dir, database_path, camera_mode, image_list=image_list)
    image_ids, camera_ids = get_colmap_image_ids(database_path)

    image_to_colmap = {}
    for image_name, idx in image_path_to_idx.items():
        colmap_camera_id = camera_ids[image_name]
        colmap_image_id = image_ids[image_name]
        image_to_colmap[idx] = {
            "colmap_imid": colmap_image_id,
            "colmap_camid": colmap_camera_id
        }

    return image_to_colmap


def run_mast_match_cluster(cache_path, mast_model, images, image_name_cluster, 
                           image_path_to_idx, image_to_colmap, colmap_db_path,
                           device, pairs_path, subsample = 8, conf_thr = 1.001, half=True,
                           pixel_tol=5, min_len_track=5, skip_geometric_verification=False):
    im_keypoints = {idx: {} for idx in range(len(image_name_cluster))}
    pairs = []
    assert pairs_path.exists(), pairs_path
    pairs_names = parse_retrieval(pairs_path)
    pairs_names = [(q, r) for q, rs in pairs_names.items() for r in rs]
    for i, j in pairs_names:
        pairs.append((images[image_path_to_idx[i]], images[image_path_to_idx[j]]))

    im_matches = {}
    for img1, img2 in tqdm(pairs,  desc='Mast cluster inference'):
        pred1, pred2 = mast_inference(cache_path, mast_model, img1, img2, device, half=half)
        im_match = get_im_matches(cache_path, pred1=pred1, pred2=pred2, pairs=(img1, img2), image_to_colmap=image_to_colmap,
                                  subsample=subsample, conf_thr=conf_thr,
                                  im_keypoints=im_keypoints, pixel_tol=pixel_tol, viz=False)
        im_matches.update(im_match.items())

    colmap_image_pairs = export_matches(colmap_db_path, images, image_to_colmap, im_keypoints, im_matches, min_len_track=min_len_track,
                   skip_geometric_verification=skip_geometric_verification)
    return colmap_image_pairs


def lightglue_find_cluster(pairs_path, match_path, images, image_name_dict, min_match_score=0.3):
    pairwise_scores = torch.zeros((len(images), len(images)), device=device)

    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_name_dict[name0], image_name_dict[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(match_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]

        pairwise_scores[id0, id1] = matches.size
        pairwise_scores[id1, id0] = matches.size
        
        matched |= {(id0, id1), (id1, id0)}

    max_number_point = 9000
    pws = (pairwise_scores.clone() / max_number_point).clip(min=np.exp(-100), max=1)
    pws.fill_diagonal_(1)
    pws = to_numpy(pws)

    distance_matrix = np.where(pws <= 1.0, -np.log(pws), 100).clip(max=100)
    clusters_dict = find_cluster(distance_matrix, image_names, show_dendrogram=False)
    return clusters_dict


def import_matches(
    image_ids,
    database_path: Path,
    pairs_path: Path,
    matches_path: Path,
    min_match_score,
    skip_geometric_verification: bool = False,
):
    print("Importing matches into the database...")

    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    matches_size = 0
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]

        if matches.shape[0] < 10:
            continue
        
        matches_size += matches.shape[0]
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        # if skip_geometric_verification:
        db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()

    return matches_size

def hloc_reconstruction(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score =  None,
    image_list = None,
    image_options = None,
    mapper_options = None,
):
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    from hloc.reconstruction import (
        create_empty_db,
        import_features,
        import_images,
        get_image_ids
    )
    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    matches_size = import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )
    return database, matches_size

def glomap_run_mapper(colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    args = [
        'mapper',
        '--database_path',
        colmap_db_path,
        '--image_path',
        image_root_path,
        '--output_path',
        recon_path
    ]
    args.insert(0, "glomap")
    glomap_process = subprocess.Popen(args)
    glomap_process.wait()

    if glomap_process.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {glomap_process.returncode} )')


def tensor_from_rgb_image(image):
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)

@torch.no_grad()
def check_images_orientation(image_dir, new_dir, image_names):
    """
    Check if images are in the correct orientation.
    If not, rotate them to the correct orientation.
    """
    import albumentations as albu
    transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)
    images_ori = {}
    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        image = load_rgb(image_path)
        x = transform(image=image)["image"]
        temp = [tensor_from_rgb_image(x).to(device)]
        prediction = check_ori_model(torch.stack(temp)).cpu().numpy()
        images_ori[image_name] = {}
        images_ori[image_name]["pred"] = prediction[0]
        index = np.argmax(prediction[0])
        images_ori[image_name]["to_normal"] = -index * 90 if images_ori[image_name]["pred"][index] > 0.6 else 0
        images_ori[image_name]["to_origin"] = -images_ori[image_name]["to_normal"]

        ## rotate image
        if images_ori[image_name]["to_normal"] != 0:
            image = PIL.Image.open(image_path)
            image = image.rotate(images_ori[image_name]["to_normal"], expand=True)
            image.save(os.path.join(new_dir, image_name))
        else:
            shutil.copy(image_path, os.path.join(new_dir, image_name))
    print(f'Orientation check completed for {len(images_ori)} images.')

    return images_ori

for dataset, predictions in samples.items():
    if datasets_to_process and dataset not in datasets_to_process:
        print(f'Skipping "{dataset}"')
        continue
    
    images_dir = os.path.join(data_dir, 'train' if is_train else 'test', dataset)
    if not os.path.exists(images_dir):
        print(f'Images dir "{images_dir}" does not exist. Skipping "{dataset}"')
        continue
    
    images_dir = Path(images_dir)

    print(f'Images dir: {images_dir}')

    image_names = [p.filename for p in predictions]
    if max_images is not None:
        image_names = image_names[:max_images]

    print(f'\nProcessing dataset "{dataset}": {len(image_names)} images')

    sorted_image_names = sorted(image_names)

    dataset_dir = os.path.join(workdir, dataset)

    new_images_dir = os.path.join(dataset_dir, "rotate_images")
    os.makedirs(new_images_dir, exist_ok=True)
    images_ori = check_images_orientation(images_dir, new_images_dir, sorted_image_names)
    images_dir = Path(new_images_dir)
    image_names = sorted(os.listdir(images_dir))

    boq_topks = boq_sort_topk(images_dir, image_names, boq_model, device, vis=False, topk=50, half=half)
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, "boq_topk.json"), "w", encoding="utf-8") as f:
        json.dump(boq_topks, f, ensure_ascii=False, indent=4)


    sfm_pairs = workdir / dataset / "pairs-sfm.txt"
    boq_make_pairs(sfm_pairs, boq_topks, image_names)

    features = workdir / dataset / "features.h5"
    matches = workdir / dataset / "matches.h5"
    sfm_dir = workdir / dataset / "sfm"
    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]

    # extract_features.main(feature_conf, images_dir, image_list=image_names, feature_path=features)
    # match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    # clusters_dict = lightglue_find_cluster(sfm_pairs, matches, images_dir, image_names, min_match_score=0.3)

    ## then we use mast
    mast_cache_path = workdir / dataset / "mast_cache"
    os.makedirs(mast_cache_path, exist_ok=True)
    images, image_name_dict = scene_prepare_images(images_dir, 224, patch_size, image_names)
    clusters_dict, distance_matrix = mast_find_cluster(mast_cache_path, mast_model, images, image_name_dict,
                                       device, sfm_pairs, subsample=8, conf_thr=0.7, half=half, pixel_tol=0)

    
    # shutil.rmtree(mast_cache_path)
    

    filename_to_prdictions_index = {p.filename: idx for idx, p in enumerate(predictions)}

    registered = 0
    prediction_cluster_index = 0
    for cluster_id, image_cluster_dict in clusters_dict.items():
        print(f'cluster {cluster_id}:')
        for img_name in image_cluster_dict["names"]:
            print(f'-- {img_name}')
        
        if len(image_cluster_dict["names"]) < 4:
            print(f'-- outlier clusters {image_cluster_dict["names"]}')
            continue
        
        image_names_cluster = image_cluster_dict["names"]

        cluster_dir_name = f"cluster_{prediction_cluster_index}"
        os.makedirs(workdir / dataset / cluster_dir_name, exist_ok=True)
        clus_sfm_pairs = workdir / dataset / cluster_dir_name / "clu-pairs-sfm.txt"
        clus_pair_len = make_cluster_paris(clus_sfm_pairs, image_names_cluster, distance_matrix, image_name_dict)
        if clus_pair_len < 28:
            print(f'-- outlier clusters {image_cluster_dict["names"]}')
            continue

        clus_sfm_dir = workdir / dataset / cluster_dir_name / "sfm"
        os.makedirs(clus_sfm_dir, exist_ok=True)


        print(f"starting constructing {dataset} with images {image_names_cluster}")

        # lightglue
        clus_features = workdir / dataset / cluster_dir_name / "features.h5"
        clus_matches = workdir / dataset / cluster_dir_name / "matches.h5"
        extract_features.main(feature_conf, images_dir, image_list=image_names_cluster, feature_path=clus_features)
        match_features.main(matcher_conf, clus_sfm_pairs, features=clus_features, matches=clus_matches)
        mapper_options = {"min_model_size" : 5,
                          "max_num_models": 80, 
                          "num_threads": 1
        }
        max_map, maps = reconstruction.main(
            clus_sfm_dir,
            images_dir,
            clus_sfm_pairs,
            features=clus_features,
            matches=clus_matches,
            image_list=image_names_cluster,
            min_match_score=0.07,
            skip_geometric_verification=True,
            mapper_options = mapper_options
        )

        for index, image in max_map.images.items():
            prediction_index = filename_to_prdictions_index[image.name]
            predictions[prediction_index].cluster_index = prediction_cluster_index
            rotation = deepcopy(image.cam_from_world.rotation.matrix())
            translation = deepcopy(image.cam_from_world.translation)
            # if images_ori[image.name]["to_normal"] != 0:
            #     tf_c_w = np.eye(4)
            #     tf_c_w[:3, :3] = rotation
            #     tf_c_w[:3, 3] = translation
            #     tf_w_c = np.linalg.inv(tf_c_w)
            #     tf_w_c[:3, :3] = tf_w_c[:3, :3] @ Rotation.from_euler('z', -images_ori[image.name]["to_normal"], degrees=True).as_matrix() 
            #     tf_c_w = np.linalg.inv(tf_w_c)
            #     rotation = tf_w_c[:3, :3]
            #     translation = tf_w_c[:3, 3]
                
            predictions[prediction_index].rotation = rotation
            predictions[prediction_index].translation = translation
            registered += 1

        for image_name in image_names_cluster:
            prediction_index = filename_to_prdictions_index[image_name]
            if predictions[prediction_index].cluster_index is None:
                predictions[prediction_index].cluster_index = prediction_cluster_index

        prediction_cluster_index += 1
        mapping_result_str = f'Dataset "{dataset}" -> Registered {registered} / {len(image_names)} images with {len(max_map.images)} clusters'
        print(mapping_result_str)

        gc.collect()
        time.sleep(1)



array_to_str = lambda array: ';'.join([f"{x:.09f}" for x in array])
none_to_str = lambda n: ';'.join(['nan'] * n)

submission_file = 'result/submission.csv'
with open(submission_file, 'w') as f:
    if is_train:
        f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')
    else:
        f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.image_id},{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')


# Definitely Compute results if running on the training set.
# Do not do this when submitting a notebook for scoring. All you have to do is save your submission to /kaggle/working/submission.csv.
is_train = True
if is_train:
    import metric
    final_score, dataset_scores = metric.score(
        gt_csv='data/image-matching-challenge-2025/train_labels.csv',
        user_csv=submission_file,
        thresholds_csv='data/image-matching-challenge-2025/train_thresholds.csv',
        mask_csv=None if is_train else os.path.join(data_dir, 'mask.csv'),
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )