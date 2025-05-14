
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
from copy import deepcopy
import PIL
import PIL.Image
from dust3r.dust3r.datasets.utils.transforms import ImgNorm
from dust3r.dust3r_visloc.datasets.utils import get_resize_function
from tqdm import tqdm
import torch
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
from dust3r.dust3r.utils.device import to_cpu 
from mast3r.fast_nn import extract_correspondences_nonsym
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.utils.parsers import names_to_pair, parse_retrieval


import pycolmap


half = True
device = "cuda:0"
mast_model = AsymmetricMASt3R.from_pretrained("ckpts/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
if half:
    mast_model.half().to(device)
else:
    mast_model.to(device)

boq_model = get_trained_boq(backbone_name="dinov2", output_dim=12288, ckpt='ckpts/dinov2_12288.pth')
if half:
    boq_model.half().to(device)
else:
    boq_model.to(device)
    
boq_model.eval()


@dataclasses.dataclass
class Prediction:
    image_id: str | None  # A unique identifier for the row -- unused otherwise. Used only on the hidden test set.
    dataset: str
    filename: str
    cluster_index: int | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None

# Set is_train=True to run the notebook on the training data.
# Set is_train=False if submitting an entry to the competition (test data is hidden, and different from what you see on the "test" folder).
is_train = False
data_dir = 'data/image-matching-challenge-2025'
workdir = 'result/'
os.makedirs(workdir, exist_ok=True)
workdir = Path(workdir)

if is_train:
    sample_submission_csv = os.path.join(data_dir, 'train_labels.csv')
else:
    sample_submission_csv = os.path.join(data_dir, 'sample_submission.csv')

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

max_images = None  # Used For debugging only. Set to None to disable.
datasets_to_process = None  # Not the best convention, but None means all datasets.

if is_train:
    # max_images = 5

    # Note: When running on the training dataset, the notebook will hit the time limit and die. Use this filter to run on a few specific datasets.
    datasets_to_process = [
    	# New data.
    	'amy_gardens',
    	'ETs',
    	'fbk_vineyard',
    	'stairs',
    	# Data from IMC 2023 and 2024.
    	'imc2024_dioscuri_baalshamin',
    	'imc2023_theather_imc2024_church',
    	'imc2023_heritage',
    	'imc2023_haiper',
    	'imc2024_lizard_pond',
    	'pt_stpeters_stpauls',
    	'pt_brandenburg_british_buckingham',
    	'pt_piazzasanmarco_grandplace',
    	'pt_sacrecoeur_trevi_tajmahal',
    ]

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


@torch.no_grad()
def mast_inference(model, img1, img2, device, half=False):
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

    return pred1, pred2


def convert_im_matches_pairs(img0, img1, im_keypoints, matches_im0, matches_im1, viz):
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
    return ravel_matches[0], ravel_matches[1]


def get_im_matches(pred1, pred2, pairs, im_keypoints,
                   subsample=8, pixel_tol=0, viz=False, device='cuda'):
    corres = extract_correspondences_nonsym(pred1['desc'], pred2['desc'], pred1['desc_conf'], pred2['desc_conf'],
                                            device=device, subsample=subsample, pixel_tol=pixel_tol)
    conf = corres[2]
    matches_im0 = corres[0].cpu().numpy()
    matches_im1 = corres[1].cpu().numpy()
    matches = convert_im_matches_pairs(pairs[0], pairs[1], im_keypoints, matches_im0, matches_im1, viz)
    return matches, conf


def export_matches(db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification):
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
    return colmap_image_pairs


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


def run_mast_match(mast_model, image_dir, image_names,
                  device, pairs_path, conf_thr, half, pixel_tol):
    images, image_name_dict = scene_prepare_images(image_dir, 512, 16, image_names)

    im_keypoints = {idx: {} for idx in range(len(image_names))}
    pairs = []
    assert pairs_path.exists(), pairs_path
    pairs_names = parse_retrieval(pairs_path)
    pairs_names = [(q, r) for q, rs in pairs_names.items() for r in rs]
    for i, j in pairs_names:
        pairs.append((images[image_name_dict[i]], images[image_name_dict[j]]))

    for img1, img2 in tqdm(pairs,  desc='Mast inference'):
        pred1, pred2 = mast_inference(mast_model, img1, img2, device, half=half)
        matches, conf = get_im_matches(pred1=pred1, pred2=pred2, pairs=(img1, img2), im_keypoints=im_keypoints, pixel_tol=pixel_tol)
        print("111")


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

    dataset_dir = os.path.join(workdir, dataset)

    boq_topks = boq_sort_topk(images_dir, image_names, boq_model, device, vis=False, topk=32, half=half)
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, "boq_topk.json"), "w", encoding="utf-8") as f:
        json.dump(boq_topks, f, ensure_ascii=False, indent=4)


    sfm_pairs = workdir / dataset / "pairs-sfm.txt"
    loc_pairs = workdir / dataset / "pairs-loc.txt"
    sfm_dir = workdir / dataset / "sfm"
    features = workdir / dataset / "features.h5"
    matches = workdir / dataset / "matches.h5"

    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]
    
    extract_features.main(feature_conf, images_dir, image_list=image_names, feature_path=features)

    boq_make_pairs(sfm_pairs, boq_topks, image_names)

    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    ## then we use mast
    run_mast_match(mast_model, images_dir, image_names, device, sfm_pairs, conf_thr=0.1, half=half, pixel_tol=0)


    # # By default colmap does not generate a reconstruction if less than 10 images are registered.
    # # Lower it to 3.
    mapper_options = {"min_model_size" : 5, "max_num_models": 45}
    max_map, maps = reconstruction.main(
        sfm_dir, images_dir, sfm_pairs, features, matches,
        image_list=image_names, min_match_score=0.1, mapper_options = mapper_options, 
    )
    gc.collect()
    time.sleep(1)

    filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}
    registered = 0
    for map_index, cur_map in maps.items():
        for index, image in cur_map.images.items():
            prediction_index = filename_to_index[image.name]
            predictions[prediction_index].cluster_index = map_index
            predictions[prediction_index].rotation = deepcopy(image.cam_from_world.rotation.matrix())
            predictions[prediction_index].translation = deepcopy(image.cam_from_world.translation)
            registered += 1
    mapping_result_str = f'Dataset "{dataset}" -> Registered {registered} / {len(image_names)} images with {len(maps)} clusters'
    print(mapping_result_str)