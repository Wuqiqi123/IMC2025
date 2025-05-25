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
import os.path as osp
from shutil import rmtree
from hloc.dataset.coarse_sfm_refinement_dataset import CoarseColmapDataset
from hloc.post_opt.multiview_match import multiview_matcher
from hloc.utils.data_io import load_obj
from hloc.post_opt.write_fixed_images import fix_all_images, fix_farest_images
from hloc import match_dense
from hloc.sfm_runner import reregistration, sfm_model_geometry_refiner

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


if is_train:
    sample_submission_csv = os.path.join(data_dir, 'train_labels.csv')
else:
    sample_submission_csv = os.path.join(data_dir, 'sample_submission.csv')


boq_model = get_trained_boq(backbone_name="dinov2", output_dim=12288, ckpt='ckpts/dinov2_12288.pth')
if half:
    boq_model.half().to(device)
else:
    boq_model.to(device)
    
boq_model.eval()


def run_image_reregistration(
    deep_sfm_dir, after_refine_dir, colmap_path, image_path="/", colmap_configs=None, verbose=True
):
    print("Running the bundle adjuster.")

    deep_sfm_model_dir = osp.join(deep_sfm_dir, "model")
    database_path = osp.join(deep_sfm_dir, "database.db")
    cmd = [
        str(colmap_path),
        "image_registrator",
        "--database_path",
        str(database_path),
        "--input_path",
        str(deep_sfm_model_dir),
        "--output_path",
        str(after_refine_dir),
    ]

    if colmap_configs is not None and colmap_configs["no_refine_intrinsics"] is True:
        cmd += [
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
        ]
    
    if 'reregistration' in colmap_configs:
        # Set to lower threshold to registrate more images
        cmd += [
            "--Mapper.abs_pose_max_error",
            str(colmap_configs['reregistration']['abs_pose_max_error']),
            "--Mapper.abs_pose_min_num_inliers",
            str(colmap_configs['reregistration']['abs_pose_min_num_inliers']),
            "--Mapper.abs_pose_min_inlier_ratio",
            str(colmap_configs['reregistration']['abs_pose_min_inlier_ratio']),
            "--Mapper.filter_max_reproj_error",
            str(colmap_configs['reregistration']['filter_max_reproj_error'])
        ]

    if verbose:
        print(' '.join(cmd))
        ret = subprocess.call(cmd)
    else:
        ret_all = subprocess.run(cmd, capture_output=True)
        with open(osp.join(after_refine_dir, 'reregistration_output.txt'), 'w') as f:
            f.write(ret_all.stdout.decode())
        ret = ret_all.returncode

    if ret != 0:
        print("Problem with image registration, existing.")
        exit(ret)


def post_optimization(
    image_lists,
    covis_pairs_pth,
    colmap_coarse_dir,
    refined_model_save_dir,
    match_out_pth,
    chunk_size=6000,
    img_resize=None,
    img_preload=False,
    fine_match_use_ray=False,  # Use ray for fine match
    ray_cfg=None,
    colmap_configs=None,
    only_basename_in_colmap=False,
    visualize_dir=None,
    vis3d_pth=None,
    verbose=True
):
    """
    Iterative n times:
        Reproject current 3D model to update keypoints;
        Refine Keypoints;
        Itertative m times:
            BA(optimize 3D points and poses);
            Adjust scene structure:
                merge feature track;
                complete feature track;
                filter feature track;
        Reregistration[Optional];
    """
    

    cfgs = {
        "coarse_colmap_data": {
            "img_resize": 1200,
            "df": None,
            "feature_track_assignment_strategy": "midium_scale",
            "img_preload": False,
        },
        "fine_match_debug": True,
        "multiview_matcher_data": {
            "max_track_length": 16,
            "chunk": 6000 
        },
        "fine_matcher": {
            "model": {
                "cfg_path": ["config/multiview_refinement_matching.yaml"],
                "weight_path": "weight/multiview_matcher.ckpt",
                "seed": 666,
            },
            "visualize": False,
            "extract_feature_method": "fine_match_backbone",
            "ray": {
                "slurm": False,
                "n_workers": 1,
                "n_cpus_per_worker": 1,
                "n_gpus_per_worker": 1,
                "local_mode": False,
            },
        },
        "visualize": False,
        "evaluation": False,
        "refine_iter_n_times": 2,
        "model_refiner_no_filter_pts": False,
        "first_iter_resize_img_to_half": False,
        "enable_update_reproj_kpts_to_model": False,
        "enable_adaptive_downscale_window": True, # Down scale searching window size after each iteration, e.g., 15->11->7
        "incremental_refiner_filter_thresholds": [3, 2, 1.5],
        "incremental_refiner_use_pba": False, # NOTE: pba does not allow share intrins or fix extrinsics, and only allow simple_radial camer model
        "enable_multiple_models": False,
    }
    
    cfgs['coarse_colmap_data']['img_preload'] = img_preload
    cfgs['incremental_refiner_use_pba'] = colmap_configs["use_pba"]
    cfgs['multiview_matcher_data']['chunk'] = chunk_size

    # Link images to temp directory for later extract colors.
    temp_image_path = osp.join(osp.dirname(refined_model_save_dir), f'temp_images')
    if osp.exists(temp_image_path):
        os.system(f"rm -rf {temp_image_path}")
    os.makedirs(temp_image_path)
    for img_path in image_lists:
        os.system(f"ln -s {img_path} {osp.join(temp_image_path, osp.basename(img_path))}")

    # Clear all previous results:
    temp_refined_dirs = [dir_name for dir_name in os.listdir(osp.dirname(refined_model_save_dir)) if 'model_refined' in dir_name or osp.basename(refined_model_save_dir) == dir_name]
    for temp_result_name in temp_refined_dirs:
        rmtree(osp.join(osp.dirname(refined_model_save_dir),  temp_result_name))

    iter_n_times = cfgs['refine_iter_n_times']
    iter_id = tqdm(range(iter_n_times)) if verbose else range(iter_n_times)

    for i in iter_id:
        if cfgs['first_iter_resize_img_to_half'] and i == 0:
            cfgs["coarse_colmap_data"]['img_resize'] = img_resize // 2
        else:
            cfgs["coarse_colmap_data"]['img_resize'] = img_resize

        # Construct scene data
        colmap_image_dataset = CoarseColmapDataset(
            cfgs["coarse_colmap_data"],
            image_lists,
            covis_pairs_pth,
            colmap_coarse_dir if i == 0 else last_model_dir,
            refined_model_save_dir,
            only_basename_in_colmap=only_basename_in_colmap,
            vis_path=vis3d_pth if vis3d_pth is not None else None,
            verbose=verbose
        )
        print("Scene data construct finish!") if verbose else None

        if cfgs['enable_update_reproj_kpts_to_model']:
            if i != 0:
                # Leverage current model to update keypoints
                colmap_image_dataset.update_kpts_by_current_model_projection(fix_ref_node=True)

        state = colmap_image_dataset.state
        if state == False:
            print(
                f"Build colmap coarse dataset fail! colmap point3D or images or cameras is empty!"
            )
            return state, None, None

        # Fine level match
        save_path = osp.join(match_out_pth.rsplit("/", 2)[0], "fine_matches.pkl")
        if not osp.exists(save_path) or cfgs["fine_match_debug"]:
            print(f"Multi-view refinement matching begin!")
            model_idx = 0 if i == 0 else 1
            rewindow_size_factor = i * 2
            fine_match_results = multiview_matcher(
                cfgs["fine_matcher"],
                cfgs["multiview_matcher_data"],
                colmap_image_dataset,
                rewindow_size_factor=rewindow_size_factor if cfgs["enable_adaptive_downscale_window"] else None,
                model_idx= None,
                visualize_dir=visualize_dir,
                use_ray=fine_match_use_ray,
                ray_cfg=ray_cfg,
                verbose=verbose
            )
        else:
            print(f"Fine matches exists! Load from {save_path}")
            fine_match_results = load_obj(save_path)
        
        if i != iter_n_times -1:
            current_model_dir = osp.join(osp.dirname(refined_model_save_dir), f'model_refined_{i}')
        else:
            current_model_dir = refined_model_save_dir

        last_model_dir = current_model_dir

        colmap_refined_kpts_dir = osp.join(osp.dirname(refined_model_save_dir), 'temp_refined_kpts')
        Path(colmap_refined_kpts_dir).mkdir(parents=True, exist_ok=True)
        colmap_image_dataset.update_refined_kpts_to_colmap_multiview(fine_match_results)

        if i == 0:
            if osp.exists(osp.join(colmap_refined_kpts_dir, 'database.db')):
                os.system(f"rm -rf {osp.join(colmap_refined_kpts_dir, 'database.db')}")
            os.system(f"cp {osp.join(osp.dirname(colmap_coarse_dir), 'database.db')} {osp.join(colmap_refined_kpts_dir, 'database.db')}")
            fix_farest_images(reconstructed_model_dir=colmap_coarse_dir, output_path=osp.join(colmap_refined_kpts_dir, 'fixed_images.txt'))

        colmap_image_dataset.save_colmap_model(osp.join(colmap_refined_kpts_dir, 'model'))

        # Refinement:
        filter_threshold = cfgs['incremental_refiner_filter_thresholds'][i] if i < len(cfgs['incremental_refiner_filter_thresholds'])-1 else cfgs['incremental_refiner_filter_thresholds'][-1]
        success = sfm_model_geometry_refiner.main(colmap_refined_kpts_dir, current_model_dir, no_filter_pts=cfgs["model_refiner_no_filter_pts"], colmap_configs=colmap_configs, image_path=temp_image_path, verbose=verbose, filter_threshold=filter_threshold)

        if not success:
            # Refine failed scenario, use the coarse model instead.
            os.system(f"cp {osp.join(colmap_refined_kpts_dir, 'model') + '/*'} {current_model_dir}")

        os.system(f"rm -rf {osp.join(colmap_refined_kpts_dir, 'model')}")
        os.makedirs(osp.join(colmap_refined_kpts_dir, 'model'), exist_ok=True)
        os.system(f"cp {current_model_dir+'/*'} {osp.join(colmap_refined_kpts_dir, 'model')}")

        # Re-registration:
        if i % 2 == 0:
            reregistration.main(colmap_refined_kpts_dir, current_model_dir, colmap_configs=colmap_configs, verbose=verbose)

    return state


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
    	'ETs',
    	# 'fbk_vineyard',
    	# 'stairs',
    	# Data from IMC 2023 and 2024.
    	# 'imc2024_dioscuri_baalshamin',
    	# 'imc2023_theather_imc2024_church',
    	# 'imc2023_heritage',
    	# 'imc2023_haiper',
    	# 'imc2024_lizard_pond',
    	# 'pt_stpeters_stpauls',
    	# 'pt_brandenburg_british_buckingham',
    	# 'pt_piazzasanmarco_grandplace',
    	# 'pt_sacrecoeur_trevi_tajmahal',
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


def make_complete_paris(sfm_pairs_path, image_list):
    pairs_name = []
    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            pairs_name.append((image_list[i], image_list[j]))
    print(f"Found {len(pairs_name)} pairs.")
    with open(sfm_pairs_path, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs_name))



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

    boq_topks = boq_sort_topk(images_dir, image_names, boq_model, device, vis=False, topk=40, half=half)
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
    clusters_dict = mast_find_cluster(mast_cache_path, mast_model, images, image_name_dict,
                                       device, sfm_pairs, subsample=8, conf_thr=0.7, half=half, pixel_tol=0)

    
    shutil.rmtree(mast_cache_path)
    

    filename_to_prdictions_index = {p.filename: idx for idx, p in enumerate(predictions)}

    registered = 0
    prediction_cluster_index = 0
    for cluster_id, image_cluster_dict in clusters_dict.items():
        print(f'cluster {cluster_id}:')
        for img_name in image_cluster_dict["names"]:
            print(f'-- {img_name}')
        
        if len(image_cluster_dict["filelist"]) < 4:
            print(f'-- outlier clusters {image_cluster_dict["filelist"]}')
            continue

        image_names_cluster = image_cluster_dict["filelist"]

        cluster_dir_name = f"cluster_{prediction_cluster_index}"
        os.makedirs(workdir / dataset / cluster_dir_name, exist_ok=True)
        clus_sfm_pairs = workdir / dataset / cluster_dir_name / "clu-pairs-sfm.txt"
        make_complete_paris(clus_sfm_pairs, image_names_cluster)

        clus_sfm_dir = workdir / dataset / cluster_dir_name / "sfm"
        os.makedirs(clus_sfm_dir, exist_ok=True)


        print(f"starting constructing {dataset} with images {image_names_cluster}")

        # lightglue
        clus_features = workdir / dataset / cluster_dir_name / "features.h5"
        clus_matches = workdir / dataset / cluster_dir_name / "matches.h5"
        extract_features.main(feature_conf, images_dir, image_list=image_names_cluster, feature_path=clus_features)
        match_features.main(matcher_conf, clus_sfm_pairs, features=clus_features, matches=clus_matches)
        mapper_options = {"min_model_size": 3, "max_num_models": 100}
        colmap_db_path, matches_size = hloc_reconstruction(clus_sfm_dir, images_dir, clus_sfm_pairs, clus_features, clus_matches,
                                            image_list=image_names_cluster, min_match_score=0.1,
                                            mapper_options = mapper_options)

        if matches_size < 5:
            print(f'-- too few matches {matches_size}')
            continue

        ## mast
        # images, image_name_dict = scene_prepare_images(images_dir, image_size, patch_size, image_names_cluster)
        # colmap_db_path = sfm_dir / "colmap.db"
        # create_empty_db(colmap_db_path)
        # image_to_colmap = import_images_and_cameras(images_dir, colmap_db_path, pycolmap.CameraMode.AUTO,
        #                                              image_list=image_names_cluster, image_path_to_idx=image_name_dict)
        # colmap_image_pairs = run_mast_match_cluster(mast_cache_path, mast_model, images, image_names_cluster,
        #                                              image_name_dict, image_to_colmap, colmap_db_path,
        #                                              device, sfm_pairs, conf_thr=1.001, half=half, pixel_tol=0,
        #                                              min_len_track=2, skip_geometric_verification=False)
        # if len(colmap_image_pairs) == 0:
        #     continue

        time.sleep(4)

        print("verifying matches")
        reconstruction_path = clus_sfm_dir / "reconstruction"
        # pycolmap.verify_matches(colmap_db_path, clus_sfm_pairs)
        glomap_run_mapper(colmap_db_path, reconstruction_path, images_dir)
        max_map = pycolmap.Reconstruction(reconstruction_path / "0")
        print(max_map.summary())

        for index, image in max_map.images.items():
            prediction_index = filename_to_prdictions_index[image.name]
            predictions[prediction_index].cluster_index = prediction_cluster_index
            predictions[prediction_index].rotation = deepcopy(image.cam_from_world.rotation.matrix())
            predictions[prediction_index].translation = deepcopy(image.cam_from_world.translation)
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
        predictions = sorted(predictions, key=lambda t: t.cluster_index if t.cluster_index != None else 100)
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = 'outliers' if prediction.cluster_index is None else f'cluster{prediction.cluster_index}'
                rotation = none_to_str(9) if prediction.rotation is None else array_to_str(prediction.rotation.flatten())
                translation = none_to_str(3) if prediction.translation is None else array_to_str(prediction.translation)
                f.write(f'{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n')
    else:
        f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')
        predictions = sorted(predictions, key=lambda t: t.cluster_index if t.cluster_index != None else 100)
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