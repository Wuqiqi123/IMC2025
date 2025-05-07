import sys
import os
from tqdm import tqdm
from time import time, sleep
import gc
import numpy as np
import h5py
import dataclasses
from copy import deepcopy
import pandas as pd
import shutil

import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF

import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from transformers import AutoImageProcessor, AutoModel
import pycolmap

import metric


device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')


device = K.utils.get_cuda_device_if_available(0)
print(f'{device=}')


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img


# Must Use efficientnet global descriptor to get matching shortlists.
def get_global_desc(fnames, device = torch.device('cpu')):
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    model = model.eval()
    model = model.to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames),total= len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            dino_mac = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1, p=2)
        global_descs_dinov2.append(dino_mac.detach().cpu())
    global_descs_dinov2 = torch.cat(global_descs_dinov2, dim=0)
    return global_descs_dinov2



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
is_train = True
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
    	# 'amy_gardens',
    	'ETs',
    	# 'fbk_vineyard',
    	'stairs',
    	# Data from IMC 2023 and 2024.
    	# 'imc2024_dioscuri_baalshamin',
    	# 'imc2023_theather_imc2024_church',
    	# 'imc2023_heritage',
    	# 'imc2023_haiper',
    	# 'imc2024_lizard_pond',
    	# Crowdsourced PhotoTourism data.
    	# 'pt_stpeters_stpauls',
    	# 'pt_brandenburg_british_buckingham',
    	# 'pt_piazzasanmarco_grandplace',
    	# 'pt_sacrecoeur_trevi_tajmahal',
    ]

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

    filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}
    
    # rm -rf $workdir/$dataset
    if os.path.exists(workdir / dataset):
        shutil.rmtree(workdir / dataset)

    sfm_pairs = workdir / dataset / "pairs-sfm.txt"
    loc_pairs = workdir / dataset / "pairs-loc.txt"
    sfm_dir = workdir / dataset / "sfm"
    features = workdir / dataset / "features.h5"
    matches = workdir / dataset / "matches.h5"

    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]

    extract_features.main(feature_conf, images_dir, image_list=image_names, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_names)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    
    # # By default colmap does not generate a reconstruction if less than 10 images are registered.
    # # Lower it to 3.
    mapper_options = {"min_model_size" : 3, "max_num_models": 25}
    max_map, maps = reconstruction.main(
        sfm_dir, images_dir, sfm_pairs, features, matches,
        image_list=image_names, min_match_score=0.2, mapper_options = mapper_options, 
    )
    gc.collect()
    sleep(1)

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



array_to_str = lambda array: ';'.join([f"{x:.09f}" for x in array])
none_to_str = lambda n: ';'.join(['nan'] * n)

submission_file = './submission.csv'
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

if is_train:
    t = time()
    final_score, dataset_scores = metric.score(
        gt_csv='/kaggle/input/image-matching-challenge-2025/train_labels.csv',
        user_csv=submission_file,
        thresholds_csv='/kaggle/input/image-matching-challenge-2025/train_thresholds.csv',
        mask_csv=None if is_train else os.path.join(data_dir, 'mask.csv'),
        inl_cf=0,
        strict_cf=-1,
        verbose=True,
    )
    print(f'Computed metric in: {time() - t:.02f} sec.')





