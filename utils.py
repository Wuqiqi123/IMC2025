import numpy as np
import cv2
from hloc.utils.io import get_keypoints, get_matches
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import heapq
from math import log
from PIL import Image
import torch
import torchvision.transforms as T

def parser_h5s(feature_path, matches_path, pairs_path, min_match_score=0.2):
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
        
    # matches = get_matches(matches_path)
    # feats, confs = get_keypoints(feature_path, True)
    feats = h5py.File(feature_path, 'r')
    res = {}
    for name0, name1 in tqdm(pairs):
        kpts0 = feats[name0]['keypoints'][:]
        kpts1 = feats[name1]['keypoints'][:]
        matches, scores = get_matches(matches_path, name0, name1)
        key = name0 + '/' + name1
        if key not in res.keys():
            res[key] = {
                        'matches': matches,
                        'scores': scores,
                        'kpts0': kpts0,
                        'kpts1': kpts1
                    }
    return res





def compute_robust_score(scores):
    if len(scores) == 0:
        return 0
    return float(np.mean(scores) * log(len(scores) + 1))

def get_topk_candidates(res, img_root, save_dir, k=5, min_matches=15, vis=False):
    os.makedirs(save_dir, exist_ok=True)

    scores_dict = defaultdict(list)

    for key, data in res.items():
        imgA, imgB = key.split('/')
        scores = np.array(data['scores'])  # shape: (N,)

        if len(scores) < min_matches:
            continue

        score = compute_robust_score(scores)

        scores_dict[imgA].append((score, imgB))
        scores_dict[imgB].append((score, imgA))

    candidates = {}
    for img, score_list in scores_dict.items():
        topk = heapq.nlargest(k, score_list, key=lambda x: x[0])
        candidates[img] = topk

    # --- 可视化并保存 ---
    if vis:
        for img_name, topk_list in candidates.items():
            try:
                img_paths = [os.path.join(img_root, f"{img_name}")]
                img_paths += [os.path.join(img_root, f"{name}") for _, name in topk_list]

                images = [cv2.imread(p) for p in img_paths if os.path.exists(p)]
                images = [cv2.resize(img, (400, 300)) for img in images]

                concat = cv2.hconcat(images)
                save_path = os.path.join(save_dir, f"{img_name}_top{k}.jpg")
                cv2.imwrite(save_path, concat)
            except Exception as e:
                print(f"Error with {img_name}: {e}")

    return candidates
        
    

if __name__ == '__main__':
    feature_path = 'result/ETs/features.h5'
    matches_path = 'result/ETs/matches.h5'
    pairs_path = 'result/ETs/pairs-sfm.txt'
    img_root = '/workspace/work/local/IMC2025/data/image-matching-challenge-2025/train/ETs'
    save_dir = 'topk_shows'

    
    # feature_path = 'result/stairs/features.h5'
    # matches_path = 'result/stairs/matches.h5'
    # pairs_path = 'result/stairs/pairs-sfm.txt'
    
    
    param_grid = []

    res = parser_h5s(feature_path, matches_path, pairs_path, min_match_score=0.2)
    
    topks = get_topk_candidates(res, img_root, save_dir, k=20, min_matches=15, vis=True)
    
    for q, keys in topks.items():
        print(q, keys)
    
    
    
    # boq_topk()
    
    