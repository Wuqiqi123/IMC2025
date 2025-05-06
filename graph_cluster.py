import networkx as nx
import numpy as np
import cv2
from hloc.utils.io import get_keypoints, get_matches
from tqdm import tqdm
import h5py
def build_image_graph(results, match_thresh=30, score_thresh=0.2, geom_verify=True, geom_thresh=20):
    """
    根据匹配结果构建图像图，并提取分簇信息与离群图像。

    Returns:
        clusters (List[Set[str]]): 图像簇列表
        outliers (List[str]): 未归入任何簇的图像
        G (networkx.Graph): 构建的图
    """
    G = nx.Graph()
    outlier_candidates = set()

    for (img_key), data in results.items():
        img0, img1 = img_key.split('/')
        matches = data["matches"]
        scores = data["scores"]
        kpts0 = data["kpts0"]
        kpts1 = data["kpts1"]

        if len(matches) < match_thresh or scores.mean() < score_thresh:
            continue

        # 几何验证
        if geom_verify:
            pts0 = kpts0[matches[:, 0]]
            pts1 = kpts1[matches[:, 1]]
            F, inliers = cv2.findFundamentalMat(pts0, pts1, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0)
            if inliers is None or inliers.sum() < geom_thresh:
                # 将无法通过几何验证的图像标记为离群图像
                outlier_candidates.add(img0)
                outlier_candidates.add(img1)
                continue

        # 满足条件，加入图
        G.add_edge(img0, img1, weight=scores.mean())

    # 提取连通分量作为簇
    clusters = list(nx.connected_components(G))
    return clusters, outlier_candidates, G




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
    

if __name__ == '__main__':
    feature_path = '/data/code/IMC2025/result/ETs/features.h5'
    matches_path = '/data/code/IMC2025/result/ETs/matches.h5'
    pairs_path = '/data/code/IMC2025/result/ETs/pairs-sfm.txt'

    results = parser_h5s(feature_path, matches_path, pairs_path, min_match_score=0.2)
    # 调用分簇函数
    clusters, outliers, G = build_image_graph(results, match_thresh=50)

    # 输出分簇结果
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx}: {sorted(cluster)}")
        
        
    print('outliers: ', outliers)
