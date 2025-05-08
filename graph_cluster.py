import os
import cv2
import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from hloc.utils.io import get_keypoints, get_matches




def build_image_graph(results, match_thresh=30, score_thresh=0.2,
                      geom_verify=True, geom_thresh=20,
                      visualize=True,
                      param_grid=None,
                      enable_hierarchical=False,
                      use_dbscan=False,
                      use_scipy_hierarchical=False):
    """
    图像图构建 + 多种分簇方法（支持默认、DBSCAN、Scipy层次聚类）
    """
    G = nx.Graph()
    for img_key, data in results.items():
        img0, img1 = img_key.split('/')
        scores_mean = data["scores"].mean()

        pts0 = data["kpts0"][data["matches"][:, 0]]
        pts1 = data["kpts1"][data["matches"][:, 1]]
        F, inliers = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 1.0)
        valid = inliers is not None and inliers.sum() >= geom_thresh
        if valid:
            weight = scores_mean
            G.add_edge(img0, img1, weight=weight)
        else:
            if img0 not in G:
                G.add_node(img0)
            if img1 not in G:
                G.add_node(img1)
            if scores_mean > 0.5:
                G.add_edge(img0, img1, weight=scores_mean)

    if visualize:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        edge_collection = nx.draw_networkx_edges(
            G, pos,
            edge_color=[G[u][v]['weight'] for u, v in G.edges()],
            width=2, edge_cmap=plt.cm.Blues
        )
        nx.draw_networkx_nodes(G, pos, node_color='lightblue')
        nx.draw_networkx_labels(G, pos)
        plt.colorbar(edge_collection, label='Edge Weight')
        plt.title("Graph Structure (Edge Darkness = Weight)")
        plt.savefig("graph_output.png")
        plt.close()

    if use_dbscan:
        clusters, outliers = cluster_with_dbscan(G)

    elif use_scipy_hierarchical:
        clusters, outliers = cluster_with_scipy_hierarchical(G)

    else:
        default_params = {'weight_thresh': 0.5, 'degree_thresh': 4}
        # param_grid = generate_param_grid(G)
        if param_grid:
            best_params, best_score = grid_search_outlier_params(G, param_grid)
            print(f"Best params: {best_params}, Score: {best_score:.2f}")
        else:
            best_params = default_params
        outliers = find_outliers(G, **best_params)
        clusters = [c for c in nx.connected_components(G) if len(c) > 1]

        if enable_hierarchical:
            small_clusters = [c for c in clusters if len(c) < 3]
            outliers = refine_with_hierarchical(G, clusters, small_clusters, outliers)

    return clusters, outliers, G


def cluster_with_scipy_hierarchical(G, method='ward', t=1.0, criterion='distance'):
    """
    Scipy 层次聚类：每个节点用 (degree, mean_weight) 表征
    """
    nodes = list(G.nodes)
    features = []

    for node in nodes:
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        mean_weight = np.mean([G[node][nbr]['weight'] for nbr in neighbors]) if neighbors else 0
        features.append([degree, mean_weight])

    features = np.array(features)
    Z = linkage(features, method=method)
    labels = fcluster(Z, t=t, criterion=criterion)

    clusters = {}
    outliers = []
    for node, label in zip(nodes, labels):
        clusters.setdefault(label, set()).add(node)

    outliers = [list(c)[0] for c in clusters.values() if len(c) == 1]
    clusters = [c for c in clusters.values() if len(c) > 1]
    return clusters, outliers


def cluster_with_dbscan(G, eps=0.3, min_samples=2):
    nodes = list(G.nodes)
    node_features = []

    for node in nodes:
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        mean_weight = np.mean([G[node][nbr]['weight'] for nbr in neighbors]) if neighbors else 0
        node_features.append([degree, mean_weight])

    node_features = np.array(node_features)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(node_features)

    clusters = {}
    outliers = []
    for node, label in zip(nodes, labels):
        if label == -1:
            outliers.append(node)
        else:
            clusters.setdefault(label, set()).add(node)
    return list(clusters.values()), outliers


def find_outliers(G, weight_thresh=0.5, degree_thresh=4):
    outliers = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            outliers.append(node)
            continue
        total_weight = sum(G[node][nbr]['weight'] for nbr in neighbors) / len(neighbors)
        degree = G.degree(node)
        if total_weight < weight_thresh and degree <= degree_thresh:
            outliers.append(node)
    return outliers


def generate_param_grid(G):
    density = G.number_of_edges() / max(1, G.number_of_nodes())
    base = 0.5 if density > 0.1 else 0.2
    return {
        'weight_thresh': [base * 0.5, base, base * 1.5],
        'degree_thresh': [1, 2]
    }


def grid_search_outlier_params(G, param_grid):
    best_score = -np.inf
    best_params = {}

    def evaluate_params(G, weight_thresh, degree_thresh):
        outliers = find_outliers(G, weight_thresh, degree_thresh)
        valid_nodes = set(G.nodes()) - set(outliers)
        subgraph = G.subgraph(valid_nodes)
        if not valid_nodes:
            return -np.inf
        avg_weight = np.mean([d['weight'] for u, v, d in subgraph.edges(data=True)]) if subgraph.edges() else 0
        outlier_ratio = len(outliers) / len(G.nodes())
        return avg_weight - 0.5 * outlier_ratio

    for weight_thresh, degree_thresh in product(param_grid['weight_thresh'], param_grid['degree_thresh']):
        score = evaluate_params(G, weight_thresh, degree_thresh)
        if score > best_score:
            best_score = score
            best_params = {'weight_thresh': weight_thresh, 'degree_thresh': degree_thresh}
    return best_params, best_score


def refine_with_hierarchical(G, clusters, small_clusters, outliers):
    new_outliers = list(outliers)
    for cluster in small_clusters:
        features = [sum(G[node][nbr]['weight'] for nbr in G.neighbors(node)) for node in cluster]
        features = np.array(features).reshape(-1, 1)
        clustering = AgglomerativeClustering(n_clusters=2)
        labels = clustering.fit_predict(features)
        for node, label in zip(cluster, labels):
            if label == 1:
                new_outliers.append(node)
    return list(set(new_outliers))

def visualize_matches(img_dir, name0, name1, kpts0, kpts1, matches, scores, save_dir='vis_matches'):
    os.makedirs(save_dir, exist_ok=True)

    # 读取图像
    img0 = cv2.imread(os.path.join(img_dir, name0))
    img1 = cv2.imread(os.path.join(img_dir, name1))
    if img0 is None or img1 is None:
        print(f"[WARN] Cannot load images: {name0}, {name1}")
        return

    # 创建 OpenCV 格式的 DMatch 对象
    good_matches = []
    for i, (m, score) in enumerate(zip(matches, scores)):
        if m[0] < 0 or m[1] < 0:
            continue
        pt0 = tuple(map(int, kpts0[m[0]]))
        pt1 = tuple(map(int, kpts1[m[1]]))
        good_matches.append(cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _distance=float(1.0 - score)))

    # 转为关键点对象
    kp0_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts0]
    kp1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts1]

    # 绘制匹配图
    match_img = cv2.drawMatches(img0, kp0_cv, img1, kp1_cv, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    save_name = f"{os.path.splitext(os.path.basename(name0))[0]}_{os.path.splitext(os.path.basename(name1))[0]}.jpg"
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, match_img)
    
def parser_h5s(feature_path, matches_path, pairs_path, img_dir, min_match_score=0.2, vis=True, point_vis_dir='vis_matches'):
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    feats = h5py.File(feature_path, 'r')
    res = {}
    for name0, name1 in tqdm(pairs):
        kpts0 = feats[name0]['keypoints'][:]
        kpts1 = feats[name1]['keypoints'][:]
        matches, scores = get_matches(matches_path, name0, name1)
        key = name0 + '/' + name1
        if key not in res:
            res[key] = {
                'matches': matches,
                'scores': scores,
                'kpts0': kpts0,
                'kpts1': kpts1
            }
        if vis:
            visualize_matches(img_dir, name0, name1, kpts0, kpts1, matches, scores, point_vis_dir)
    return res



# --- 主程序 ---
if __name__ == '__main__':
    # base_dir = '/data/code/IMC2025/result/ETs/'  # 替换为你的路径
    # img_dir = '/data/code/IMC2025/data/image-matching-challenge-2025/train/ETs'


    base_dir = '/data/code/IMC2025/result/stairs/'  # 替换为你的路径
    img_dir = '/data/code/IMC2025/data/image-matching-challenge-2025/train/stairs'
    point_vis_dir = 'vis_matches_stairs'
    feature_path = os.path.join(base_dir, 'features.h5')
    matches_path = os.path.join(base_dir, 'matches.h5')
    pairs_path = os.path.join(base_dir, 'pairs-sfm.txt')

    results = parser_h5s(feature_path, matches_path, pairs_path, img_dir, min_match_score=0.2, point_vis_dir=point_vis_dir)

    clusters, outliers, G = build_image_graph(
        results,
        match_thresh=50,
        use_dbscan=False,
        use_scipy_hierarchical=False,   # ✅ 启用 Scipy 层次聚类
        visualize=True
    )

    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx}: {sorted(cluster)}")
    print('Outliers:', outliers)
