import networkx as nx
import numpy as np
import cv2
from hloc.utils.io import get_keypoints, get_matches
from tqdm import tqdm
import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from itertools import product

def build_image_graph(results, match_thresh=30, score_thresh=0.2, 
                    geom_verify=True, geom_thresh=20,
                    visualize=True, 
                    param_grid=None, 
                    enable_hierarchical=False):
    """
    改进版图像图构建与离群点检测
    
    Args:
        results: 图像匹配结果字典
        visualize: 是否可视化图结构
        param_grid: 网格搜索参数 (e.g., {'weight_thresh': [0.3, 0.5], 'degree_thresh': [1, 2]})
        enable_hierarchical: 是否对小簇启用层次聚类
    """
    # 第一阶段：构建带权图
    G = nx.Graph()
    for img_key, data in results.items():
        img0, img1 = img_key.split('/')
        scores_mean = data["scores"].mean()
        
        # if len(data["matches"]) < match_thresh or scores_mean < score_thresh:
        #     continue
            
        pts0 = data["kpts0"][data["matches"][:, 0]]
        pts1 = data["kpts1"][data["matches"][:, 1]]
        F, inliers = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 1.0)
        valid = inliers is not None and inliers.sum() >= geom_thresh
        if valid:
            weight = scores_mean
            G.add_edge(img0, img1, weight=weight)
        else:
            weight = scores_mean 
            print('outliers: ', scores_mean)
            G.add_edge(img0, img0, weight=0)
            G.add_edge(img1, img1, weight=0)
            if weight > 0.5:
                G.add_edge(img0, img1, weight=weight)
            
        

    # 可视化调试
    if visualize:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # 修改点1：捕获draw返回的边集合
        edge_collection = nx.draw_networkx_edges(
            G, pos, 
            edge_color=[G[u][v]['weight'] for u,v in G.edges()],
            width=2, edge_cmap=plt.cm.Blues
        )
        nx.draw_networkx_nodes(G, pos, node_color='lightblue')
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Graph Structure (Edge Darkness = Weight)")
        
        # 修改点2：使用edge_collection创建colorbar
        plt.colorbar(edge_collection, label='Edge Weight')
        plt.savefig("output.png")
        plt.close()  # 防止内存泄漏

    # 参数调优网格搜索
    default_params = {'weight_thresh': 0.5, 'degree_thresh': 4}
    
    
    def generate_param_graph(graph_density):
        base = 0.5 if graph_density > 0.1 else 0.2
        return {
            'weight_thresh': [base * 0.5, base, base * 1.5],
            'degree_thresh': [1, 2]
        }

    # 根据图密度生成网格
    param_grid = generate_param_graph(G.number_of_edges() / G.number_of_nodes())
    
    
    if param_grid:
        best_params, best_score = grid_search_outlier_params(G, param_grid)
        print(f"Best params: {best_params}, Score: {best_score:.2f}")
    else:
        best_params = default_params

    # best_params['degree_thresh'] = min([len(c) for c in clusters])//2
    # 第二阶段：离群点检测
    outliers = find_outliers(G, **best_params)
    
    # 第三阶段：迭代优化（处理小簇）
    clusters = [c for c in nx.connected_components(G) if len(c) > 1]
    

    if enable_hierarchical:
        small_clusters = [c for c in clusters if len(c) < 3]  # 定义小簇阈值
        outliers = refine_with_hierarchical(clusters, small_clusters, outliers)

    return clusters, outliers, G

# --- 辅助函数 ---
def find_outliers(G, weight_thresh=0.5, degree_thresh=4):
    """基于连接权重和度数检测离群点"""
    outliers = []
    for node in G.nodes():
        total_weight = sum(G[node][nbr]['weight'] for nbr in G.neighbors(node))/len(list(G.neighbors(node)))
        degree = G.degree(node)
        if total_weight < weight_thresh and degree <= degree_thresh:
            outliers.append(node)
    return outliers

def grid_search_outlier_params(G, param_grid):
    """网格搜索最优离群点阈值参数"""
    best_score = -np.inf
    best_params = {}
    
    # 定义评分函数（示例：最大化簇内平均边权与离群点比例的平衡）
    def evaluate_params(G, weight_thresh, degree_thresh):
        outliers = find_outliers(G, weight_thresh, degree_thresh)
        valid_nodes = set(G.nodes()) - set(outliers)
        subgraph = G.subgraph(valid_nodes)
        
        if len(valid_nodes) == 0:
            return -np.inf
            
        # 评分指标 = 簇内平均边权 + 离群点比例惩罚
        avg_weight = np.mean([d['weight'] for u,v,d in subgraph.edges(data=True)]) if subgraph.edges() else 0
        outlier_ratio = len(outliers) / len(G.nodes())
        return avg_weight - 0.5 * outlier_ratio  # 可调整权重

    # 遍历所有参数组合
    for weight_thresh, degree_thresh in product(*param_grid.values()):
        score = evaluate_params(G, weight_thresh, degree_thresh)
        if score > best_score:
            best_score = score
            best_params = {'weight_thresh': weight_thresh, 'degree_thresh': degree_thresh}
    
    return best_params, best_score

def refine_with_hierarchical(clusters, small_clusters, outliers):
    """用层次聚类处理小簇"""
    new_outliers = list(outliers)
    for cluster in small_clusters:
        # 提取小簇节点的特征（例如平均边权）
        features = [sum(G[node][nbr]['weight'] for nbr in G.neighbors(node)) for node in cluster]
        features = np.array(features).reshape(-1, 1)
        
        # 层次聚类（分为2类：保留或标记为离群点）
        clustering = AgglomerativeClustering(n_clusters=2)
        labels = clustering.fit_predict(features)
        
        # 将标签为1的节点加入离群点
        for node, label in zip(cluster, labels):
            if label == 1:
                new_outliers.append(node)
    return list(set(new_outliers))  # 去重


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
    
    
    feature_path = '/data/code/IMC2025/result/stairs/features.h5'
    matches_path = '/data/code/IMC2025/result/stairs/matches.h5'
    pairs_path = '/data/code/IMC2025/result/stairs/pairs-sfm.txt'
    
    
    param_grid = []

    results = parser_h5s(feature_path, matches_path, pairs_path, min_match_score=0.2)
    # 调用分簇函数
    clusters, outliers, G = build_image_graph(results, match_thresh=50)

    # 输出分簇结果
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx}: {sorted(cluster)}")
        
    
        
    print('outliers: ', outliers)
