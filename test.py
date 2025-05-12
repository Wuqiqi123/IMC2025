from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import os
import numpy as np
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import torch
import glob
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever
from mast3r.utils.misc import mkdir_for
from cust3r.utils.image import load_images
from dust3r.dust3r.utils.device import to_numpy
from dust3r.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
import matplotlib.pyplot as pl
import imageio.v2 as iio
import time
from boq.boq_infer import get_trained_boq, boq_sort_topk
import json

def _convert_scene_output_to_glb(imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)
    scene = trimesh.Scene()
    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)
    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    return scene

def get_3D_model_from_scene(silent, scene, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    # get optimized values from scene
    scene = scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)
    

def get_reconstructed_scene(model, device, filelist,
                            cache_path,
                            retrieval_model = None,
                            silent = False,
                            optim_level = "refine+depth",
                            lr1 = 0.07, niter1 = 200, lr2 = 0.01, niter2 = 200,
                            min_conf_thr = 1.5,
                            matching_conf_thr = 0.0,
                            as_pointcloud = True, mask_sky = False, clean_depth =True, transparent_cams = False, cam_size = 0.2,
                            scenegraph_type = "complete", winsize=1, win_cyclic=False, refid=0,
                            TSDF_thresh=0.0, shared_intrinsics= False, half=False,
                            **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs, imgs_id_dict = load_images(filelist, size=224, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)
        # Cleanup
        del retriever
        torch.cuda.empty_cache()
    boq_topks = None
    if 'boq' in scenegraph_type:
        with open(os.path.join(cache_path, "boq_topk.json"), "r", encoding="utf-8") as f:
            boq_topks = json.load(f)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, 
                       symmetrize=False, sim_mat=sim_matrix, boq_topk_dict=boq_topks, 
                       imgs_id_dict=imgs_id_dict)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    scenes, outlier_imgs = sparse_global_alignment(filelist, imgs, imgs_id_dict, pairs, cache_path,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, half=half, **kw)
    trimesh_scenes = []
    for i, scene in enumerate(scenes):
        trimesh_scene = get_3D_model_from_scene(silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
        trimesh_scenes.append(trimesh_scene)
    return trimesh_scenes, outlier_imgs

device = 'cuda:0'
half = True
model = AsymmetricMASt3R.from_pretrained("ckpts/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
if half:
    model.half().to(device)
else:
    model.to(device)
    
image_list = []
for filename in glob.glob('data/image-matching-challenge-2025/train/imc2023_haiper/*.png'): #assuming gif
    image_list.append(filename)

boq_model = get_trained_boq(backbone_name="dinov2", output_dim=12288, ckpt='ckpts/dinov2_12288.pth')
if half:
    boq_model.half().to(device)
else:
    boq_model.to(device)
    
boq_model.eval()
boq_topks = boq_sort_topk(image_list, boq_model, device, vis=False, topk=32, half=half)

os.makedirs("outputs/imc2023_haiper", exist_ok=True)
with open(os.path.join("outputs/imc2023_haiper", "boq_topk.json"), "w", encoding="utf-8") as f:
    json.dump(boq_topks, f, ensure_ascii=False, indent=4)
    
trimesh_scenes, outlier_imgs = get_reconstructed_scene(model, device, image_list, "outputs/imc2023_haiper", scenegraph_type = "boq", half=half)
del model, boq_model
torch.cuda.empty_cache()
trimesh_scenes[0].show()