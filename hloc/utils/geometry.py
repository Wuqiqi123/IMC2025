import numpy as np
import pycolmap
from hloc.utils.read_write_model import qvec2rotmat


def to_homogeneous(p):
    return np.pad(p, ((0, 0),) * (p.ndim - 1) + ((0, 1),), constant_values=1)


def compute_epipolar_errors(j_from_i: pycolmap.Rigid3d, p2d_i, p2d_j):
    j_E_i = j_from_i.essential_matrix()
    l2d_j = to_homogeneous(p2d_i) @ j_E_i.T
    l2d_i = to_homogeneous(p2d_j) @ j_E_i
    dist = np.abs(np.sum(to_homogeneous(p2d_i) * l2d_i, axis=1))
    errors_i = dist / np.linalg.norm(l2d_i[:, :2], axis=1)
    errors_j = dist / np.linalg.norm(l2d_j[:, :2], axis=1)
    return errors_i, errors_j


def convert_pose2T(pose):
    # pose: [R: 3*3, t: 3]
    R, t = pose
    return np.concatenate(
        [np.concatenate([R, t[:, None]], axis=1), [[0, 0, 0, 1]]], axis=0
    )  # 4*4


def convert_T2pose(T):
    # T: 4*4
    return [T[:3, :3], T[:3, 3]]


def get_pose_from_colmap_image(image):
    # return: [R: numpy.array 3*3, t: numpy.array 3]
    qvec = image.qvec
    R = qvec2rotmat(qvec)  # 3*3
    t = image.tvec  # 3
    return [R, t]


def get_intrinsic_from_colmap_camera(camera):
    model = camera.model
    if model == 'SIMPLE_RADIAL':
        focal = camera.params[0]
        x0 = camera.params[1]
        y0 = camera.params[2]
        intrinsic = np.array([[focal, 0, x0], [0, focal, y0], [0, 0, 1]])
    elif model == 'PINHOLE':
        focal0 = camera.params[0]
        focal1 = camera.params[1]
        x0 = camera.params[2]
        y0 = camera.params[3]
        intrinsic = np.array([[focal0, 0, x0], [0, focal1, y0], [0, 0, 1]])
    else:
        raise NotImplementedError

    return intrinsic


def project_point_cloud_to_image(intrinsic, pose, point_cloud):
    """
    Parameters:
    -------------
    intrinisc: np.array 3*3 or N*3*3
    pose: [R: np.array 3*3, t: np.array 3] or N*4*4
    point_cloud: np.array K*3 or N * K * 3

    Return:
    ----------
    keypoints: K*2 or N * K * 2
    depth: K or N * K
    """
    if len(intrinsic.shape) == 2:
        R, t = pose
        point_cloud_f = (
            R @ point_cloud.T + t[:, None]
        )  # 3*N point cloud in camera coordiante
        point_cloud_rpj = (intrinsic @ point_cloud_f).T  # N*3

        keypoints = point_cloud_rpj[:, :2] / (point_cloud_rpj[:, [2]] + 1e-4)
        depth = point_cloud_rpj[:, 2]
    elif len(intrinsic.shape) == 3:
        assert len(pose.shape) == 3 and len(point_cloud.shape) == 3
        R = pose[:, :3, :3]  # N * 3 * 3
        t = pose[:, :3, [3]] # N * 3 * 1

        point_cloud_f = (
            R @ point_cloud.transpose(0,2,1) + t
        )  # N * 3 * K point cloud in camera coordiante
        point_cloud_rpj = (intrinsic @ point_cloud_f).transpose(0,2,1)  # N * K * 3

        keypoints = point_cloud_rpj[..., :2] / (point_cloud_rpj[..., [2]] + 1e-4)
        depth = point_cloud_rpj[..., 2]

    return keypoints, depth

def transform_point_cloud_to_camera(pose, point_cloud):
    """
    Parameters:
    -------------
    pose: [R: np.array 3*3, t: np.array 3] or np.array 4*4
    point_cloud: np.array N*3

    Return:
    ----------
    pts_f: N*3
    """
    if isinstance(pose, list):
        R, t = pose
    elif isinstance(pose, np.ndarray):
        R, t = convert_T2pose(pose)
    point_cloud_f = (
        R @ point_cloud.T + t[:, None]
    ).T  # N*3 point cloud in camera coordiante

    return point_cloud_f # N * 3