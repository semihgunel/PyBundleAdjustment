from itertools import product
from typing import *

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from pyba.Camera import Camera
from pyba.CameraNetwork import CameraNetwork


def bundle_adjust(
    camNet: CameraNetwork,
    max_num_images: int = 1e3,
    update_intrinsic: bool = True,
    update_distort: bool = True,
):
    """ 
    max_num_images: large number of points take too long to optimize. will select random points instead.
    """
    (
        x0,
        points_2d,
        n_cameras,
        n_points,
        camera_indices,
        point_indices,
    ) = prepare_bundle_adjust_param(camNet=camNet, max_num_images=max_num_images)

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(
        residuals,
        x0,
        jac_sparsity=A,
        verbose=2,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        args=(
            camNet.cam_list,
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
            points_2d,
            update_intrinsic,
            update_distort,
        ),
        max_nfev=100,
    )
    camNet.triangulate()
    return res


def bundle_adjustment_sparsity(
    n_cameras: int, n_points: int, camera_indices: np.ndarray, point_indices: np.ndarray
):
    assert camera_indices.shape[0] == point_indices.shape[0]
    n_camera_params = 13
    m = camera_indices.size * 2
    # all the parameters, 13 camera parameters and x,y,z values for n_points
    n = n_cameras * n_camera_params + n_points * 3
    A = lil_matrix((m, n), dtype=int)  # sparse matrix
    i = np.arange(camera_indices.size)

    for s in range(n_camera_params):
        # assign camera parameters to points residuals (reprojection error)
        A[2 * i, camera_indices * n_camera_params + s] = 1
        A[2 * i + 1, camera_indices * n_camera_params + s] = 1

    for s in range(3):
        # assign 3d points to residuals (reprojection error)
        A[2 * i, n_cameras * n_camera_params + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * n_camera_params + point_indices * 3 + s] = 1

    return A


def prepare_bundle_adjust_param(camNet: CameraNetwork, max_num_images: int = 500):
    # prepare intrinsic
    camera_params = np.zeros(shape=(len(camNet.cam_list), 13), dtype=float)
    for cid in range(len(camNet.cam_list)):
        camera_params[cid, 0:3] = np.squeeze(camNet[cid].rvec)
        camera_params[cid, 3:6] = np.squeeze(camNet[cid].tvec)
        camera_params[cid, 6] = camNet[cid].fx
        camera_params[cid, 7] = camNet[cid].fy
        camera_params[cid, 8:13] = np.squeeze(camNet[cid].distort)

    # select which images to calculate residuals on
    img_id_list = np.arange(camNet.get_nimages())
    if camNet.get_nimages() > max_num_images:
        img_id_list = np.random.randint(
            0, high=camNet.get_nimages() - 1, size=(int(max_num_images))
        )

    point_indices, camera_indices, pts2d, pts3d = list(), list(), list(), list()
    # for all image and joint
    for img_id, jid in product(img_id_list, range(camNet.get_njoints())):
        pts3d.append(camNet.points3d[img_id, jid])

        for cid, cam in enumerate(camNet):
            if not cam.can_see(img_id, jid):
                continue

            pts2d.append(cam[img_id, jid])
            point_indices.append(len(pts3d) - 1)
            camera_indices.append(cid)

    pts3d = np.stack(pts3d)
    pts2d = np.stack(pts2d)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)

    n_cameras = camera_params.shape[0]
    n_points = pts3d.shape[0]

    x0 = np.hstack((camera_params.ravel(), pts3d.ravel()))

    return (
        x0.copy(),
        pts2d.copy(),
        n_cameras,
        n_points,
        camera_indices,
        point_indices,
    )


def update_parameters(
    camera_params: np.ndarray,
    cam: Camera,
    update_intrinsic: bool = True,
    update_distort: bool = True,
):
    cam.rvec = camera_params[0:3]
    cam.tvec = camera_params[3:6]
    if update_intrinsic:
        cam.fx = camera_params[6]
        cam.fy = camera_params[7]
    if update_distort:
        cam.distort = camera_params[8:13]


def residuals(
    params,
    cam_list: List[Camera],
    n_cameras: int,
    n_points: int,
    camera_indices: List[int],
    point_indices: List[int],
    points_2d: np.ndarray,
    update_intrinsic: bool = True,
    update_distort: bool = True,
):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    assert point_indices.shape[0] == points_2d.shape[0]
    assert camera_indices.shape[0] == points_2d.shape[0]

    camera_params = params[: n_cameras * 13].reshape((n_cameras, 13))
    points3d = params[n_cameras * 13 :].reshape((n_points, 3))
    cam_indices_list = list(set(camera_indices))

    points_proj = np.zeros(shape=(point_indices.shape[0], 2), dtype=np.float)
    for cam_id in cam_indices_list:
        update_parameters(
            camera_params[cam_id], cam_list[cam_id], update_intrinsic, update_distort
        )

        points2d_mask = camera_indices == cam_id
        points3d_where = point_indices[points2d_mask]
        points_proj[points2d_mask, :] = cam_list[cam_id].project(
            points3d[points3d_where][None, :]
        )

    res = points_proj - points_2d
    res = res.ravel()

    return res
