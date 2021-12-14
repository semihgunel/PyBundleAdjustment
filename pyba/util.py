import numpy as np
import cv2

def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.
    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)
    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert type(projective) == np.ndarray
    assert (projective.shape[0] == 4) | (projective.shape[0] == 3)
    return (projective / projective[-1, :])[0:-1, :]

def extrinsic_from_Rt(R:np.ndarray, tvec:np.ndarray):
    extrinsic = np.zeros(shape=(3, 4))
    extrinsic[:3, :3] = R
    extrinsic[:, 3] = np.squeeze(tvec)
    return extrinsic


def P_from_RtvecK(R:np.ndarray, tvec:np.ndarray, intrinsic:np.ndarray):
    if R is None or tvec is None or intrinsic is None:
        return None
    extrinsic = extrinsic_from_Rt(R, tvec)
    P = np.matmul(intrinsic, extrinsic)
    return P


"""
n-view linear triangulation
https://github.com/smidm/camera.py/blob/master/camera.py
"""


def nview_linear_triangulation_single(cameras, correspondences):
    """
    Computes ONE world coordinate from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param correspondences: image coordinates correspondences in n views
    :type correspondences: numpy.ndarray, shape=(2, n)
    :return: world coordinate
    :rtype: numpy.ndarray, shape=(3, 1)
    """
    assert len(cameras) >= 2
    assert type(cameras) == list
    assert correspondences.shape == (2, len(cameras))

    def _construct_D_block(P, uv):
        """
        Constructs 2 rows block of matrix D.
        See [1, p. 88, The Triangulation Problem]
        :param P: camera matrix
        :type P: numpy.ndarray, shape=(3, 4)
        :param uv: image point coordinates (xy)
        :type uv: numpy.ndarray, shape=(2,)
        :return: block of matrix D
        :rtype: numpy.ndarray, shape=(2, 4)
        """
        return np.vstack((uv[0] * P[2, :] - P[0, :], uv[1] * P[2, :] - P[1, :]))

    D = np.zeros((len(cameras) * 2, 4))
    for cam_idx, cam, uv in zip(range(len(cameras)), cameras, correspondences.T):
        D[cam_idx * 2 : cam_idx * 2 + 2, :] = _construct_D_block(cam, uv)
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    return p2e(u[:, -1, np.newaxis])


def nview_linear_triangulations(cameras, image_points):
    """
    Computes world coordinates from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param image_points: image coordinates of m correspondences in n views
    :type image_points: sequence of m numpy.ndarray, shape=(2, n)
    :return: m world coordinates
    :rtype: numpy.ndarray, shape=(3, m)
    """
    assert type(cameras) == list
    assert type(image_points) == list
    assert len(cameras) == image_points[0].shape[1]
    assert image_points[0].shape[0] == 2

    world = np.zeros((3, len(image_points)))
    for i, correspondence in enumerate(image_points):
        world[:, i] = np.squeeze(
            nview_linear_triangulation_single(cameras, correspondence)
        )
    return world


def triangulate(cam_list, point_list):
    """
    :param cam_list: list of camera object
    :param point_list: list of nx2 numpy arrays
    """
    num_cameras = len(cam_list)
    num_points = point_list[0].shape[0]
    image_points = []
    for count_points in range(num_points):
        correspondence = np.empty(shape=(2, num_cameras))
        for count_cameras in range(num_cameras):
            correspondence[:, count_cameras] = point_list[count_cameras][
                count_points
            ]
        image_points.append(correspondence)
    points3d = nview_linear_triangulations(cam_list, image_points)
    if points3d.shape[0] == 3:
        points3d = points3d.transpose()
    print(points3d.shape, num_cameras, correspondence.shape)
    return points3d


