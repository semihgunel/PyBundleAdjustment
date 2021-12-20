import pickle
from itertools import product
from typing import *

import numpy as np

from pyba.Camera import Camera
from pyba.util import triangulate


class CameraNetwork:
    """ 
    camera rig. can be used for multi-view triangulation and calibration.
    """

    def __init__(
        self,
        points2d: Union[str, np.ndarray],
        calib: Optional[Union[str, dict]] = None,
        num_images: Optional[int] = None,
        image_path: Optional[str] = None,
        bones: Optional[np.ndarray] = None,
        colors: Optional[List[Tuple]] = None,
    ):
        """ 
        points2d: CxTxJx2, where C is the number of cameras, J is the number of joints 
        """
        # T x J x 3
        self.image_path = image_path
        self.bones = bones
        self.colors = colors

        self.points2d = pickle.load(points2d) if isinstance(points2d, str) else points2d
        self.points2d = points2d[:, :num_images] if num_images is not None else points2d
        if calib is not None:
            calib = pickle.load(calib) if isinstance(calib, str) else calib

        self.cam_list = list()
        for cam_id in range(points2d.shape[0]):
            image_path = (
                self.image_path.replace("{cam_id}", str(cam_id))
                if image_path is not None
                else None
            )
            cal = (
                {k: calib[cam_id][k] for k in calib[cam_id].keys()}
                if calib is not None
                else {}
            )
            cam = Camera(points2d=points2d[cam_id], image_path=image_path, **cal)
            self.cam_list.append(cam)

        self._points3d = np.zeros((self.get_nimages(), self.get_njoints(), 3))

        if self.has_calibration():
            self.triangulate()

    @property
    def points3d(self):
        return self._points3d

    def set_points3d(self, img_id, jid, pts):
        self._points3d[img_id, jid] = np.squeeze(pts)

    def __getitem__(self, item):
        return self.cam_list[item]

    def get_ncams(self):
        return len(self.cam_list)

    def get_nimages(self):
        return self[0].points2d.shape[0]

    def get_njoints(self):
        return self[0].points2d.shape[1]

    def has_calibration(self):
        return all([c.has_calibration() for c in self])

    def triangulate(self):
        n_joints = self[0].get_njoints()
        n_images = self.get_nimages()

        for img_id, j_id in product(range(n_images), range(n_joints)):
            self.set_points3d(
                img_id,
                j_id,
                triangulate(
                    [
                        c.P for c in self if c.can_see(img_id, j_id)
                    ],  # set of projection matrices
                    np.array(
                        [c[img_id][j_id] for c in self if c.can_see(img_id, j_id)]
                    ),  # set of 2d points
                ),
            )

        return self.points3d

    def reprojection_error(self):
        return [
            np.mean(np.sum(np.abs(c.reprojection_error(self.points3d)), axis=2))
            for c in self
        ]

    def plot_2d(
        self, img_id: int, points: Optional[str] = "points2d",
    ):
        """ points: which points to plot 
                can be point2d or reprojection
        """

        def get_plot_points2d(cid, img_id, points: str):
            if points == "points2d":
                return self[cid][img_id]
            elif points == "reprojection":
                return np.squeeze(self[cid].project(self.points3d[[img_id]]))
            else:
                raise NotImplementedError

        return np.concatenate(
            [
                c.plot_2d(
                    img_id,
                    points2d=get_plot_points2d(cid, img_id, points),
                    bones=self.bones,
                    colors=self.colors,
                )
                for (cid, c) in enumerate(self)
            ],
            axis=1,
        )

    def plot_3d(self, ax_3d, img_id: int, size: float = 1):
        for bone in self.bones:
            ax_3d.plot(
                self.points3d[img_id, bone, 0] * size,
                self.points3d[img_id, bone, 1] * size,
                self.points3d[img_id, bone, 2] * size,
                c="blue",
                linewidth=2,
            )

    def summarize(self):
        calib = {cid: c.summarize() for (cid, c) in enumerate(self)}
        return {**calib, **{"points3d": self.points3d, "points2d": self.points2d}}

    def draw(self, ax3d, size: float = 1):
        for cid, c in enumerate(self):
            c.draw(ax3d, size=size, text=f"{cid}")

    def bundle_adjust(
        self,
        max_num_images: int = 1e3,
        update_intrinsic: bool = True,
        update_distort: bool = True,
    ):
        from pyba.pyba import bundle_adjust

        bundle_adjust(self, max_num_images, update_intrinsic, update_distort)

