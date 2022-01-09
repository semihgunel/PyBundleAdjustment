import pickle
from itertools import product
from typing import *

import numpy as np

from pyba.Camera import Camera
from pyba.plot import plot_3d
from pyba.util import triangulate


class CameraNetwork:
    def __init__(
        self,
        points2d: np.ndarray,
        calib: Optional[Union[str, dict]] = None,
        image_path: Optional[str] = None,
        bones: Optional[np.ndarray] = None,
        colors: Optional[List[Tuple]] = None,
    ):
        """camera rig composed of multiple cameras. can be used for multi-view triangulation and calibration.

        points2d: numpy array of shape CxTxJx2, where C is the number of cameras, J is the number of joints and T is the time axis.
            points are in the (row,column) format. use plot_2d function for sanity checking.

        calib: should include keys "R", "tvec", "intr", "distort".

        image_path: should include './data/test/camera_{cam_id}_img_00000{img_id}.jpg'

        """

        assert image_path is None or ("cam_id" in image_path and "img_id" in image_path)
        # T x J x 3
        self.image_path = image_path
        self.bones = bones
        self.colors = colors

        # convert (row, column) format into (column, row format)
        #   which is the convention in pyba (and opencv) coordinate system.
        tmp = np.copy(points2d[..., 0])
        points2d[..., 0] = points2d[..., 1]
        points2d[..., 1] = tmp

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

    def triangulate(self, cam_id: Optional[List[int]] = None):
        """cam_id: set of camera indices used for triangulation. if None, all the cameras are used."""
        if cam_id is None:
            cam_id = np.arange(self.get_ncams())
        n_joints = self[0].get_njoints()
        n_images = self.get_nimages()

        for img_id, j_id in product(range(n_images), range(n_joints)):
            cameras = [
                c
                for (idx, c) in enumerate(self)
                if c.can_see(img_id, j_id) and idx in cam_id
            ]
            pt3d = triangulate(
                [c.P for c in cameras],  # set of projection matrices
                [np.array(c[img_id][j_id]) for c in cameras],  # set of 2d points
            )
            self.set_points3d(img_id, j_id, pt3d)

        return self.points3d

    def reprojection_error(self) -> float:
        return [
            np.mean(np.sum(np.abs(c.reprojection_error(self.points3d)), axis=2))
            for c in self
        ]

    def plot_2d(
        self,
        img_id: int,
        points: Optional[str] = "points2d",
    ) -> np.ndarray:
        """points: which points to plot
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

    def plot_3d(
        self,
        ax_3d,
        img_id: int,
        size: float = 1.0,
        thickness: float = 5.0,
    ):
        plot_3d(
            ax_3d,
            self.points3d[img_id] * size,
            self.bones,
            np.array(self.colors) / 255,
            lim=3,
            thickness=thickness,
            zorder=None,
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
        cam_id: Optional[List[int]] = None,
    ):
        from pyba.pyba import bundle_adjust

        if cam_id is None:
            cam_id = np.arange(self.get_ncams())

        # set 2d points invisible for camera that are not going to be used,
        #   now, they are not going to be used during bundle adjustment
        unused_cam_id = np.setdiff1d(np.arange(self.get_ncams()), cam_id)
        backup = {cid: np.copy(self[cid].points2d) for cid in unused_cam_id}
        for cid in unused_cam_id:
            self[cid].points2d[:] = 0

        bundle_adjust(self, max_num_images, update_intrinsic, update_distort)

        # put 2d points back
        for cid in unused_cam_id:
            self[cid].points2d = backup[cid]
