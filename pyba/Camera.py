from pyba.plot import draw_reference_frame
from typing import *
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyba.util import P_from_RtvecK


class Camera:
    def __init__(
        self,
        points2d: np.ndarray,
        intr: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
        distort: Optional[np.ndarray] = None,
        image_path: Optional[dict] = None,
    ):
        """ 
        fx, fy: focal length in pixels
        tvec: translation vector
        cx, cy: optical axis in pixels
        points2d: numpy array in pixels, TxJx2
        distort: list with 5 numbers
        image_path: Example: img_{img_id}.jpg
        """

        # fmt: off
        assert points2d.ndim == 3 and points2d.shape[2] == 2
        assert R is None or R.ndim == 2 and R.shape[0] == 3 and R.shape[1] == 3
        assert tvec is None or tvec.ndim == 1 and tvec.shape[0] == 3 
        assert distort is None or (distort.ndim == 1 and distort.shape[0] == 5)
        assert intr is None or intr.ndim == 2 and intr.shape[0] == 3 and intr.shape[1] == 3
        # fmt: on

        self.image_path = image_path
        self._points2d = points2d

        self.intrinsic = intr
        self.tvec = tvec
        self.R = R
        self.distort = np.zeros(5, dtype=np.float) if distort is None else distort

    @property
    def P(self):
        return P_from_RtvecK(self.R, self.tvec, self.intrinsic)

    @property
    def rvec(self):
        return cv2.Rodrigues(self.R)[0]

    def camera2world(self, pts: np.ndarray):
        return self.R.T @ (pts - self.tvec)

    @property
    def C(self):
        return -1 * self.R.T @ self.tvec

    @rvec.setter
    def rvec(self, rvec):
        self.R = cv2.Rodrigues(rvec)[0]

    @property
    def fx(self):
        return self.intrinsic[0, 0]

    @property
    def fy(self):
        return self.intrinsic[1, 1]

    @property
    def cx(self):
        return self.intrinsic[0, 2]

    @property
    def cy(self):
        return self.intrinsic[1, 2]

    @fx.setter
    def fx(self, fx):
        self.intrinsic[0, 0] = fx

    @fy.setter
    def fy(self, fy):
        self.intrinsic[1, 1] = fy

    @cx.setter
    def cx(self, cx):
        self.intrinsic[0, 2] = cx

    @cy.setter
    def cy(self, cy):
        self.intrinsic[1, 2] = cy

    @property
    def points2d(self):
        return self._points2d

    def __getitem__(self, idx: int):
        return self.points2d[idx]

    def set_intrinsic(self, intrinsic):
        self.intrinsic = intrinsic

    def can_see(self, img_id, jid):
        return not np.any(self.points2d[img_id, jid] == 0)

    def can_see_mask(self):
        return np.any(self.points2d == 0, axis=2)

    def get_njoints(self):
        return self.points2d.shape[1]

    def get_image(self, img_id: int):
        img = plt.imread(self.image_path.format(img_id=img_id))
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def has_calibration(self):
        return all(
            [self.intrinsic is not None, self.R is not None, self.tvec is not None]
        )

    def summarize(self):
        return {
            "R": self.R,
            "tvec": self.tvec,
            "distort": self.distort,
            "intr": self.intrinsic,
        }

    def plot_2d(
        self,
        img_id: int,
        points2d: Optional[np.ndarray] = None,
        bones: Optional[np.ndarray] = None,
        colors: Optional[List[Tuple]] = None,
    ):
        img = self.get_image(img_id)
        points2d = self.points2d[img_id] if points2d is None else points2d

        # bones
        if bones is not None:
            for idx, b in enumerate(bones):
                if self.can_see(img_id, b[0]):
                    img = cv2.line(
                        img,
                        tuple(points2d[b[0]].astype(int)),
                        tuple(points2d[b[1]].astype(int)),
                        colors[idx] if colors is not None else (128, 0, 0),
                        5,
                    )
        else:  # scatter
            for jid in range(self.get_njoints()):
                if self.can_see(img_id, jid):
                    img = cv2.circle(
                        img, tuple(points2d[jid].astype(int)), 5, [0, 0, 128], 5
                    )

        return img

    def project(self, points3d: np.ndarray):
        """
        points3d: TxJx3
        returns: points2d: TxJx2
        """

        assert points3d.ndim == 3 and points3d.shape[2] == 3

        # original shape
        t, j = points3d.shape[0], points3d.shape[1]

        # opencv wants nx3 matrix
        # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        points3d = points3d.reshape(t * j, 3)  # (txj)x3
        points2d, _ = cv2.projectPoints(
            points3d, self.rvec, self.tvec, self.intrinsic, self.distort
        )

        # reshape back to txjx2
        points2d = points2d.reshape(t, j, 2)  # txjx2
        return points2d

    def reprojection_error(self, points3d: np.ndarray):
        err = self.points2d - self.project(points3d)
        err[self.can_see_mask(), :] = 0
        return err

    def draw(self, ax3d, size: float = 1, text: Optional[str] = None):
        x = self.camera2world(np.array([size, 0, 0]))
        y = self.camera2world(np.array([0, size, 0]))
        z = self.camera2world(np.array([0, 0, size]))

        draw_reference_frame(ax3d, center=self.C, x=x, y=y, z=z, text=text)

