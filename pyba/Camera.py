from typing import *
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyba.util import P_from_RtvecK

class Camera:
    def __init__(self,
                 intrinsic:np.ndarray,
                 R:np.ndarray,
                 tvec:np.ndarray,
                 points2d:np.ndarray,
                 distort:np.ndarray=None,
                 image_path:Optional[dict]=None):
        """ 
        fx, fy: focal length in pixels
        tvec: translation vector
        cx, cy: optical axis in pixels
        points2d: numpy array in pixels, TxJx2
        distort: list with 5 numbers
        image_path: Example: img_{img_id}.jpg
        """

        assert points2d.ndim == 3 and points2d.shape[2] == 2
        assert R.ndim == 2 and R.shape[0] == 3  and R.shape[1] == 3       
        assert tvec.ndim == 1 and tvec.shape[0] == 3        
        assert distort is None or (distort.ndim == 1 and distort.shape[0] == 5)
        assert intrinsic.ndim == 2 and intrinsic.shape[0] == 3 and intrinsic.shape[1] == 3

        self.R = None
        self.tvec = None
        self.intrinsic = None
        self.distort = None
        
        self.image_path = image_path
        self._points2d = points2d

        self.intrinsic = intrinsic
        self.tvec = tvec
        self.R = R
        self.distort = self.set_distort(np.zeros(5, dtype=np.float)) if distort is None else self.set_distort(distort)

    @property
    def P(self):
        return P_from_RtvecK(self.R, self.tvec, self.intrinsic)

    @property
    def rvec(self):
        return cv2.Rodrigues(self.R)[0]

    @property
    def points2d(self):
        return self._points2d

    def __getitem__(self, idx:int):
        return self.points2d[idx]

    def set_intrinsic(self, intrinsic):
        self.intrinsic = intrinsic

    def can_see(self, img_id, jid):
        return (not np.any(self.points2d[img_id, jid] == 0))

    def can_see_mask(self):
        return np.any(self.points2d==0, axis=2)

    def get_njoints(self):
        return self.points2d.shape[1]

    def get_image(self, img_id:int):
        img = plt.imread(self.image_path.format(img_id=img_id))
        return img

    def summarize(self):
        raise NotImplementedError

    def plot_2d(self, img_id:int, points2d:Optional[np.ndarray]=None):
        img = self.get_image(img_id)
        points2d = self.points2d[img_id] if points2d is None else points2d
        for jid in range(self.get_njoints()):
            img = cv2.circle(img, points2d[jid].astype(int), 5, [255, 0, 0], 5)
        return img


    def project(self, points3d: np.ndarray):
        """
        points3d: TxJx3
        returns: points2d: TxJx2
        """

        assert points3d.ndim == 3 and points3d.shape[2] == 3

        os = list(points3d.shape)

        # opencv wants 3xn matrix
        points3d = np.swapaxes(points3d, 0, 2) # txjx3 -> 3xtxj
        points3d = points3d.reshape(os[0]*os[1], os[2])
        points2d, _ = cv2.projectPoints(
            points3d, self.rvec , self.tvec, self.intrinsic, self.distort
        )

        # reshape back to TxJx2
        os[2] = 2
        points2d = points2d.reshape(os)
        return np.squeeze(points2d)

    def reprojection_error(self, points3d:np.ndarray):
        err = self.points2d - self.project(points3d)
        err[self.can_see_mask(), :] = 0
        return err

       
