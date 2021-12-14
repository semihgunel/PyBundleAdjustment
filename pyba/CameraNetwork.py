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
        points2d:Union[str, np.ndarray],
        calib:Union[str, dict],
        num_images:Optional[int]=None,
        image_path:Optional[str]=None,
        bones:Optional[List[List[int]]]=None
    ):
        ''' 
        pts2d: if string, assumed to be the path 
        '''
        # T x J x 3
        self.bones = bones

        points2d = pickle.load(points2d) if isinstance(points2d, str) else points2d
        points2d = points2d[:, :num_images] if num_images is not None else points2d
        calib = pickle.load(calib) if isinstance(calib, str) else calib

        self.cam_list = list()
        for cam_id in range(points2d.shape[0]):
            cam = Camera(intrinsic=calib[cam_id]['intr'],
                         R=calib[cam_id]['R'],
                         tvec=calib[cam_id]['tvec'],
                         distort=calib[cam_id]['distort'],
                         points2d=points2d[cam_id],
                         image_path=image_path.format(cam_id=cam_id, img_id='{img_id}'))
            self.cam_list.append(cam)

        self._points3d = np.zeros((self.get_nimages(), self.get_njoints(), 3))
        self.triangulate()

    def __getitem__(self, item):
        return self.cam_list[item]

    def get_ncams(self):
        return len(self.cam_list)

    def get_nimages(self):
        return self[0].points2d.shape[0]

    def get_njoints(self):
        return self[0].points2d.shape[1]

    @property
    def points3d(self):
        return self._points3d

    def set_points3d(self, img_id, jid, pts):
        self._points3d[img_id, jid] = pts

    def triangulate(self):
        n_joints = self[0].get_njoints()
        n_images = self.get_nimages()
        
        for img_id, j_id in product(range(n_images), range(n_joints)):
            self.set_points3d(img_id, j_id, triangulate(
                [c.P for c in self if c.can_see(img_id, j_id)], # set of projection matrices
                [c[img_id][j_id] for c in self if c.can_see(img_id, j_id)]  # set of 2d points
            )) 

        return self.points3d


    def reprojection_error(self):
        return [np.mean(np.sum(np.abs(c.reprojection_error(self.points3d)), axis=2)) for c in self]


    def plot_2d(self, img_id:int, points:Optional[str]='points2d'):
        """ points: which points to plot 
                can be point2d or reprojection
        """

        def get_plot_points2d(cid, img_id, points:str):
            if points=='points2d':
                return self[cid][img_id]
            elif points=='reprojection':
                return self[cid].project(self.points3d[[img_id]])
            else:
                raise NotImplementedError

        return np.concatenate([c.plot_2d(img_id, points2d=get_plot_points2d(cid, img_id, points)) for (cid, c) in enumerate(self)], axis=1)


    def summarize(self):
        return {cid:c.summarize() for (cid, c) in enumerate(self)}
