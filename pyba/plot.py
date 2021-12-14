import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from typing import *

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (100, 100), (100, 100), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_reference_frame(ax3d, center:np.ndarray, x:np.ndarray, y:np.ndarray, z:np.ndarray, text:Optional[str]=None):
    # create the arrows
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
    a = Arrow3D([center[0], x[0]], [center[1], x[1]], [center[2], x[2]], **arrow_prop_dict, color='r')
    ax3d.add_artist(a)
    a = Arrow3D([center[0], y[0]], [center[1], y[1]], [center[2], y[2]], **arrow_prop_dict, color='b')
    ax3d.add_artist(a)
    a = Arrow3D([center[0], z[0]], [center[1], z[1]], [center[2], z[2]], **arrow_prop_dict, color='g')
    ax3d.add_artist(a)

    #Give the optical axis a name
    # ax3d.text(y[0], y[1], y[2], r'$y$')

    #Give the optical axis a name
    ax3d.text(center[0], center[1], center[2], text)
