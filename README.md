# Python Bundle Adjustment

- Load the data.
```python
from pyba.CameraNetwork import CameraNetwork
import pickle
import glob
import numpy as np

image_path = './data/test/camera_{cam_id}_img_00000{img_id}.jpg'
pr_path = './data/test/df3d_2/pose_result*.pkl'

d = pickle.load(open(glob.glob(pr_path)[0], 'rb'))
camNet = CameraNetwork(points2d=d['points2d'], calib=d, image_path=image_path)
```

points2d is a numpy array with shape T x J x 2. 
All units are in pixels. calib is a nested dictionary. 

```python
calib = {0: {'R': array([[ 0.90885957,  0.006461  , -0.41705219],
         [ 0.01010426,  0.99924554,  0.03750006],
         [ 0.41697983, -0.0382963 ,  0.90810859]]),
  'tvec': array([1.65191596e+00, 2.22582670e-02, 1.18353733e+02]),
  'intr': array([[1.60410e+04, 0.00000e+00, 2.40000e+02],
         [0.00000e+00, 1.59717e+04, 4.80000e+02],
         [0.00000e+00, 0.00000e+00, 1.00000e+00]]),
  'distort': array([0., 0., 0., 0., 0.])},
 1: {'R': array([[ 0.59137248,  0.02689833, -0.80594979],
         [-0.00894927,  0.9996009 ,  0.02679478],
         [ 0.80634887, -0.00863303,  0.59137718]]),
  'tvec': array([ 1.02706542e+00, -9.25820468e-02,  1.18251732e+02]),
  'intr': array([[1.60410e+04, 0.00000e+00, 2.40000e+02],
         [0.00000e+00, 1.59717e+04, 4.80000e+02],
         [0.00000e+00, 0.00000e+00, 1.00000e+00]]),
  'distort': array([0., 0., 0., 0., 0.])},
}
```


- Visualize the 2d pose.
```python
import matplotlib.pyplot as plt
img = camNet.plot_2d(0, points='points2d')
plt.figure(figsize=(20,20))
plt.imshow(img, cmap='gray')
plt.axis('off')
```

![image](https://user-images.githubusercontent.com/20509861/146374004-6ae50ba5-67b8-4326-a115-9901e102df6d.png)


- Do the bundle adjustment.
```python
from pyba.pyba import bundle_adjust 
bundle_adjust(camNet)
```

```
   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
       0              1         7.1659e+05                                    7.27e+05    
       1              2         2.9376e+05      4.23e+05       1.08e+01       3.12e+05    
       2              4         2.6084e+05      3.29e+04       2.39e+00       1.85e+05    
       3              5         2.4676e+05      1.41e+04       3.04e+00       2.20e+04    
       4              7         2.4604e+05      7.20e+02       1.32e+00       1.75e+04    
       5              8         2.4579e+05      2.53e+02       2.67e+00       2.86e+04    
       6              9         2.4487e+05      9.20e+02       2.53e+00       2.18e+04    
       7             10         2.4472e+05      1.43e+02       2.48e+00       2.02e+04    
       8             11         2.4441e+05      3.18e+02       6.71e-01       1.77e+03    
       9             12         2.4440e+05      9.43e+00       6.78e-01       2.13e+03    
`ftol` termination condition is satisfied.
Function evaluations 12, initial cost 7.1659e+05, final cost 2.4440e+05, first-order optimality 2.13e+03.
```


- Visualize the resulting camera rig.
```python
fig = plt.figure(figsize=(10,10))
ax3d = fig.add_subplot(111, projection='3d')

camNet.draw(ax3d, size=20)
camNet.plot_3d(ax3d, img_id=0, size=10)
```

<img src="https://user-images.githubusercontent.com/20509861/146374042-1a3a65d2-310d-4783-b6d0-6864c582959f.png" width="500">


## TODO 
- [ ] Implement distortion, limit to two parameters
- [ ] Implement intrinsic optimization
- [ ] make losses the same with the scipy tutorial
