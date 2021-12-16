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

points2d has the format B x T x J x 2. Units are in pixels. calib is a nested dictionary. 


- Visualize the 2d pose.
```python
import matplotlib.pyplot as plt
img = camNet.plot_2d(0, points='points2d')
plt.figure(figsize=(20,20))
plt.imshow(img, cmap='gray')
plt.axis('off')
```

- Do the bundle adjustment.
```python
from pyba.pyba import bundle_adjust 
bundle_adjust(camNet)
```

- Visualize the resulting camera rig.
```python
fig = plt.figure(figsize=(10,10))
ax3d = fig.add_subplot(111, projection='3d')

camNet.draw(ax3d, size=20)
camNet.plot_3d(ax3d, img_id=0, size=10)
```


## TODO 
- [ ] Implement distortion, limit to two parameters
- [ ] Implement intrinsic optimization
- [ ] make losses the same with the scipy tutorial