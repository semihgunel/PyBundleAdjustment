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

- Do the bundle adjustment.

```python
from pyba.pyba import bundle_adjust 
bundle_adjust(camNet)
```

## TODO 
- [ ] optimizing distortion is broken
- [ ] Implement distortion
- [ ] Implement intrinsic optimization
- [ ] make losses the same iwth the tutorial