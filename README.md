# PyTorch Implementations of Motion In-betweening Papers
This repository contains PyTorch implementations of motion in-betweening papers.

## Implementations
* Recurrent Transition Networks for Character Locomotion [[Paper]](https://arxiv.org/abs/1810.02363) [[Code]](model/rtn.py)
* Robust Motion In-betweening [[Paper]](https://arxiv.org/abs/2102.04942) [Code (Coming soon)]

## Motion features
We first extract the motion features from `.bvh` files and store them in a `.pkl` file in a key-value data structure.

After then, you can extract the motion features by using predefined keys and sliding window parameters `window_size` and `offset`.

The keys are:
```
offset: bone offsets
parents: parent indices
global_pos: global joint positions
global_vel: global joint velocities
local_pos: local joint positions
local_vel: local joint velocities
local_euler: local joint rotation in Euler angles (degrees)
local_quat: local joint rotation in quaternion
phase: labelled phase features
```

### Example code
```python
from data.dataset import MotionDataset
train_dset = MotionDataset(path, train, window, offset, phase, target_fps)
train_data = train_dset.extract_features("local_quat")
```

We also provide phase-functioned encoders and decoders through `PhaseMLP` in [here](model/base_model.py), which are originally from the paper [Phase-Functioned Neural Networks for Character Control](https://dl.acm.org/doi/abs/10.1145/3072959.3073663).

To use labelled phase features, you should add `.phase` files within the same directory of `.bvh` files.

`PhaseRTN` and `PhaseRMIB` use phase-functioned MLPs for their encoders and decoders.

## Training
You can train the network by using a simple command:
```
python train_{network}.py
```
After training the network, the checkpoint and the log will be saved in `save` and `log`, respectively.

## Test with visualization
We provide visualization with a simple stick character through `display` fucntion in [here](vis/vis.py). Feel free to modify as you want!
![teaser](media/rtn.gif)