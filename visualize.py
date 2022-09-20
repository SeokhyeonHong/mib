from data import bvh, dataset
from vis import vis
from random import randint

# training parameters
epochs = 100
lr = 1e-3
batch_size = 32

window_size = 1000
offset = 25
# window_time = 1.0
# phase_channels = 8

motion = dataset.MotionDataset("D:/data/LaFAN1", train=False, window=window_size, offset=offset, keys=["global_pos"], write_pkl=False)

# PhaseNet parameters
# dof = motion.data.shape[-1]

if __name__ == "__main__":
    idx = randint(0, len(motion))
    clip = motion.data()[idx].view(-1, 22, 3).numpy()
    vis.display(clip, motion.parents(), fps=30)