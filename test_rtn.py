import torch
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import MotionDataset
from model.rtn import RTN, PhaseRTN
from vis.vis import display

from tqdm import tqdm

batch_size = 1

# RTN parameters
teacher_prob = 0.2
total_frames = 50
past_context = 10
target_frame = 40

# dataset parameters
window_size = 50
offset = 25

# dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
motion = MotionDataset("D:/data/PFNN", train=False, window=window_size, offset=offset, keys=["global_root_vel", "root_rel_pos"], phase=True, sampling_interval=4)
motion_loader = DataLoader(motion, batch_size=batch_size, shuffle=True, drop_last=True)
mean, std = motion.mean(), motion.std()
dof = motion.dof()

def tensor_to_numpy(tensor, mean, std):
    tensor = tensor.cpu()
    mean = mean.cpu()
    std = std.cpu()
    tensor = tensor * std + mean
    return tensor.detach().view(tensor.shape[0], -1, 3).numpy()

if __name__ == "__main__":
    model_rtn = RTN(dof, device=device).to(device)
    model_prtn = PhaseRTN(dof, device=device).to(device)
    
    model_rtn.load_state_dict(torch.load("save/rtn.pth")["model"])
    model_prtn.load_state_dict(torch.load("save/prtn.pth")["model"])
    model_rtn.eval()
    model_prtn.eval()

    with torch.no_grad():
        for idx, (data, phase) in enumerate(tqdm(motion_loader, desc="Batch")):
            data = (data - mean) / std
            data = data.to(device)
            phase = phase.to(device)

            target_data = data[:, target_frame:target_frame+1, :]
            target_phase = phase[:, target_frame:target_frame+1]

            # network initialization
            model_rtn.init_hidden(batch_size, data[:, 0:1, :])
            model_rtn.set_target(target_data)
            model_prtn.init_hidden(batch_size, data[:, 0:1, :], phase[:, 0:1])
            model_prtn.set_target(target_data, target_phase)

            # RTN
            pred_rtn = []
            for f in range(0, target_frame):
                input_data = data[:, f:f+1, :] if f < past_context else pred
                pred = model_rtn(input_data)
                pred_rtn.append(data[:, f:f+1, :] if f < past_context else pred)
            pred_rtn = torch.cat(pred_rtn, dim=1)

            # PhaseRTN
            pred_prtn = []
            for f in range(0, target_frame):
                input_data = data[:, f:f+1, :] if f < past_context else pred
                input_phase = phase[:, f:f+1] if f < past_context else input_phase + delta_phase
                input_phase = torch.where(input_phase < 1, input_phase, input_phase - 1)

                pred, delta_phase = model_prtn(input_data, input_phase)

                pred_prtn.append(data[:, f:f+1, :] if f < past_context else pred)

            pred_prtn = torch.cat(pred_prtn, dim=1)

            # gt
            gt = data[:, :target_frame, :]

            # make animation to visualize
            pred_rtn = tensor_to_numpy(pred_rtn[0], mean[:target_frame], std[:target_frame])
            pred_prtn = tensor_to_numpy(pred_prtn[0], mean[:target_frame], std[:target_frame])
            gt = tensor_to_numpy(gt[0], mean[:target_frame], std[:target_frame])

            ########## ground truth motion ##########
            def get_root_pos(data):
                start_root_pos = motion.get_global_root_pos()[idx * batch_size, 0:1]
                root_pos_delta = np.cumsum(data[:, 0:1, :], axis=0)
                root_pos = start_root_pos + root_pos_delta

                joint_pos = data[:, 1:]
                anim = np.concatenate([root_pos, joint_pos+root_pos], axis=1)
                return anim

            gt_anim = get_root_pos(gt)
            rtn_anim = get_root_pos(pred_rtn)
            prtn_anim = get_root_pos(pred_prtn)
            
            # correction = gt_anim[-1] - pred_anim[-1]
            # for frame in range(past_context, target_frame):
            #     t = 1 - (target_frame - frame) / (past_context - target_frame)
            #     pred_anim[frame] += t * correction
            
            display(rtn_anim, motion.parents(), gt=prtn_anim, fps=30)