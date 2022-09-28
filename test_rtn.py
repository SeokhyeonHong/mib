import torch
import numpy as np

from data.dataset import MotionDataset
from model.rtn import RTN
from vis.vis import display

# RTN parameters
teacher_prob = 0.2
total_frames = 50
past_context = 10
target_frame = 40

# dataset parameters
window_size = 50
offset = 25
target_fps = 30

# dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
test_dset = MotionDataset("D:/data/PFNN", train=False, window=window_size, offset=offset, phase=False, target_fps=target_fps)

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().view(tensor.shape[0], -1, 3).numpy()

if __name__ == "__main__":
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare training data
    gv_root = test_dset.extract_features("global_vel_root")
    gp_root = test_dset.extract_features("global_pos_root")
    gp = test_dset.extract_features("global_pos")
    gp_root_rel = gp - gp_root.tile(1, 1, gp.shape[-1] // gp_root.shape[-1])
    test_data = torch.cat([gv_root, gp_root_rel[..., 3:]], dim=-1)
    
    # load model
    model = RTN(test_data.shape[-1]).to(device)
    model.load_state_dict(torch.load("save/rtn.pth")["model"])
    model.eval()
    with torch.no_grad():
        for idx in range(test_data.shape[0]):
            data = test_data[idx:idx+1].to(device)
            mean, std = data.mean(), data.std()
            data = (data - mean) / (std + 1e-8)

            # network initialization
            model.init_hidden(1, data[:, 0, :])
            model.set_target(data[:, target_frame, :])

            # RTN
            preds = []
            for f in range(0, target_frame):
                input_data = data[:, f, :] if f < past_context else pred
                pred = model(input_data)
                preds.append(data[:, f+1, :] if f < past_context else pred)
            preds = torch.stack(preds, dim=1) * std + mean

            # gt
            gt = data[:, 1:target_frame+1, :] * std + mean

            # make animation to visualize
            pred_rtn = tensor_to_numpy(preds[0])
            gt = tensor_to_numpy(gt[0])

            ########## ground truth motion ##########
            def get_joint_pos(data):
                start_root_pos = gp_root[idx, 0:1].numpy()
                root_pos_delta = np.cumsum(data[:, 0:1, :] / target_fps, axis=0)
                root_pos = start_root_pos + root_pos_delta

                joint_pos = data[:, 1:]
                anim = np.concatenate([root_pos, joint_pos+root_pos], axis=1)
                return anim

            gt_anim = get_joint_pos(gt)
            rtn_anim = get_joint_pos(pred_rtn)
            
            # correction = gt_anim[-1] - pred_anim[-1]
            # for frame in range(past_context, target_frame):
            #     t = 1 - (target_frame - frame) / (past_context - target_frame)
            #     pred_anim[frame] += t * correction
            
            display(rtn_anim, test_dset.parents, gt=gt_anim, fps=30, bone_radius=0.5, eye=(0, 50, 125))
            message = input("Type 'q' to quit: ")
            if message == "q":
                break