import torch
import torch.nn.functional as F
from data.dataset import MotionDataset
from model.rmib import RMIB
import data.utils as utils

from vis.vis import display_with_keys
import random

# RMIB parameters
total_frames = 300
past_context = 10
target_frames = [30, 60, 90, 120, 150, 180, 210, 240, 270, 299]

# dataset parameters
window_size = 300
offset = 150
target_fps = 30

# dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
test_dset = MotionDataset("D:/data/LaFAN1", train=False, window=window_size, offset=offset, phase=False, target_fps=target_fps)

def anim(x):
    return x.detach().cpu().reshape(x.shape[0], -1, 3).numpy()

if __name__ == "__main__":
    seed = 777
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare training data
    local_quat = test_dset.extract_features("local_quat")
    global_vel_root = test_dset.extract_features("global_vel_root") / target_fps
    contacts = torch.cat([test_dset.extract_features("contacts_l"), test_dset.extract_features("contacts_r")], dim=-1)
    global_pos = test_dset.extract_features("global_pos")
    parents = test_dset.extract_features("parents")
    offset = test_dset.extract_features("offset").reshape(local_quat.shape[0], local_quat.shape[1], -1, 3)

    # data augmentation
    local_quat = local_quat.reshape(local_quat.shape[0], local_quat.shape[1], -1, 4)
    global_vel_root = global_vel_root.reshape(global_vel_root.shape[0], global_vel_root.shape[1], -1, 3)
    global_pos = global_pos.reshape(global_pos.shape[0], global_pos.shape[1], -1, 3)

    delta_quat = utils.delta_rotate_at_frame_torch(local_quat, 10)
    
    local_quat[..., 0:1, :] = utils.quat_mul_torch(delta_quat, local_quat[..., 0:1, :])
    global_vel_root = utils.quat_mul_vec_torch(delta_quat, global_vel_root)
    global_pos = utils.quat_mul_vec_torch(delta_quat, global_pos)
    
    local_quat = local_quat.reshape(local_quat.shape[0], local_quat.shape[1], -1)
    global_vel_root = global_vel_root.reshape(global_vel_root.shape[0], global_vel_root.shape[1], -1)
    global_pos = global_pos.reshape(global_pos.shape[0], global_pos.shape[1], -1)
    
    # load trained model
    input_dims = {
        "state_encoder": local_quat.shape[-1] + global_vel_root.shape[-1] + contacts.shape[-1],
        "offset_encoder": local_quat.shape[-1] + global_vel_root.shape[-1],
        "target_encoder": local_quat.shape[-1],
    }
    model = RMIB(dof=local_quat.shape[-1] + global_vel_root.shape[-1] + contacts.shape[-1],
                input_dims=input_dims,
                device=device).to(device)
    model.load_state_dict(torch.load("save/rmib.pth")["generator"])
    
    model.eval()
    with torch.no_grad():
        while True:
            i = random.randint(0, local_quat.shape[0]-1)
            # print("Progress ", round(100 * i / num_batch, 2), "%", end="\r")
            lq = local_quat[i:i+1].to(device)
            gvr = global_vel_root[i:i+1].to(device)
            c = contacts[i:i+1].to(device)

            lq_preds, gvr_preds, c_preds = [], [], []
            for t_idx, t_frame in enumerate(target_frames):
                model.init_hidden(1)
                model.set_target([lq[:, t_frame, :], gvr[:, t_frame, :]])

                # prediction
                if t_idx == 0:
                    for f in range(0, past_context):
                        lq_pred, gvr_pred, c_pred = model([lq[:, f, :], gvr[:, f, :], c[:, f, :]], t_frame -f)
                        lq_preds.append(lq[:, f+1, :])
                        gvr_preds.append(gvr[:, f+1, :])
                        c_preds.append(c[:, f+1, :])
                    for f in range(past_context, t_frame):
                        lq_pred, gvr_pred, c_pred = model([lq_pred, gvr_pred, c_pred], t_frame - f)
                        lq_preds.append(lq_pred)
                        gvr_preds.append(gvr_pred)
                        c_preds.append(c_pred)
                else:
                    for f in range(-past_context, 0):
                        lq_pred_norm = utils.quat_normalize_torch(lq_preds[f].reshape(-1, 4))
                        lq_pred, gvr_pred, c_pred = model([lq_pred_norm.reshape(1, -1),
                                        gvr_preds[f],
                                        (c_preds[f] > 0.9).float()], t_frame - target_frames[t_idx-1] - f)
                    for f in range(0, t_frame - target_frames[t_idx-1]):
                        lq_pred, gvr_pred, c_pred = model([lq_pred, gvr_pred, c_pred], t_frame - target_frames[t_idx-1] - f)
                        lq_preds.append(lq_pred)
                        gvr_preds.append(gvr_pred)
                        c_preds.append(c_pred)

            lq_preds = torch.stack(lq_preds, dim=1)
            gvr_preds = torch.stack(gvr_preds, dim=1)
            c_preds = torch.stack(c_preds, dim=1)

            # solve FK
            gp = global_pos[i:i+1].to(device)
            gpr_preds = gp[:, 0:1, :3] + torch.cumsum(gvr_preds, dim=1)
            _, gp_preds = utils.quat_fk_torch(utils.quat_normalize_torch(lq_preds.reshape(*lq_preds.shape[:2], -1, 4)),
                                              gpr_preds.reshape(*gpr_preds.shape[:2], -1, 3),
                                              offset[i:i+1][:, :target_frames[-1]].to(device),
                                              parents)

            gp_preds = gp_preds.reshape(*gp_preds.shape[:2], -1)

            pred_anim = anim(gp_preds[0])
            gt_anim = anim(gp[0])
            key_anim = gt_anim[target_frames]
            display_with_keys(pred_anim, parents, key_anim, target_frames, gt=gt_anim)#, save_gif=True, gif_name="save/RMIB.gif")

            answer = input("Enter 'q' to quit: ")
            if answer == "q":
                break