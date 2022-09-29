import torch
import torch.nn.functional as F
from data.dataset import MotionDataset
from model.rmib import PhaseRMIB
import data.utils as utils

from vis.vis import display
import random

# RMIB parameters
total_frames = 50
past_context = 10
target_frame = 40

# dataset parameters
window_size = 50
offset = 25
target_fps = 30

# dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
test_dset = MotionDataset("D:/data/PFNN", train=False, window=window_size, offset=offset, phase=True, target_fps=target_fps)

def item(x):
    return x.detach().cpu().numpy()

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
    phase = test_dset.extract_features("phase").squeeze(-1)

    # data augmentation (rotate according to the 10th frame)
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

    # training settings
    input_dims = {
        "state_encoder": local_quat.shape[-1] + global_vel_root.shape[-1] + contacts.shape[-1],
        "offset_encoder": local_quat.shape[-1] + global_vel_root.shape[-1],
        "target_encoder": local_quat.shape[-1],
    }

    generator = PhaseRMIB(dof=local_quat.shape[-1] + global_vel_root.shape[-1] + contacts.shape[-1],
                input_dims=input_dims,
                device=device).to(device)
    generator.load_state_dict(torch.load("save/prmib.pth")["generator"])
    
    generator.eval()
    with torch.no_grad():
        while True:
            i = random.randint(0, local_quat.shape[0] - 1)

            lq = local_quat[i:i+1].to(device)
            gvr = global_vel_root[i:i+1].to(device)
            c = contacts[i:i+1].to(device)
            p = phase[i:i+1].to(device)

            # network initialization
            target_frame = 40
            generator.init_hidden(1)
            generator.set_target([lq[:, target_frame, :], gvr[:, target_frame, :]], p[:, target_frame])

            # prediction
            lq_preds, gvr_preds, c_preds = [], [], []
            input_phase = p[:, 0]
            for f in range(0, target_frame):
                if f < past_context:
                    lq_pred, gvr_pred, c_pred, delta_phase_pred = generator([lq[:, f, :], gvr[:, f, :], c[:, f, :]], input_phase, target_frame -f)
                    lq_preds.append(lq[:, f+1, :])
                    gvr_preds.append(gvr[:, f+1, :])
                    c_preds.append(c[:, f+1, :])
                    input_phase = p[:, f+1]
                else:
                    lq_pred, gvr_pred, c_pred, delta_phase_pred = generator([lq_pred, gvr_pred, c_pred], input_phase, target_frame - f)
                    lq_preds.append(lq_pred)
                    gvr_preds.append(gvr_pred)
                    c_preds.append(c_pred)
                    input_phase = input_phase + delta_phase_pred
                    input_phase = torch.where(input_phase >= 1, input_phase - 1, input_phase)

            lq_preds = torch.stack(lq_preds, dim=1)
            gvr_preds = torch.stack(gvr_preds, dim=1)
            c_preds = torch.stack(c_preds, dim=1)

            # solve FK
            gp = global_pos[i:i+1].to(device)
            gpr_preds = gp[:, 0:1, :3] + torch.cumsum(gvr_preds, dim=1)
            _, gp_preds = utils.quat_fk_torch(utils.quat_normalize_torch(lq_preds.reshape(*lq_preds.shape[:2], -1, 4)),
                                              gpr_preds.reshape(*gpr_preds.shape[:2], -1, 3),
                                              offset[i:i+1][:, 1:target_frame+1].to(device),
                                              parents)

            gp_preds = gp_preds.reshape(*gp_preds.shape[:2], -1)

            pred_anim = item(gp_preds[0]).reshape(gp_preds[0].shape[0], -1, 3)
            gt_anim = item(gp[0]).reshape(gp[0].shape[0], -1, 3)
            display(pred_anim, parents.numpy(), gt=gt_anim, eye=(0, 50, 125), bone_radius=0.3)

            answer = input("Enter 'q' to quit: ")
            if answer == "q":
                break