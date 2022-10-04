import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import random

from data.dataset import MotionDataset
from model.rmib import RMIB, Discriminator
import data.utils as utils

# training parameters
epochs = 100
lr = 1e-3
batch_size = 32

# RMIB parameters
max_transition = 30
p_min, p_max = 5, 5
total_frames = 50
past_context = 10

# dataset parameters
window_size = 50
offset = 25
target_fps = 30

# dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dset = MotionDataset("D:/data/LaFAN1", train=True, window=window_size, offset=offset, phase=False, target_fps=target_fps)

if __name__ == "__main__":
    seed = 777
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare training data
    local_quat = train_dset.extract_features("local_quat")
    global_vel_root = train_dset.extract_features("global_vel_root") / target_fps
    contacts = torch.cat([train_dset.extract_features("contacts_l"), train_dset.extract_features("contacts_r")], dim=-1)
    global_pos = train_dset.extract_features("global_pos")
    parents = train_dset.extract_features("parents")
    offset = train_dset.extract_features("offset").reshape(local_quat.shape[0], local_quat.shape[1], -1, 3)

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
    num_batch = local_quat.shape[0] // batch_size

    generator = RMIB(dof=local_quat.shape[-1] + global_vel_root.shape[-1] + contacts.shape[-1],
                input_dims=input_dims,
                device=device).to(device)
    discriminator_short = Discriminator(
        input_dim=(global_pos.shape[-1] + global_pos.shape[-1] + global_vel_root.shape[-1]) * 2,
    ).to(device)
    discriminator_long = Discriminator(
        input_dim=(global_pos.shape[-1] + global_pos.shape[-1] + global_vel_root.shape[-1]) * 10,
    ).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    discriminator_optimizer = torch.optim.Adam(
        list(discriminator_short.parameters()) + list(discriminator_long.parameters()),
        lr=lr,
        betas=(0.5, 0.9)
    )
    writer = SummaryWriter("log/RMIB")

    # training loop
    for epoch in range(1, epochs+1):
        loss_sum = 0
        loss_lq_sum, loss_gvr_sum, loss_gp_sum, loss_c_sum = 0, 0, 0, 0
        loss_adv_sum = 0
        train_idx = torch.randperm(local_quat.shape[0])
        for i in tqdm(range(num_batch), desc=f"Epoch {epoch}/{epochs}"):
            # print("Progress ", round(100 * i / num_batch, 2), "%", end="\r")
            idx = train_idx[i*batch_size:(i+1)*batch_size]
            lq = local_quat[idx].to(device)
            gvr = global_vel_root[idx].to(device)
            c = contacts[idx].to(device)

            # network initialization
            target_frame = past_context + random.randint(p_min, p_max)
            generator.init_hidden(batch_size)
            generator.set_target([lq[:, target_frame, :], gvr[:, target_frame, :]])

            # prediction
            lq_preds, gvr_preds, c_preds = [], [], []
            for f in range(0, target_frame):
                if f < past_context:
                    lq_pred, gvr_pred, c_pred = generator([lq[:, f, :], gvr[:, f, :], c[:, f, :]], target_frame -f)
                else:
                    lq_pred, gvr_pred, c_pred = generator([lq_pred, gvr_pred, c_pred], target_frame - f)
                
                lq_preds.append(lq_pred)
                gvr_preds.append(gvr_pred)
                c_preds.append(c_pred)

            lq_preds = torch.stack(lq_preds, dim=1)
            gvr_preds = torch.stack(gvr_preds, dim=1)
            c_preds = torch.stack(c_preds, dim=1)

            # solve FK
            gp = global_pos[idx].to(device)
            gpr_preds = gp[:, 0:1, :3] + torch.cumsum(gvr_preds, dim=1)
            _, gp_preds = utils.quat_fk_torch(utils.quat_normalize_torch(lq_preds.reshape(*lq_preds.shape[:2], -1, 4)),
                                              gpr_preds.reshape(*gpr_preds.shape[:2], -1, 3),
                                              offset[idx][:, 1:target_frame+1].to(device),
                                              parents)
            gp_preds = gp_preds.reshape(*gp_preds.shape[:2], -1)

            # reconstruction loss (local quaternion, global root velocity, global position, contacts)
            loss_lq  = F.l1_loss(lq_preds,  lq[:, 1:target_frame+1, :])
            loss_gvr = F.l1_loss(gvr_preds, gvr[:, 1:target_frame+1, :])
            loss_gp = 0.5 * F.l1_loss(gp_preds, gp[:, 1:target_frame+1, :])
            loss_c = 0.1 * F.l1_loss(c_preds, c[:, 1:target_frame+1, :])

            # update generator
            generator_optimizer.zero_grad()
            loss = loss_lq + loss_gvr + loss_gp + loss_c
            loss.backward()
            loss_sum += loss.item()
            loss_lq_sum += loss_lq.item()
            loss_gvr_sum += loss_gvr.item()
            loss_gp_sum += loss_gp.item()
            loss_c_sum += loss_c.item()
            generator_optimizer.step()

            # adversarial loss
            gv_preds = gp_preds - torch.cat([gp[:, 0:1, :], gp_preds[:, :-1, :]], dim=1)
            root_rel_pos_preds = gp_preds.reshape(*gp_preds.shape[:2], -1, 3) - gpr_preds.reshape(*gpr_preds.shape[:2], 1, 3)
            root_rel_vel_preds = gv_preds.reshape(*gv_preds.shape[:2], -1, 3) - gvr_preds.reshape(*gvr_preds.shape[:2], 1, 3)
            root_rel_pos_preds = root_rel_pos_preds.reshape(*root_rel_pos_preds.shape[:2], -1)
            root_rel_vel_preds = root_rel_vel_preds.reshape(*root_rel_vel_preds.shape[:2], -1)

            fake_input = torch.cat([gvr_preds, root_rel_pos_preds, root_rel_vel_preds], dim=-1)
            fake_input_short = torch.stack([fake_input[:, i:i+2, :].view(fake_input.shape[0], -1) for i in range(9, target_frame - 1)], dim=1)
            fake_input_long = torch.stack([fake_input[:, i:i+10, :].view(fake_input.shape[0], -1) for i in range(1, target_frame - 9)], dim=1)

            gp = gp.reshape(*gp.shape[:2], -1, 3)
            root_rel_pos = gp[:, 1:target_frame+1, :, :] - gp[:, 1:target_frame+1, 0:1, :]
            root_rel_vel = (gp[:, 1:target_frame+1, :, :] - gp[:, 0:target_frame, :, :]) - (gp[:, 1:target_frame+1, 0:1, :] - gp[:, 0:target_frame, 0:1, :])
            root_rel_pos = root_rel_pos.reshape(*root_rel_pos.shape[:2], -1)
            root_rel_vel = root_rel_vel.reshape(*root_rel_vel.shape[:2], -1)
            gp = gp.reshape(*gp.shape[:2], -1)

            real_input = torch.cat([gvr[:, 1:target_frame+1, :], root_rel_pos, root_rel_vel], dim=-1)
            real_input_short = torch.stack([real_input[:, i:i+2, :].view(real_input.shape[0], -1) for i in range(9, target_frame - 1)], dim=1)
            real_input_long = torch.stack([real_input[:, i:i+10, :].view(real_input.shape[0], -1) for i in range(1, target_frame - 9)], dim=1)

            fake_output_short = discriminator_short(fake_input_short.detach())
            fake_output_long = discriminator_long(fake_input_long.detach())

            real_output_short = discriminator_short(real_input_short)
            real_output_long = discriminator_long(real_input_long)

            loss_gen = 0.5 * torch.mean(torch.mean((fake_output_short - 1) ** 2, dim=1) + torch.mean((fake_output_long - 1) ** 2, dim=1))
            loss_disc = 0.5 * torch.mean(torch.mean((real_output_short - 1) ** 2, dim=1)\
                                        + torch.mean((real_output_long - 1) ** 2, dim=1)\
                                        + torch.mean(fake_output_short ** 2, dim=1)\
                                        + torch.mean(fake_output_long ** 2, dim=1))
            
            # update discriminators
            discriminator_optimizer.zero_grad()
            loss_adv = 0.1 * (loss_gen + loss_disc)
            loss_adv.backward()
            loss_sum += loss_adv.item()
            loss_adv_sum += loss_adv.item()
            discriminator_optimizer.step()

        # log
        print(f"Epoch {epoch} loss: {(loss_sum / num_batch):.4f} quat loss: {(loss_lq_sum / num_batch):.4f} vel loss: {(loss_gvr_sum / num_batch):.4f} pos loss: {(loss_gp_sum / num_batch):.4f} contact loss: {(loss_c_sum / num_batch):.4f} adv loss: {(loss_adv_sum / num_batch):.4f}")
        writer.add_scalar("Loss/Total", loss_sum / num_batch, epoch)
        writer.add_scalar("Loss/Quat", loss_lq_sum / num_batch, epoch)
        writer.add_scalar("Loss/RootVel", loss_gvr_sum / num_batch, epoch)
        writer.add_scalar("Loss/Pos", loss_gp_sum / num_batch, epoch)
        writer.add_scalar("Loss/Contact", loss_c_sum / num_batch, epoch)
        writer.add_scalar("Loss/Adversarial", loss_adv_sum / num_batch, epoch)

        # curriculum learning
        if epoch % 20 == 0:
            p_max *= 3
            p_max = min(p_max, max_transition)

    save_dict = {
        "generator": generator.state_dict(),
        "discriminator_short": discriminator_short.state_dict(),
        "discriminator_long": discriminator_long.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
    }
    torch.save(save_dict, "save/rmib.pth")