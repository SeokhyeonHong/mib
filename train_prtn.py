import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from random import random
from tqdm import tqdm

from data.dataset import MotionDataset
from model.rtn import PhaseRTN

# training parameters
epochs = 100
lr = 1e-4
batch_size = 32

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
motion = MotionDataset("D:/data/PFNN", train=True, window=window_size, offset=offset, keys=["global_root_vel", "root_rel_pos"], phase=True, sampling_interval=4)
motion_loader = DataLoader(motion, batch_size=batch_size, shuffle=True, drop_last=True)
mean, std = motion.mean(), motion.std()
dof = motion.dof()

if __name__ == "__main__":
    model = PhaseRTN(dof).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    writer = SummaryWriter("log/PhaseRTN")

    for epoch in tqdm(range(1, epochs+1), desc="Epoch"):
        loss_avg = 0
        loss_recon_avg, loss_phase_avg = 0, 0
        for idx, (data, phase) in enumerate(tqdm(motion_loader, desc="Batch")):
            data = (data - mean) / std
            data = data.to(device)
            phase = phase.to(device)

            # network initialization
            model.init_hidden(batch_size, data[:, 0:1, :], phase[:, 0:1])
            model.set_target(data[:, target_frame:target_frame+1, :], phase[:, target_frame:target_frame+1])

            # prediction
            preds, deltas = [], []
            for f in range(0, target_frame):
                rand = random()
                input_data = data[:, f:f+1, :] if f < past_context or rand < teacher_prob else pred
                input_phase = phase[:, f:f+1]

                pred, delta_phase = model(input_data, input_phase)

                preds.append(pred)
                deltas.append(delta_phase)
            
            preds = torch.cat(preds, dim=1)
            deltas = torch.cat(deltas, dim=1)

            # compute loss and update
            optimizer.zero_grad()
            
            loss_recon = F.mse_loss(preds, data[:, 1:target_frame+1, :]) # reconstruction loss

            delta_phase_gt = phase[:, 1:target_frame+1] - phase[:, 0:target_frame]
            delta_phase_gt = torch.where(delta_phase_gt < 0, delta_phase_gt + 1, delta_phase_gt)
            loss_phase = F.mse_loss(deltas, delta_phase_gt) # phase loss

            loss = loss_recon + loss_phase # total loss
            loss.backward()
            loss_avg += loss.item()
            loss_recon_avg += loss_recon.item()
            loss_phase_avg += loss_phase.item()
            optimizer.step()

        print(f"Epoch {epoch} / {epochs} Loss: {loss_avg / len(motion_loader)} Recon: {loss_recon_avg / len(motion_loader)} Phase: {loss_phase_avg / len(motion_loader)}")
        writer.add_scalar("Loss/Total", loss_avg / len(motion_loader), epoch)
        writer.add_scalar("Loss/Recon", loss_recon_avg / len(motion_loader), epoch)
        writer.add_scalar("Loss/Phase", loss_phase_avg / len(motion_loader), epoch)

    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(save_dict, "save/prtn.pth")