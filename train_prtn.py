import torch
from torch.utils.tensorboard import SummaryWriter
import random

from data.dataset import MotionDataset
from model.rtn import PhaseRTN

# training parameters
epochs = 100
lr = 1e-4
batch_size = 64

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
train_dset = MotionDataset("D:/data/PFNN", train=True, window=window_size, offset=offset, phase=True, target_fps=target_fps)

if __name__ == "__main__":
    seed = 777
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare training data
    gv_root = train_dset.extract_features("global_vel_root")
    gp_root = train_dset.extract_features("global_pos_root")
    gp = train_dset.extract_features("global_pos")
    gp_root_rel = gp - gp_root.tile(1, 1, gp.shape[-1] // gp_root.shape[-1])
    train_data = torch.cat([gv_root, gp_root_rel[..., 3:]], dim=-1)
    phase_data = train_dset.extract_features("phase").squeeze(-1)
    
    # training settings
    num_batch = train_data.shape[0] // batch_size
    model = PhaseRTN(train_data.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    writer = SummaryWriter("log/PhaseRTN")

    # training loop
    for epoch in range(1, epochs+1):
        loss_avg, loss_recon_avg, loss_phase_avg = 0, 0, 0
        train_idx = torch.randperm(train_data.shape[0])
        for i in range(num_batch):
            print("Progress ", round(100 * i / num_batch, 2), "%", end="\r")
            batch = train_data[train_idx[i*batch_size:(i+1)*batch_size]].to(device)
            phase = phase_data[train_idx[i*batch_size:(i+1)*batch_size]].to(device)
            batch = (batch - batch.mean()) / (batch.std() + 1e-8)

            # network initialization
            model.init_hidden(batch_size, batch[:, 0, :], phase[:, 0])
            model.set_target(batch[:, target_frame, :], phase[:, target_frame])

            # prediction
            preds, deltas = [], []
            for f in range(0, target_frame):
                input_data = batch[:, f, :] if f < past_context or random.random() < teacher_prob else pred
                input_phase = phase[:, f]

                pred, delta_phase = model(input_data, input_phase)

                if f >= past_context:
                    preds.append(pred)
                    deltas.append(delta_phase)
            
            preds = torch.stack(preds, dim=1)
            deltas = torch.stack(deltas, dim=1)

            # compute loss and update
            optimizer.zero_grad()
            
            loss_recon = loss_fn(preds, batch[:, past_context+1:target_frame+1, :]) # reconstruction loss

            delta_phase_gt = phase[:, past_context+1:target_frame+1] - phase[:, past_context:target_frame]
            delta_phase_gt = torch.where(delta_phase_gt < 0, delta_phase_gt + 1, delta_phase_gt)
            loss_phase = loss_fn(deltas, delta_phase_gt) # phase loss

            loss = loss_recon + loss_phase # total loss
            loss.backward()
            loss_avg += loss.item()
            loss_recon_avg += loss_recon.item()
            loss_phase_avg += loss_phase.item()
            optimizer.step()

        # log
        print(f"Epoch {epoch} / {epochs}     Loss: {loss_avg / num_batch}     Recon: {loss_recon_avg / num_batch}     Phase: {loss_phase_avg / num_batch}")
        writer.add_scalar("Loss/Total", loss_avg / num_batch, epoch)
        writer.add_scalar("Loss/Recon", loss_recon_avg / num_batch, epoch)
        writer.add_scalar("Loss/Phase", loss_phase_avg / num_batch, epoch)

    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(save_dict, "save/prtn.pth")