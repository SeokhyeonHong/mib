import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from random import random
from tqdm import tqdm

from data.dataset import MotionDataset
from model.rtn import RTN

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
motion = MotionDataset("D:/data/PFNN", train=True, window=window_size, offset=offset, keys=["global_root_vel", "root_rel_pos"], phase=False, sampling_interval=4)
motion_loader = DataLoader(motion, batch_size=batch_size, shuffle=True, drop_last=True)
mean, std = motion.mean(), motion.std()
dof = motion.dof()

if __name__ == "__main__":
    model = RTN(dof).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    writer = SummaryWriter("log/RTN")

    for epoch in tqdm(range(1, epochs+1), desc="Epoch"):
        loss_avg = 0
        for idx, data in enumerate(tqdm(motion_loader, desc="Batch")):
            data = (data - mean) / std
            data = data.to(device)

            target = data[:, target_frame:target_frame+1, :]

            # network initialization
            model.init_hidden(batch_size, data[:, 0:1, :])
            model.set_target(target)

            # prediction
            preds = []
            for f in range(0, target_frame):
                input = data[:, f:f+1, :] if f < past_context or random() < teacher_prob else pred
                pred = model(input)
                preds.append(pred)

            preds = torch.cat(preds, dim=1)

            # compute loss and update
            optimizer.zero_grad()
            loss = F.l1_loss(preds, data[:, 1:target_frame+1, :])
            loss.backward()
            loss_avg += loss.item()
            optimizer.step()

        print(f"Epoch {epoch} loss: {loss_avg / len(motion_loader)}")
        writer.add_scalar("loss", loss_avg / len(motion_loader), epoch)

    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(save_dict, "save/rtn.pth")