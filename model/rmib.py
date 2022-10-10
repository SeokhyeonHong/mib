import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP, PhaseMLP

class PLU(nn.Module):
    def __init__(self, alpha=0.1, c=1.0):
        super(PLU, self).__init__()
        self.alpha = alpha
        self.c = c
    
    def forward(self, x):
        return torch.max(self.alpha * (x + self.c) - self.c, torch.min(self.alpha * (x - self.c) + self.c, x))

class RMIB(nn.Module):
    def __init__(self,
        dof,
        input_dims,
        hidden_dims={
            "state_encoder": [512],
            "offset_encoder": [512],
            "target_encoder": [512],
            "lstm": 512,
            "decoder": [512, 256],
        },
        output_dim=256,
        device="cpu"
    ):
        super(RMIB, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = device

        self.state_encoder = MLP(input_dims["state_encoder"], hidden_dims["state_encoder"], output_dim, activation=PLU, activation_at_last=True)
        self.offset_encoder = MLP(input_dims["offset_encoder"], hidden_dims["offset_encoder"], output_dim, activation=PLU, activation_at_last=True)
        self.target_encoder = MLP(input_dims["target_encoder"], hidden_dims["target_encoder"], output_dim, activation=PLU, activation_at_last=True)
        self.lstm = nn.LSTM(input_size=output_dim * 3, hidden_size=hidden_dims["lstm"], num_layers=1, batch_first=True)

        self.decoder = MLP(input_dim=hidden_dims["lstm"], hidden_dims=hidden_dims["decoder"], output_dim=dof, activation=PLU)

        self.tta_embedding = TimeToArrival(dim=output_dim, max_len=40)
    
    def set_target(self, target):
        self.target_lq, self.target_gvr = target
        self.h_target = self.target_encoder(self.target_lq)

    def init_hidden(self, batch_size):
        self.h = torch.zeros(1, batch_size, self.hidden_dims["lstm"], device=self.device)
        self.c = torch.zeros(1, batch_size, self.hidden_dims["lstm"], device=self.device)
    
    def sample_z_target(self, size, tta):
        z_target = torch.normal(mean=0, std=0.5, size=size, device=self.device)
        if tta >= 30:
            return z_target
        elif tta >= 5:
            return z_target * (tta - 5) / 25
        return torch.zeros_like(z_target, device=self.device)

    def forward(self, x, tta):
        lq, gvr, c = x
        h_state = self.state_encoder(torch.cat([lq, gvr, c], dim=-1))
        h_offset = self.offset_encoder(torch.cat([self.target_lq - lq, self.target_gvr - gvr], dim=-1))
        z_target = self.sample_z_target(size=(h_offset.shape[0], self.h_target.shape[1] + h_offset.shape[1]), tta=tta)
        h_in = torch.cat([self.tta_embedding(h_state, tta), z_target + torch.cat([self.tta_embedding(h_offset, tta), self.tta_embedding(self.h_target, tta)], dim=-1)], dim=-1).unsqueeze(1)
        h_out, (self.h, self.c) = self.lstm(h_in, (self.h, self.c))
        h_out = self.decoder(h_out.squeeze(1))
        lq_out, gvr_out, c_out = torch.split(h_out, (lq.shape[-1], gvr.shape[-1], c.shape[-1]), dim=-1)
        return lq_out + lq, gvr_out + gvr, torch.sigmoid(c_out)

class PhaseRMIB(nn.Module):
    def __init__(self,
        dof,
        input_dims,
        hidden_dims={
            "state_encoder": [512],
            "offset_encoder": [512],
            "target_encoder": [512],
            "lstm": 512,
            "decoder": [512, 256],
        },
        output_dim=256,
        device="cpu"
    ):
        super(PhaseRMIB, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = device

        self.state_encoder = PhaseMLP(input_dims["state_encoder"], hidden_dims["state_encoder"], output_dim, activation=PLU, activation_at_last=True)
        self.offset_encoder = PhaseMLP(input_dims["offset_encoder"], hidden_dims["offset_encoder"], output_dim, activation=PLU, activation_at_last=True)
        self.target_encoder = PhaseMLP(input_dims["target_encoder"], hidden_dims["target_encoder"], output_dim, activation=PLU, activation_at_last=True)
        self.lstm = nn.LSTM(input_size=output_dim * 3, hidden_size=hidden_dims["lstm"], num_layers=1, batch_first=True)

        self.decoder = PhaseMLP(input_dim=hidden_dims["lstm"], hidden_dims=hidden_dims["decoder"], output_dim=dof+1, activation=PLU)

        self.tta_embedding = TimeToArrival(dim=output_dim, max_len=40)
    
    def set_target(self, target, phase):
        self.target_lq, self.target_gvr = target
        self.h_target = self.target_encoder(self.target_lq, phase)

    def init_hidden(self, batch_size):
        self.h = torch.zeros(1, batch_size, self.hidden_dims["lstm"], device=self.device)
        self.c = torch.zeros(1, batch_size, self.hidden_dims["lstm"], device=self.device)
    
    def sample_z_target(self, size, tta):
        z_target = torch.normal(mean=0, std=0.5, size=size, device=self.device)
        if tta >= 30:
            return z_target
        elif tta >= 5:
            return z_target * (tta - 5) / 25
        return torch.zeros_like(z_target, device=self.device)

    def forward(self, x, phase, tta):
        lq, gvr, c = x
        h_state = self.state_encoder(torch.cat([lq, gvr, c], dim=-1), phase)
        h_offset = self.offset_encoder(torch.cat([self.target_lq - lq, self.target_gvr - gvr], dim=-1), phase)
        z_target = self.sample_z_target(size=(h_offset.shape[0], self.h_target.shape[1] + h_offset.shape[1]), tta=tta)
        h_in = torch.cat([self.tta_embedding(h_state, tta), z_target + torch.cat([self.tta_embedding(h_offset, tta), self.tta_embedding(self.h_target, tta)], dim=-1)], dim=-1).unsqueeze(1)
        h_out, (self.h, self.c) = self.lstm(h_in, (self.h, self.c))
        h_out = self.decoder(h_out.squeeze(1), phase)
        lq_out, gvr_out, c_out, delta_phase_out = torch.split(h_out, (lq.shape[-1], gvr.shape[-1], c.shape[-1], 1), dim=-1)
        return lq_out + lq, gvr_out + gvr, torch.sigmoid(c_out), delta_phase_out.squeeze(-1)

class TimeToArrival(nn.Module):
    def __init__(self, dim, max_len):
        super(TimeToArrival, self).__init__()

        pos = torch.arange(0, max_len, step=1).unsqueeze(1)
        div_term = 1.0 / torch.pow(10000, torch.arange(0, dim, step=2) / dim)

        embedding = torch.empty((max_len, dim))
        embedding[:, 0::2] = torch.sin(pos * div_term)
        embedding[:, 1::2] = torch.cos(pos * div_term)
        embedding[-5:] = embedding[-5]
        self.embedding = nn.Parameter(embedding, requires_grad=False)

    def forward(self, x, tta):
        return x + self.embedding[tta - 1]

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=1):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.encoder = MLP(input_dim, hidden_dims, output_dim, activation=nn.ReLU(), activation_at_last=False)

    def forward(self, x):
        return self.encoder(x).squeeze(-1)