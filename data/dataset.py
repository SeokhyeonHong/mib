import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from . import bvh
from . import utils

def process_pkl(pickle_name, dir_path, window, offset, phase, sampling_interval):
    if not os.path.exists(os.path.join(dir_path, "pickle")):
        os.makedirs(os.path.join(dir_path, "pickle"))

    path = os.path.join(dir_path, "pickle", pickle_name)
    if pickle_name in os.listdir(os.path.join(dir_path, "pickle")):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = bvh.get_raw_data(dir_path, window, offset, phase, sampling_interval)
        with open(path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return data

class MotionDataset(Dataset):
    def __init__(
        self,
        dir_path,
        train,
        window=50,
        offset=20,
        keys=["local_quat", "global_pos", "global_vel"],
        phase=False,
        sampling_interval=1,
    ):
        super(MotionDataset, self).__init__()

        self.dir_path = os.path.join(dir_path, "train" if train else "test")
        self.window = window
        self.offset = offset
        self.phase = phase
        self.sampling_interval = sampling_interval

        if self.phase:
            pickle_name = f"window{self.window}_offset{self.offset}_interval{self.sampling_interval}_phase.pkl"
        else:
            pickle_name = f"window{self.window}_offset{self.offset}_interval{self.sampling_interval}.pkl"
        self.raw_data = process_pkl(pickle_name, self.dir_path, self.window, self.offset, self.phase, self.sampling_interval)
        
        if self.phase:
            self.phase_data = torch.from_numpy(self.raw_data["phase"]).float()

        func_dict = {
            "offset": self.get_offset,
            "global_pos": self.get_global_pos,
            "global_root_pos": self.get_global_root_pos,
            "global_vel": self.get_global_vel,
            "global_root_vel": self.get_global_root_vel,
            "local_pos": self.get_local_pos,
            "local_quat": self.get_local_quat,
            "local_vel": self.get_local_vel,
            "root_rel_pos": self.get_root_rel_pos,
        }

        self.data = []
        for key in keys:
            data = torch.from_numpy(func_dict[key]()).float()
            self.data.append(data.view(*data.shape[:-2], -1))
        self.data = torch.cat(self.data, dim=-1) # [#windows x window size x #joints*#channels]
        print(f"Dataset shape: {self.data.shape}")

    def get_offset(self):
        offset = self.raw_data["offset"]
        return offset[0:1, 0:1, 1:, :]

    def get_global_pos(self):
        return self.raw_data["global_pos"]

    def get_global_root_pos(self):
        return self.raw_data["global_pos"][..., 0:1, :]

    def get_global_vel(self):
        return self.raw_data["global_vel"]

    def get_global_root_vel(self):
        return self.raw_data["global_vel"][..., 0:1, :]

    def get_local_quat(self):
        return self.raw_data["local_quat"]
        
    def get_root_quat(self):
        return self.raw_data["local_quat"][..., 0:1, :]

    def get_local_pos(self):
        global_pos = self.get_global_pos()
        root_pos = self.get_global_root_pos()
        root_rot = self.get_root_quat()
        local_pos = utils.quat_mul_vec(utils.quat_inv(root_rot), (global_pos - root_pos))
        return local_pos

    def get_local_vel(self):
        global_vel = self.get_global_vel()
        root_vel = self.get_global_root_vel()
        root_rot = self.get_root_quat()
        local_vel = utils.quat_mul_vec(utils.quat_inv(root_rot), (global_vel - root_vel))
        return local_vel

    def get_root_rel_pos(self):
        res = self.get_global_pos() - self.get_global_root_pos()
        return res[..., 1:, :]

    def mean(self):
        return self.data.mean(dim=0)

    def std(self):
        return self.data.std(dim=0) + 1e-8

    def parents(self):
        return self.raw_data["parents"]
    
    def dof(self):
        return self.data.shape[-1]

    def find_data(self, key):
        return self.raw_data[key]
    
    def transpose(self, dim0, dim1):
        self.data = self.data.transpose(dim0, dim1).contiguous()
        print(f"Dataset shape transposed: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.phase:
            return self.data[index], self.phase_data[index, 1:]
        else:
            return self.data[index]