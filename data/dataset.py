import os
import torch
from torch.utils.data import Dataset
import pickle

from . import bvh

def process_pkl(pickle_name, dir_path, phase, target_fps):
    if not os.path.exists(os.path.join(dir_path, "pickle")):
        os.makedirs(os.path.join(dir_path, "pickle"))

    path = os.path.join(dir_path, "pickle", pickle_name)
    if pickle_name in os.listdir(os.path.join(dir_path, "pickle")):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = bvh.get_data_dict(dir_path, phase, target_fps)
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
        phase=False,
        target_fps=30,
    ):
        super(MotionDataset, self).__init__()
        self.window = window
        self.offset = offset
        self.phase = phase

        dir_path = os.path.join(dir_path, "train" if train else "test")


        if self.phase:
            pickle_name = f"processed_fps{target_fps}_phase.pkl"
        else:
            pickle_name = f"processed_fps{target_fps}.pkl"

        print(f"Loading {'train' if train else 'test'} data...")
        self.seq_features = process_pkl(pickle_name, dir_path, phase, target_fps)
        if train == False:
            self.parents, self.bone_offset = self.seq_features[0]["parents"], self.seq_features[0]["offset"]
    
    def extract_features(self, key):
        X = []
        num_windows = []
        print(f"Extracting {key} features...")

        # extract features by feature keys
        for seq in self.seq_features:
            if key in seq:
                feature = seq[key]
                if key == "parents":
                    X.append(torch.from_numpy(feature))
                    continue
            elif key.endswith("_root"):
                feature = seq[key[:-5]][:, 0:1, :]
            elif key.endswith("_noroot"):
                feature = seq[key[:-7]][:, 1:, :]
            else:
                raise Exception(f"Key {key} not found")

            # sliding windows
            idx = 0
            windows = 0
            while idx + self.window < feature.shape[0]:
                X.append(torch.from_numpy(feature[idx:idx+self.window]))
                idx += self.offset
                windows += 1
            
            num_windows.append(windows)
        
        # convert to tensor
        X = torch.stack(X, dim=0).float()
        # print(X.shape)
        X = X.view(X.shape[0], X.shape[1], -1).float()

        # check whether the parents are same
        if key == "parents":
            assert torch.all(torch.eq(X[0], X[1:]))
            X = X[0].view(-1).long()

        if not hasattr(self, "num_windows"):
            self.num_windows = num_windows

        return X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]