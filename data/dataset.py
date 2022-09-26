import os
import torch
from torch.utils.data import Dataset
import pickle

from . import bvh
from . import utils

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
    
def extract_features(seq_features, feature_keys, window, offset):
    X = []
    for key in feature_keys:
        features = []
        print(f"Extracting {key} features...")
        for seq in seq_features:
            if key in seq:
                feature = seq[key]
            elif key.endswith("_root"):
                feature = seq[key[:-5]][:, 0:1, :]
            elif key.endswith("_noroot"):
                feature = seq[key[:-7]][:, 1:, :]
            else:
                raise Exception(f"Key {key} not found")

            idx = 0
            while idx + window < feature.shape[0]:
                features.append(torch.from_numpy(feature[idx:idx+window]))
                print(torch.from_numpy(feature[idx:idx+window]))
                idx += offset
        features = torch.stack(features, dim=0)
        X.append(features.view(*features.shape[:-2], -1))
    X = torch.cat(X, dim=-1).float()
    return X


class MotionDataset(Dataset):
    def __init__(
        self,
        dir_path,
        train,
        window=50,
        offset=20,
        keys=["local_quat"],
        phase=False,
        target_fps=30,
    ):
        super(MotionDataset, self).__init__()

        dir_path = os.path.join(dir_path, "train" if train else "test")
        self.phase = phase

        if self.phase:
            pickle_name = f"processed_fps{target_fps}_phase.pkl"
        else:
            pickle_name = f"processed_fps{target_fps}.pkl"

        print("Loading data...")
        seq_features = process_pkl(pickle_name, dir_path, phase, target_fps)
        self.data, self.num_windows = extract_features(seq_features, keys, window, offset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]