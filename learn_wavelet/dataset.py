import torch
import numpy as np
import os
import random
random.seed (2023)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class WaveletSamples(torch.utils.data.Dataset):
    def __init__(self,
                 interval: int = 1,
                 data_files_name=None
                ):
        super(WaveletSamples, self).__init__()

        # get file
        self.data_list = self.list_npy_files_recursive(data_files_name)
        # interval
        self.interval = interval

    
        # data length
        self.data_len = len(self.data_list)

    def list_npy_files_recursive(self, directory):
        npy_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".npy"):
                    npy_files.append(os.path.join(root, file))
        return npy_files

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        idx = idx * self.interval
        path = self.data_list[idx]
        gt_np = np.load(path)
        gt_torch = torch.from_numpy(gt_np).to(device)
        return gt_torch
    
    def get_gt(self, iter):
        idx = iter % self.data_len
        path = self.data_list[idx]
        gt_np = np.load(path)
        gt_torch = torch.from_numpy(gt_np).to(device)
        return gt_torch

    def get_name(self, iter):
        idx = iter % self.data_len
        name = self.data_list[idx].split('/')[-1][:-4]
        return name
        