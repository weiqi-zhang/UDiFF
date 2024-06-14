import os
import sys
import torch
import random
import numpy as np
import traceback
from multiprocessing import Pool
from configs import config
from models.module.dwt import DWTForward3d_Laplacian, DWT3d_Laplacian
from fnmatch import fnmatch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
padding_mode = 'zero'

def convert_file(args_list):
    results = []
    dwt_3d_lap = DWT3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=padding_mode).to(device)
    checkpoint = torch.load(r'<PATH_TO_CHECKPOINT>', map_location=device)
    dwt_3d_lap.load_state_dict(checkpoint['dwt_3d_lap'])
    for args in args_list:
        try:
            idx, path, resolution_index, clip_value = args
            assert path.endswith('.npy')
            voxels_np = np.load(path)
            voxels_torch = torch.from_numpy(voxels_np).unsqueeze(0).unsqueeze(0).float().to(device) # [1, 1, 256, 256, 256]
            

            if clip_value is not None:
                voxels_torch = torch.clip(voxels_torch, -clip_value, clip_value)

            low_lap, highs_lap = dwt_3d_lap(voxels_torch)
            if resolution_index == config.max_depth:
                results.append(low_lap[0,0].detach().cpu().numpy()[None, :]) # [46, 46, 46]
            else:
                results.append(highs_lap[resolution_index][0,0].detach().cpu().numpy()[None, :])
        except:
            traceback.print_exc()

        print(f"index {idx} Done!")

    results = np.concatenate(results, axis = 0)

    return results



def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == '__main__':
    category_id = r'deepfashion'
    udf_save_folder = r'<PATH_TO_UDF>'
    npy_save_folder = r'<PATH_TO_SAVE>'
    workers = 1
    resolution_index = config.max_depth
    clip_value = 0.1
    save_new_path =  os.path.join(npy_save_folder, category_id + f'_{clip_value if clip_value is not None else "no_clip"}_{config.wavelet_type}_{resolution_index}_{padding_mode}.npy')

    pattern = "*.npy"
    paths = []
    args = []

    for path, subdirs, files in os.walk(udf_save_folder):
        for name in files:
            if fnmatch(name, pattern):
                paths.append(os.path.join(path, name))


    for idx, path in enumerate(paths):
        args.append((idx, path, resolution_index, clip_value))

    print(f"{len(args)} left to be processed!")
    results = convert_file(args)

    print(results.shape)
    np.save(save_new_path, results)