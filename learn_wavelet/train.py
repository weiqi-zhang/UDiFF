import sys
import torch
import numpy as np
from learn_wavelet.dataset import WaveletSamples
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
import argparse
from learn_wavelet.utils import get_root_logger, print_log
import time
from models.module.dwt import DWTForward3d_Laplacian, DWTInverse3d_Laplacian, DWT3d_Laplacian
from models.network import SparseComposer
from configs import config
from torch.utils.tensorboard import SummaryWriter
import mcubes
import random
random.seed (2024)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
padding_mode = 'zero'

class Runner:

    def __init__(self, is_continue=False) -> None:
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        
        self.base_exp_dir = os.path.join('./learn_wavelet/outs', self.timestamp)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.train_dataset = WaveletSamples(data_files_name=r'<PATH_TO_DEEPFASHION>')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=25)

        self.loss_fn = nn.MSELoss(reduction='mean')
        self.maxiter = 500000
        self.iter_step = 0
        self.learning_rate = 5e-4
        self.warm_up_end = 1000
        self.clip_value = 0.1
        self.report_freq = 1798
        self.save_freq = 500
        self.device = device
        self.threshold = 0.01

        self.resolution = 256
        self.b_max_np = np.array([-0.5, -0.5, -0.5])
        self.b_min_np = np.array([0.5, 0.5, 0.5])


        self.dwt_3d_lap = DWT3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
        self.dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
        self.composer_parms = self.dwt_inverse_3d_lap
        self.dwt_sparse_composer = SparseComposer(input_shape=[config.resolution, config.resolution, config.resolution],
                                                J=config.max_depth,
                                                wave=config.wavelet, mode=config.padding_mode,
                                                inverse_dwt_module=self.composer_parms).to(device)


        params_to_train = []
        params_to_train += list(self.dwt_3d_lap.parameters())

        self.optimizer = torch.optim.Adam(params=params_to_train, lr=self.learning_rate)
        self.mc_threshold=[0, 0.005, 0.01]

        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                     model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            print('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr


    def save_checkpoint(self):
        checkpoint = {
            'dwt__3d_lap': self.dwt_3d_lap.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.dwt_3d_lap.load_state_dict(checkpoint['dwt__3d_lap'])
        self.iter_step = checkpoint['iter_step']
    
    def train(self):
        self.writer = SummaryWriter(os.path.join(self.base_exp_dir, 'logs'))
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{self.timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        res_step = self.maxiter - self.iter_step
        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i)

            gt_torch = self.train_dataset.get_gt(iter_i).unsqueeze(0).unsqueeze(0)
            gt_torch = torch.clip(gt_torch, -self.clip_value, self.clip_value)
            gt_numpy = gt_torch.clone().detach().cpu().numpy()
            index = np.where(gt_numpy <= self.threshold)

            low_lap, highs_lap = self.dwt_3d_lap(gt_torch)
            low_samples = low_lap[0,0].unsqueeze(0).unsqueeze(0)
            highs_samples = [torch.zeros(tuple([1, 1] + self.dwt_sparse_composer.shape_list[i]), device=device) for i in
                                range(config.max_depth)]

            highs_samples[2] = highs_lap[2][0,0].unsqueeze(0).unsqueeze(0)
            voxels_pred = self.dwt_3d_lap.inverse((low_samples, highs_samples))
            
            loss = 0
            loss_surface = self.loss_fn(gt_torch[index], voxels_pred[index])
            
            loss = loss + loss_surface

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("train_loss", loss.item(), global_step=self.iter_step)
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=self.iter_step)
            
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} loss = {}'.format(self.iter_step, loss), logger=logger)
            
            if self.iter_step % self.save_freq == 0 and self.iter_step != 0:
                self.save_checkpoint()
            
            self.iter_step += 1

    def validate_mesh(self):
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        with torch.no_grad():
            idx = random.randint(0, self.train_dataset.data_len - 1)
            name = self.train_dataset.get_name(idx)
            gt_torch = self.train_dataset.get_gt(idx).unsqueeze(0).unsqueeze(0)
            gt_torch = torch.clip(gt_torch, -self.clip_value, self.clip_value)
            gt_numpy = gt_torch.clone().detach().cpu().numpy()
            index = np.where(gt_numpy <= self.threshold)
            low_lap, highs_lap = self.dwt_3d_lap(gt_torch)
            low_samples = low_lap[0,0].unsqueeze(0).unsqueeze(0)
            highs_samples = [torch.zeros(tuple([1, 1] + self.dwt_sparse_composer.shape_list[i]), device=device) for i in
                                range(config.max_depth)]
            highs_samples[2] = highs_lap[2][0,0].unsqueeze(0).unsqueeze(0)
            voxels_pred = self.dwt_3d_lap.inverse((low_samples, highs_samples))

            raw_dwt_forward_3d_lap = DWTForward3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=padding_mode).to(device)
            raw_dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
            raw_low_lap, raw_highs_lap = raw_dwt_forward_3d_lap(gt_torch)
            raw_low_samples = raw_low_lap[0,0].unsqueeze(0).unsqueeze(0)
            raw_highs_samples = [torch.zeros(tuple([1, 1] + self.dwt_sparse_composer.shape_list[i]), device=device) for i in
                                range(config.max_depth)]
            raw_highs_samples[2] = raw_highs_lap[2][0,0].unsqueeze(0).unsqueeze(0)
            voxels_raw = raw_dwt_inverse_3d_lap((raw_low_samples, raw_highs_samples))

            loss_1 = self.loss_fn(gt_torch[index], voxels_pred[index])
            loss_2 = self.loss_fn(gt_torch[index], voxels_raw[index])

            print(f'pred loss:  {loss_1}')
            print(f'raw loss:  {loss_2}')

            udf_gt = gt_torch.squeeze(0).squeeze(0).clone().detach().cpu().numpy()
            index = np.where(udf_gt <= self.threshold)
            udf_pred = voxels_pred.squeeze(0).squeeze(0).clone().detach().cpu().numpy()
            udf_raw = voxels_raw.squeeze(0).squeeze(0).clone().detach().cpu().numpy()
            pred_np_path = os.path.join(self.base_exp_dir, 'outputs', f'{name}-pred-mc.npy')
            np.save(pred_np_path, udf_pred)
            raw_np_path = os.path.join(self.base_exp_dir, 'outputs', f'{name}-raw-mc.npy')
            np.save(raw_np_path, udf_raw)
            print(len(np.where(udf_pred[index] < 0)[0]))
            print(len(np.where(udf_raw[index] < 0)[0]))

            print(name)


            for threhold in self.mc_threshold:
                vertices, traingles = mcubes.marching_cubes(udf_gt, threhold)
                vertices = vertices / (self.resolution - 1.0) * (self.b_max_np - self.b_min_np)[None, :] + self.b_min_np[None, :]
                gt_path = os.path.join(self.base_exp_dir, 'outputs', f'{name}-gt-mc-{threhold}.obj')
                mcubes.export_obj(vertices, traingles, gt_path)

                vertices, traingles = mcubes.marching_cubes(udf_pred, threhold)
                vertices = vertices / (self.resolution - 1.0) * (self.b_max_np - self.b_min_np)[None, :] + self.b_min_np[None, :]
                pred_path = os.path.join(self.base_exp_dir, 'outputs', f'{name}-pred-mc-{threhold}.obj')
                mcubes.export_obj(vertices, traingles, pred_path)

                vertices, traingles = mcubes.marching_cubes(udf_raw, threhold)
                vertices = vertices / (self.resolution - 1.0) * (self.b_max_np - self.b_min_np)[None, :] + self.b_min_np[None, :]
                raw_path = os.path.join(self.base_exp_dir, 'outputs', f'{name}-raw-mc-{threhold}.obj')
                mcubes.export_obj(vertices, traingles, raw_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()

    runner = Runner(is_continue=args.is_continue)
    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validata_mesh':
        runner.validate_mesh()
    