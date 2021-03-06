import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from functions import read_image, create_reals_pyramid, adjust_scales2image, torch2np, \
    generate_noise
from exceptions import SinGanOne_PathNotFound, SinGanOne_NonTorchDevice
from config import get_arguments, post_config

class SinGanOne_Dataset(Dataset):
    def __init__(self, opt):
        super(SinGanOne_Dataset, self).__init__()

        if not isinstance(opt.device,torch.device):
            raise SinGanOne_NonTorchDevice(opt.device)
        self.device = opt.device

        self.real_path = os.path.join(opt.reals_dir,opt.real_name)
        if not os.path.exists(self.real_path):
            raise SinGanOne_PathNotFound(self.real_path)

        self.real_original = read_image(self.real_path, opt.device)
        self.real_scaled   = adjust_scales2image(self.real_original,opt)
        self.reals         = []
        self.reals         = create_reals_pyramid(self.real_scaled, self.reals, opt)

        self.m_noise = nn.ZeroPad2d(int(opt.pad_noise))
        self.m_image = nn.ZeroPad2d(int(opt.pad_image))
        self.len = opt.niter

    def set_z_rec(self):
        """
        Z Reconstruction fixed during training.
        """
        pass

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        pass

class SinGanOne_Dataset_ScaleZero(SinGanOne_Dataset):
    def __init__(self, opt):
        super(SinGanOne_Dataset_ScaleZero, self).__init__(opt)

        self.curr_scale = 0
        self.real = self.reals[0]

        self.set_z_rec()

    def set_z_rec(self):
        """
        Z Reconstruction fixed during training.
        """
        self.z_rec = generate_noise((1, self.reals[self.curr_scale].shape[1], self.reals[self.curr_scale].shape[2]),
                                    device=self.device, dtype=torch.float64)
        self.z_rec = self.m_noise(self.z_rec.expand(3, self.reals[self.curr_scale].shape[1],
                                                    self.reals[self.curr_scale].shape[2]))

    def __getitem__(self, item):
        real  = self.real
        z_rec = self.z_rec

        if item == 0:
            self.prev   = torch.zeros([3, real.shape[1],real.shape[2]], device=self.device, dtype=torch.float64)
            self.prev   = self.m_image(self.prev)
            self.z_prev = torch.zeros([3, real.shape[1], real.shape[2]], device=self.device, dtype=torch.float64)
            self.z_prev = self.m_noise(self.z_prev)
        self.noise = generate_noise([1, real.shape[1],real.shape[2]], device=self.device)
        self.noise = self.m_noise(self.noise.expand(3, real.shape[1],real.shape[2]))
        ret = (real, z_rec, self.prev, self.z_prev, self.noise)

        return ret