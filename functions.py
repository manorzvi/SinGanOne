import numpy as np
import math
from skimage import io as skimgio
import torch
import torch.nn as nn

from imresize import imresize_in

def np2torch(x, device, type=torch.FloatTensor):
    x = x.transpose((2, 0, 1))/255
    x = torch.from_numpy(x)
    x.type(type)
    x.to(device)
    x = norm(x)
    return x

def norm(x):
    out = (x -0.5) * 2
    return out.clamp(-1, 1)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def read_image(img_path, device):
    x = skimgio.imread(img_path)
    x = np2torch(x, device)
    return x

def torch2np(x, type=np.uint8):
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(type)
    return x

def imresize(img,scale_factor, device):
    img = torch2np(img)
    img = imresize_in(img, scale_factor=scale_factor)
    img = np2torch(img, device)
    return img

def adjust_scales2image(real_, opt):
    min_real_ = min(real_.shape[1], real_.shape[2])
    max_real_ = max(real_.shape[1], real_.shape[2])
    opt.num_scales = math.ceil(math.log(opt.min_size             / min_real_, opt.scale_factor)) + 1
    scale2stop = math.ceil(math.log(min(opt.max_size, max_real_) / max_real_, opt.scale_factor))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max_real_ ,1)

    real = imresize(real_, opt.scale1, opt.device)

    min_real = min(real.shape[1], real.shape[2])
    max_real = max(real.shape[1], real.shape[2])
    opt.scale_factor = math.pow(opt.min_size/min_real, 1/opt.stop_scale)

    return real

def create_reals_pyramid(real,reals,opt):
    for i in range(opt.stop_scale+1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt.device)
        reals.append(curr_real)
    return reals

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1, dtype=torch.float64):
    if type == 'gaussian':
        noise = torch.randn(size, device=device, dtype=dtype)
        noise = noise[None,:,:,:]
        noise = upsampling(noise, size[1], size[2])
        noise = noise.squeeze(0)
    if type =='gaussian_mixture':
        noise1 = torch.randn(size, device=device, dtype=dtype)+5
        noise2 = torch.randn(size, device=device, dtype=dtype)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(size, device=device, dtype=dtype)
    return noise