import math
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))

class WDiscriminator(nn.Module):
    def __init__(self, nfc:int, min_nfc:int, ker_size:int, padd_size:int, num_layer:int):
        """
        SinGan Discriminator
        :param nfc:
        :param min_nfc:
        :param ker_size:
        :param padd_size:
        :param num_layer:
        """
        super(WDiscriminator, self).__init__()
        N = int(nfc)
        self.head = ConvBlock(3, N, ker_size, padd_size, 1)
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), ker_size, padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(max(N, min_nfc), 1, kernel_size=ker_size, stride=1, padding=padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, nfc:int, min_nfc:int, ker_size:int, padd_size:int, num_layer:int):
        """
        SinGan Generator
        :param nfc:
        :param min_nfc:
        :param ker_size:
        :param padd_size:
        :param num_layer:
        """
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        N = nfc
        self.head = ConvBlock(3, N, ker_size, padd_size,1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), ker_size, padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, min_nfc), 3, kernel_size=ker_size, stride=1, padding=padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

class SinGanOneModel(nn.Module):
    def __init__(self, scale_num, opt):
        super(SinGanOneModel, self).__init__()

        self.device = opt.device
        self.model_dir = os.path.join(opt.model_dir, str(scale_num),)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            shutil.rmtree(self.model_dir)
            os.makedirs(self.model_dir)

        self.nfc     = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        self.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        self.lambda_grad  = opt.lambda_grad

        self.set_generator(opt)
        self.set_discriminator(opt)
        self.set_z_rec_loss()

        self.set_optimizerG(opt)
        self.set_optimizerD(opt)

    def set_generator(self, opt):
        self.netG = GeneratorConcatSkip2CleanAdd(self.nfc, self.min_nfc, opt.ker_size,
                                                 opt.padd_size, opt.num_layer).to(opt.device).type(torch.float64)
        self.initG()

    def set_discriminator(self, opt):
        self.netD = WDiscriminator(self.nfc, self.min_nfc, opt.ker_size,
                                   opt.padd_size, opt.num_layer).to(self.device).type(torch.float64)
        self.initD()

    def set_optimizerG(self, opt):
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        self.schedulerG = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizerG, milestones=[1600], gamma=opt.gamma)

    def set_optimizerD(self, opt):
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.schedulerD = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizerD, milestones=[1600], gamma=opt.gamma)

    def initG(self):
        def __weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('Norm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        self.netG.apply(__weights_init)

    def initD(self):
        def __weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('Norm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        self.netD.apply(__weights_init)

    def eval(self):
        for p in self.netG.parameters():
            # probably doesn't needed since self.netG.eval() takes care of disabling the grads
            p.requires_grad_(False)
        self.netG.eval() # equivalent to self.netG.train(false)
        for p in self.netD.parameters():
            p.requires_grad_(False)
        self.netD.eval() # equivalent to self.netD.train(false)

    def set_z_rec_loss(self):
        self.z_rec_loss = nn.MSELoss()

    def train(self, mode=True):
        for p in self.netG.parameters():
            # probably doesn't needed since self.netG.train(True) takes care of activating the grads
            p.requires_grad_(True)
        self.netG.train(True)
        for p in self.netD.parameters():
            p.requires_grad_(True)
        self.netD.train(True)

    def save(self):
        torch.save(self.netG, os.path.join(self.model_dir, 'G'))
        torch.save(self.netD, os.path.join(self.model_dir, 'D'))

    def load(self):
        self.netG.load_state_dict(torch.load(os.path.join(self.model_dir, 'G'), map_location=self.device))
        self.netD.load_state_dict(torch.load(os.path.join(self.model_dir, 'D'), map_location=self.device))

    def to(self, *args, **kwargs):

        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        self.netG.to(device)
        self.netD.to(device)

    def gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(1, 1, dtype=torch.float64)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.to(self.device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        # LAMBDA = 1
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_grad
        return gradient_penalty

    def __str__(self):
        ret_str  = '|' + '-'*20 + 'Generator' + '-'*20 + '|' + '\n'
        ret_str += '|' + '-'*49 + '|' + '\n'
        ret_str += str(self.netG) + '\n'
        ret_str += '|' + '-' * 20 + 'Discriminator' + '-' * 20 + '|' + '\n'
        ret_str += '|' + '-' * 53 + '|' + '\n'
        ret_str += str(self.netD) + '\n'

        return ret_str
