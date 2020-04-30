import time
import os
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models import SinGanOneModel
from train_results import *
from functions import torch2np

class SinGanOneTrainer():
    def __init__(self, model:SinGanOneModel, num_epochs:int, device:torch.device, lambda_grad:float, alpha:float,
                 seed:int=42):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        assert isinstance(model,SinGanOneModel), "model type is not supported"
        assert isinstance(device, torch.device), "Please provide device as torch.device"

        torch.manual_seed(seed)

        self.model       = model
        self.device      = device
        self.lambda_grad = lambda_grad
        self.alpha       = alpha

        self.model.to(device)

    def fit(self, dl:DataLoader, verbose:bool=True, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains a single scale of a SinGanOne model for multiple epochs on a given image/
        :param dl_train: DataLoader for the training.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        epochs_result = []
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        self.model.train(True)

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = self.train_epoch.__name__

        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            for i, batch in enumerate(dl):
                batch_result = self.train_epoch(batch,i)

                pbar.set_description(f'ErrD(r)={batch_result.errD_real:.3f} | '
                                     f'ErrD(f)={batch_result.errD_fake:.3f} | '
                                     f'Penalty={batch_result.gradPenalty:.3f} | '
                                     f'ErrD(total)={batch_result.errD:.3f} | '
                                     f'ErrG={batch_result.errG:.3f} | '
                                     f'RecLoss={batch_result.recLoss:.3f}')
                pbar.update()

                epochs_result.append(batch_result)

                if post_epoch_fn:
                    post_epoch_fn(batch)

            errD_reals = [batch_result.errD_real for batch_result in epochs_result]
            errD_fakes = [batch_result.errD_fake for batch_result in epochs_result]
            penalties  = [batch_result.gradPenalty for batch_result in epochs_result]
            errDs      = [batch_result.errD for batch_result in epochs_result]
            errGs      = [batch_result.errG for batch_result in epochs_result]
            rec_losses = [batch_result.recLoss for batch_result in epochs_result]

            fake_images = [batch_result.fake_image for batch_result in epochs_result
                           if batch_result.fake_image is not None]
            G_z_images  = [batch_result.G_z_image for batch_result in epochs_result
                           if batch_result.G_z_image is not None]
            D_fakes     = [batch_result.D_fake for batch_result in epochs_result
                           if batch_result.D_fake is not None]
            D_reals     = [batch_result.D_real for batch_result in epochs_result
                           if batch_result.D_real is not None]


            avg_errD_real       = sum(errD_reals)   / num_batches
            avg_errD_fake       = sum(errD_fakes)   / num_batches
            avg_grads_penalty   = sum(penalties)    / num_batches
            avg_errD            = sum(errDs)        / num_batches
            avg_errG            = sum(errGs)        / num_batches
            avg_rec_loss        = sum(rec_losses)   / num_batches

            pbar.set_description(f'Avg. ErrD(r)={avg_errD_real:.3f} | '
                                 f'Avg. ErrD(f)={avg_errD_fake:.3f} | '
                                 f'Avg. Penalty={avg_grads_penalty:.3f} | '
                                 f'Avg. ErrD(total)={avg_errD:.3f} | '
                                 f'Avg. ErrG={avg_errG:.3f} | '
                                 f'Avg. RecLoss={avg_rec_loss:.3f}')

        self.save_images(fake_images[-1], G_z_images[-1], D_fakes[-1], D_reals[-1], batch[0])

        fitresult_return = FitResult(errD_reals=errD_reals, errD_fakes=errD_fakes, gradPenalties=penalties, errDs=errDs,
                         errGs=errGs, recLosses=rec_losses, fake_images=fake_images, G_z_images=G_z_images,
                         D_fakes=D_fakes, D_reals=D_reals)
        fitresult_return.save(os.path.join(self.model.model_dir, 'FitResult'))

        return fitresult_return

    def save_images(self, fakes, Gzs, Dfakes, Freals, reals):
        plt.imsave(os.path.join(self.model.model_dir, 'G(z).png'), torch2np(make_grid(fakes)),    vmin=0, vmax=1)
        plt.imsave(os.path.join(self.model.model_dir, 'G(z_rec).png'), torch2np(make_grid(Gzs)), vmin=0, vmax=1)
        plt.imsave(os.path.join(self.model.model_dir, 'D(fake).png'), torch2np(make_grid(Dfakes)))
        plt.imsave(os.path.join(self.model.model_dir, 'D(real).png'), torch2np(make_grid(Freals)))
        plt.imsave(os.path.join(self.model.model_dir, 'real.png'), torch2np(make_grid(reals)))

    def train_epoch(self, batch,i) -> EpochResult:
        real   = batch[0]
        z_rec  = batch[1]
        prev   = batch[2]
        z_prev = batch[3]
        noise  = batch[4]

        # |---------------------------------------------|
        # |(1) Update D network: maximize D(x) + D(G(z))|
        # |---------------------------------------------|
        # |---------------|
        # |train with real|
        # |---------------|
        self.model.netD.zero_grad()
        output = self.model.netD(real).to(self.device)
        D_real_map = output.detach()
        errD_real = -(output.mean())
        errD_real.backward(retain_graph=True)
        # |---------------|
        # |train with fake|
        # |---------------|
        fake = self.model.netG(noise.detach(), prev)
        output = self.model.netD(fake.detach())
        errD_fake = output.mean()
        errD_fake.backward(retain_graph=True)

        gradient_penalty = self.model.gradient_penalty(real, fake)
        gradient_penalty.backward()

        errD = errD_real + errD_fake + gradient_penalty
        self.model.optimizerD.step()

        # |--------------------------------------|
        # |(2) Update G network: maximize D(G(z))|
        # |--------------------------------------|
        self.model.netG.zero_grad()
        output = self.model.netD(fake)
        D_fake_map = output.detach()
        errG = -output.mean()
        errG.backward(retain_graph=True)

        z_rec_out = self.model.netG(z_rec.detach(), z_prev)
        rec_loss = self.alpha * self.model.z_rec_loss(z_rec_out, real)
        rec_loss.backward(retain_graph=True)

        self.model.optimizerG.step()

        self.model.schedulerG.step()
        self.model.schedulerD.step()

        if i%25 == 0:
            return EpochResult(errD_real=-errD_real.item(), errD_fake=errD_fake.item(),
                               gradPenalty=gradient_penalty.detach(), errD=errD.detach(), errG=errG.detach(),
                               recLoss=rec_loss.detach(), fake_image=fake.detach(), G_z_image=z_rec_out.detach(),
                               D_fake=D_fake_map, D_real=D_real_map)
        else:
            return EpochResult(errD_real=-errD_real.item(), errD_fake=errD_fake.item(),
                               gradPenalty=gradient_penalty.detach(), errD=errD.detach(), errG=errG.detach(),
                               recLoss=rec_loss.detach(), fake_image=None, G_z_image=None, D_fake=None, D_real=None)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)