from typing import NamedTuple, List
import pickle
import torch

class EpochResult(NamedTuple):

    errD_real   : float
    errD_fake   : float
    gradPenalty : float
    errD        : float
    errG        :float
    recLoss     : float

    fake_image  : torch.Tensor
    G_z_image   : torch.Tensor
    D_fake      : torch.Tensor
    D_real      : torch.Tensor

class FitResult(NamedTuple):

    errD_reals      : List[float]
    errD_fakes      : List[float]
    gradPenalties   : List[float]
    errDs           : List[float]
    errGs           : List[float]
    recLosses       : List[float]

    fake_images : List[torch.Tensor]
    G_z_images  : List[torch.Tensor]
    D_fakes     : List[torch.Tensor]
    D_reals     : List[torch.Tensor]

    def save(self, fname):
        with open(fname+'.pkl', "wb") as fp:  # Pickling
            pickle.dump(self, fp)
    def load(self,fname):
        with open(fname+'.pkl', "rb") as fp:  # Unpickling
            return pickle.load(fp)