import os
import random
import argparse
import torch

def get_arguments():
    parser = argparse.ArgumentParser()
    # workspace:
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--plotting',   default=True, help='Plot multiple figures for debugging')

    # load, input, save configurations:
    parser.add_argument('--netG',           default='',                 help="Path to netG (to continue training)")
    parser.add_argument('--netD',           default='',                 help="Path to netD (to continue training)")
    parser.add_argument('--seed',           default=42,                 help='Manual seed')
    parser.add_argument('--top_res_dir',    default='Output',           help='Generation output folder')
    parser.add_argument('--top_models_dir', default='TrainedModels',    help='Trained Models output folder')

    # networks hyper parameters:
    parser.add_argument('--nfc',        type=int,                           default=32)
    parser.add_argument('--min_nfc',    type=int,                           default=32)
    parser.add_argument('--ker_size',   type=int, help='kernel size',       default=3)
    parser.add_argument('--num_layer',  type=int, help='number of layers',  default=5)
    parser.add_argument('--stride',               help='stride',            default=1)
    parser.add_argument('--padd_size',  type=int, help='net pad size',      default=0)  # math.floor(opt.ker_size/2)

    # pyramid parameters:
    parser.add_argument('--scale_factor',   type=float, help='Pyramid scale factor',                    default=0.75)  # pow(0.5,1/6))
    # parser.add_argument('--noise_amp',      type=float, help='Additive noise cont weight',              default=0.1)
    parser.add_argument('--min_size',       type=int,   help='Image minimal size at the coarser scale', default=25)
    parser.add_argument('--max_size',       type=int,   help='Image minimal size at the coarser scale', default=250)

    # optimization hyper parameters:
    parser.add_argument('--niter',          type=int,   default=2000,   help='Number of epochs to train per scale')
    parser.add_argument('--num_workers',    type=int,   default=1)
    parser.add_argument('--batch_size',     type=int,   default=3)
    parser.add_argument('--gamma',          type=float, default=0.1,    help='Scheduler gamma')
    parser.add_argument('--lr_g',           type=float, default=0.0005, help='Learning rate, default=0.0005')
    parser.add_argument('--lr_d',           type=float, default=0.0005, help='Learning rate, default=0.0005')
    parser.add_argument('--beta1',          type=float, default=0.5,    help='Beta1 for adam. default=0.5')
    parser.add_argument('--lambda_grad',    type=float, default=0.1,    help='Gradient penelty weight')
    parser.add_argument('--alpha',          type=float, default=10,     help='Reconstruction loss weight')

    return parser

def post_config(opt):
    opt.model_dir = os.path.join(opt.top_models_dir, opt.real_name[:-4],
                                 f'scale_factor={opt.scale_factor},'
                                 f'alpha={opt.alpha},'
                                 f'min_size={opt.min_size},'
                                 f'max_size={opt.max_size}',
                                 f'batch_size={opt.batch_size},'
                                 f'niter={opt.niter}')
    opt.res_dir   = os.path.join(opt.top_res_dir, opt.real_name[:-4])

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    #TODO: consider stride and padding in the following calculation
    if opt.ker_size % 2 == 0:
        opt.pad_noise = int((opt.ker_size - 1) * opt.num_layer)
        opt.pad_image = int((opt.ker_size - 1) * opt.num_layer)
    else: # opt.ker_size % 2 != 0
        opt.pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        opt.pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc

    return opt