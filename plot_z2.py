import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import pickle

import datasets
from torch.utils.data import DataLoader

from cryodrgn.model2 import HetOnlyVAE
from cryodrgn.lattice import Lattice
from cryodrgn.utils import log
from cryodrgn import config
import cryodrgn.ctf as ctf
from cryodrgn import starfile
from train_vae import eval_z, save_checkpoint
import utils
import pprint

def add_args(parser):

    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nvae-diff/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')

    # ????

    parser.add_argument('--zdim', type=int, required=True, help='Dimension of latent variable')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=1000, help='Logging interval in N_IMGS (default: %(default)s)')
    parser.add_argument('-v','--verbose',action='store_true',help='Increaes verbosity')
    parser.add_argument('-c', '--config', metavar='PKL', required=True, help='CryoDRGN config.pkl file')
    # datasets
    group = parser.add_argument_group('Datasets')
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('--starfiles', type=os.path.abspath, help='Starfiles (.star)')
    parser.add_argument('--poses', type=os.path.abspath, required=True, help='Image poses (.pkl)')
    parser.add_argument('--ctf', metavar='pkl', type=os.path.abspath, help='CTF parameters (.pkl)')
    # parser.add_argument('--seed', type=int, default=np.random.randint(0,100000), help='Random seed')
    group.add_argument('--ind', type=os.path.abspath, metavar='PKL', help='Filter particle stack by these indices')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.85,  help='Windowing radius (default: %(default)s)')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')
    group.add_argument('--preprocessed', action='store_true', help='Skip preprocessing steps if input data is from cryodrgn preprocess_mrcs')
    group.add_argument('--max-threads', type=int, default=16, help='Maximum number of CPU cores for FFT parallelization (default: %(default)s)')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: 0, std of dataset)')
    group.add_argument('--use-real', action='store_true', help='Use real space image for encoder (for convolutional encoder)')
    # pose
    group = parser.add_argument_group('Pose SGD')
    group.add_argument('--do-pose-sgd', action='store_true', help='Refine poses with gradient descent')
    group.add_argument('--pretrain', type=int, default=1, help='Number of epochs with fixed poses before pose SGD (default: %(default)s)')
    group.add_argument('--emb-type', choices=('s2s2','quat'), default='quat', help='SO(3) embedding type for pose SGD (default: %(default)s)')
    group.add_argument('--pose-lr', type=float, default=3e-4, help='Learning rate for pose optimizer (default: %(default)s)')


    # checkpoint
    group = parser.add_argument_group('Checkpoint')
    group.add_argument('--vae_checkpoint', type=str, default='',
                        help='Pretrained VAE checkpoint.')
    group.add_argument('--vada_checkpoint', type=str, default='',
                        help='Pretrained VADA checkpoint.')


    group = parser.add_argument_group('Overwrite architecture hyperparameters in config.pkl')
    #group.add_argument('--norm', nargs=2, type=float)
    group.add_argument('-D', type=int, help='Box size')
    group.add_argument('--enc-layers', dest='qlayers', type=int, help='Number of hidden layers')
    group.add_argument('--enc-dim', dest='qdim', type=int, help='Number of nodes in hidden layers')
    #group.add_argument('--zdim', type=int,  help='Dimension of latent variable')
    group.add_argument('--encode-mode', choices=('conv','resid','mlp','tilt'), help='Type of encoder network')
    group.add_argument('--dec-layers', dest='players', type=int, help='Number of hidden layers')
    group.add_argument('--dec-dim', dest='pdim', type=int, help='Number of nodes in hidden layers')
    group.add_argument('--enc-mask', type=int, help='Circular mask radius for image encoder')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf','none'), help='Type of positional encoding')
    group.add_argument('--feat-sigma', type=float, help="Scale for random Gaussian features")
    group.add_argument('--pe-dim', type=int, help='Num sinusoid features in positional encoding (default: D/2)')
    group.add_argument('--domain', choices=('hartley','fourier'))
    group.add_argument('--l-extent', type=float, help='Coordinate lattice size')
    group.add_argument('--activation', choices=('relu','leaky_relu'), default='relu', help='Activation (default: %(default)s)')
    group.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')




    return parser




def main(args):

    args.save = args.root + '/' + args.save
    utils.create_exp_dir(args.save)

    data, posetracker, ctf_params = datasets.get_loaders(args)

    s = starfile.Starfile.load(args.starfiles)
    N = len(s.df)
    log(f"{N} particles")

    if "_rlnUnknownLabel" in s.headers:
        a = s.df["_rlnUnknownLabel"].str.split('_').values
        a = np.array([np.array(list(map(float, x))) for x in a])
        b = a[:, 0]
        c = a[:, 1]
        d = np.array([[np.min(b[np.nonzero(b)]), np.min(c[np.nonzero(c)])]])
        colorp = np.rint(a / d).astype('int')
        colorp1 = np.remainder(colorp, [3, 3])
        colorp2 = 3 * colorp1[:, 0] + colorp1[:, 1] # color labels
        colorp2 = np.vstack((b, c, colorp2))


    device = torch.device('cuda')


    # model
    if args.vae_checkpoint != '' or args.vada_checkpoint != '':
        assert not (args.vae_checkpoint != '' and args.vada_checkpoint != ''), 'provide only 1 checkpoint'
        checkpoint_path = args.vada_checkpoint if args.vada_checkpoint != '' else args.vae_checkpoint
        log('loading pretrained vae checkpoint:')
        log(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    cfg = config.overwrite_config(args.config, args)
    log('Loaded configuration:')
    # D = cfg['lattice_args']['D'] # image size + 1
    # zdim = cfg['model_args']['zdim']

    vae, lattice = HetOnlyVAE.load(cfg, checkpoint_path, device=device)
    vae.eval()

    out_z = '{}/zzcolor.pkl'.format(args.save)
    with torch.no_grad():
        z_mu, z_logvar, z_color = eval_z_wc(vae, lattice, data, colorp2, args.batch_size, device, posetracker.trans,
                                ctf_params,  args.use_real)
        save_z(vae, z_mu, z_logvar, z_color, out_z)


def save_z(model, z_mu, z_logvar, z_color, out_z):
    '''Save model weights, latent encoding z, and decoder volumes'''
    # save model weights
    with open(out_z, 'wb') as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)
        pickle.dump(z_color, f)


def eval_z_wc(model, lattice, data, z_color, batch_size, device, trans=None, ctf_params=None, use_real=False):
    assert not model.training
    z_mu_all = []
    z_logvar_all = []
    z_color_all = []
    data_generator = DataLoader(data, batch_size=batch_size, shuffle=False)
    for minibatch in data_generator:
        ind = minibatch[-1]
        y = minibatch[0].to(device)
        B = len(ind)
        D = lattice.D
        bfactor=50
        if ctf_params is not None:
            freqs = lattice.freqs2d.unsqueeze(0).expand(B, *lattice.freqs2d.shape) / ctf_params[ind, 0].view(B, 1, 1)
            c = ctf.compute_ctf(freqs, *torch.split(ctf_params[ind, 1:], 1, 1), bfactor).view(B, D, D)
        if trans is not None:
            y = lattice.translate_ht(y.view(B, -1), trans[ind].unsqueeze(1)).view(B, D, D)
        if use_real:
            input_ = (torch.from_numpy(data.particles_real[ind]).to(device),)
        else:
            input_ = (y,)
        if ctf_params is not None:
            assert not use_real, "Not implemented"
            input_ = (x * c.sign() for x in input_)  # phase flip by the ctf
        z_mu, z_logvar = _unparallelize(model).encode(*input_)
        z_mu_all.append(z_mu.detach().cpu().numpy())
        z_logvar_all.append(z_logvar.detach().cpu().numpy())
        z_color_all.append(z_color[:, ind])
    z_mu_all = np.vstack(z_mu_all)
    z_logvar_all = np.vstack(z_logvar_all)
    z_color_all = np.vstack(z_color)
    return z_mu_all, z_logvar_all, z_color_all

def _unparallelize(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='__doc__')
    args = add_args(parser).parse_args()
    main(args)
