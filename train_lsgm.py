import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from datetime import datetime as dt
import pickle

from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from cryodrgn.model2 import HetOnlyVAE
from cryodrgn.lattice import Lattice
import cryodrgn.ctf as ctf

from score_sde.ncsnpp import NCSNpp
from score_sde.diffusion_discretized import DiffusionDiscretized
from score_sde.diffusion_continuous import make_diffusion
# try:
#     from apex.optimizers import FusedAdam
# except ImportError:
#     print("No Apex Available. Using PyTorch's native Adam. Install Apex for faster training.")
#     from torch.optim import Adam as FusedAdam
from torch.optim import Adam as FusedAdam
from score_sde.ema import EMA

from training_obj_joint import train_vada_joint
from training_obj_disjoint import train_vada_disjoint
from train_vae import eval_z, save_checkpoint

import utils
import datasets



def add_args(parser):
    # parser.add_argument('--latent-dim', dest='zdim', type=int, default=20,
    #                    help='Dimensions of latent variables (default: %(default)s)')

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

    # datasets
    group = parser.add_argument_group('Datasets')
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
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

    # training
    group = parser.add_argument_group('Training')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs (default: %(default)s)')
    # group.add_argument('-b','--batch-size', type=int, default=8, help='Minibatch size (default: %(default)s)')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer (default: %(default)s)')
    group.add_argument('--lr', type=float, default=1e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--amp', action='store_true', help='Accelerate training speed with mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')
    group.add_argument('--beta', default=None,
                       help='Choice of beta schedule or a constant for KLD weight (default: 1/zdim)')

    group.add_argument('--beta-control', type=float, help='KL-Controlled VAE gamma. Beta is KL target. (default: %(default)s)')

    # checkpoint
    group = parser.add_argument_group('Checkpoint')
    group.add_argument('--vae_checkpoint', type=str, default='',
                        help='Pretrained VAE checkpoint.')
    group.add_argument('--vada_checkpoint', type=str, default='',
                        help='Pretrained VADA checkpoint.')
    group.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    group.add_argument('--discard_vae_weights', action='store_true', default=False,
                        help='set true to ignore the vae weights from the checkpoint.')
    group.add_argument('--discard_dae_weights', action='store_true', default=False,
                        help='set true to ignore the dae weights from the checkpoint.')

    # encoder
    group = parser.add_argument_group('Encoder Network')
    group.add_argument('--enc-layers', dest='qlayers', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--enc-dim', dest='qdim', type=int, default=256, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--encode-mode', default='resid', choices=('conv','resid','mlp','tilt'), help='Type of encoder network (default: %(default)s)')
    group.add_argument('--enc-mask', type=int, help='Circular mask of image for encoder (default: D/2; -1 for no mask)')
    # decoder
    group = parser.add_argument_group('Decoder Network')
    group.add_argument('--dec-layers', dest='players', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--dec-dim', dest='pdim', type=int, default=256, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf', 'gaussian', 'none'), default='geom_lowf', help='Type of positional encoding (default: %(default)s)')
    group.add_argument('--feat-sigma', type=float, default=0.5, help="Scale for random Gaussian features")
    group.add_argument('--pe-dim', type=int, help='Num features in positional encoding (default: image D)')
    group.add_argument('--domain', choices=('hartley','fourier'), default='fourier', help='Decoder representation domain (default: %(default)s)')
    group.add_argument('--activation', choices=('relu','leaky_relu'), default='relu', help='Activation (default: %(default)s)')


    # DDP
    group = parser.add_argument_group('Distributed Data Parallel')
    group.add_argument('--autocast_train', action='store_true', default=True,
                        help='This flag enables FP16 in training.')
    group.add_argument('--autocast_eval', action='store_true', default=True,
                        help='This flag enables FP16 in evaluation.')
    group.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    group.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    group.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    group.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    group.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    group.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    group.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')

    # lsgm training
    group = parser.add_argument_group('Distributed Data Parallel')
    group.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    group.add_argument('--dae_arch', type=str, default='unet', choices=['ncsnpp'],
                        help='Switch between different DAE architectures.')
    parser.add_argument('--fir', action='store_true', default=False,
                        help='Enable FIR upsampling/downsampling')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='none', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    group.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    group.add_argument('--disjoint_training', action='store_true', default=False,
                        help='When p (sgm prior) and q (vae) have different objectives, trains them in two separate forward calls (Algorithm 2).')
    group.add_argument('--train_vae', action='store_true', default=False,
                        help='set true to train the vae model.')
    group.add_argument('--iw_sample_p', type=str, default='ll_uniform', choices=['ll_uniform', 'll_iw',
                        'drop_all_uniform', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw', 'drop_sigma2t_uniform'],
                        help='Specifies the weighting mechanism used for training p (sgm prior) and whether or not to use importance sampling')
    group.add_argument('--iw_sample_q', type=str, default='reweight_p_samples', choices=['reweight_p_samples', 'll_uniform', 'll_iw'],
                        help='Specifies the weighting mechanism used for training q (vae) and whether or not to use importance sampling. '
                             'reweight_p_samples indicates reweight the t samples generated for the prior as done in Algorithm 3.')
    group.add_argument('--update_q_ema', action='store_true', default=False,
                        help='Enables updating q with EMA parameters of prior.')
    # second stage VADA KL annealing
    group.add_argument('--cont_kl_anneal', action='store_true', default=False,
                        help='If true, we continue KL annealing using below setup when training LSGM.')
    group.add_argument('--kl_anneal_portion_vada', type=float, default=0.1,
                        help='The portions epochs that KL is annealed')
    group.add_argument('--kl_const_portion_vada', type=float, default=0.0,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    group.add_argument('--kl_const_coeff_vada', type=float, default=0.7,
                        help='The constant value used for min KL coeff')
    group.add_argument('--kl_max_coeff_vada', type=float, default=1.,
                        help='The constant value used for max KL coeff')
    group.add_argument('--kl_balance_vada', action='store_true', default=False,
                        help='If true, we use KL balancing during VADA KL annealing.')

    group.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    group.add_argument('--learning_rate_vae', type=float, default=1e-4,
                        help='init learning rate')
    group.add_argument('--learning_rate_min_vae', type=float, default=1e-5,
                        help='min learning rate')
    group.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay factor')
    group.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    group.add_argument('--weight_decay_norm_vae', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    group.add_argument('--grad_clip_max_norm', type=float, default=0.,
                        help='The maximum norm used in gradient norm clipping (0 applies no clipping).')
    group.add_argument('--learning_rate_dae', type=float, default=3e-4,
                        help='init learning rate')
    group.add_argument('--learning_rate_min_dae', type=float, default=3e-4,
                        help='min learning rate')
    group.add_argument('--weight_decay_norm_dae', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')


    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='dimension used for time embeddings')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='number of diffusion steps')
    parser.add_argument('--sigma2_0', type=float, default=0.0,
                        help='initial SDE variance at t=0 (sort of represents Normal perturbation of input data)')
    parser.add_argument('--beta_start', type=float, default=0.1,
                        help='initial beta variance value')
    parser.add_argument('--beta_end', type=float, default=20.0,
                        help='final beta variance value')
    parser.add_argument('--vpsde_power', type=int, default=2,
                        help='vpsde power for power-VPSDE')
    parser.add_argument('--sigma2_min', type=float, default=1e-4,
                        help='initial beta variance value')
    parser.add_argument('--sigma2_max', type=float, default=0.99,
                        help='final beta variance value')
    parser.add_argument('--sde_type', type=str, default='geometric_sde',
                        choices=['geometric_sde', 'vpsde', 'sub_vpsde', 'vesde'],
                        help='what kind of sde type to use when training/evaluating in continuous manner.')
    parser.add_argument('--train_ode_eps', type=float, default=1e-2,
                        help='ODE can only be integrated up to some epsilon > 0.')
    parser.add_argument('--train_ode_solver_tol', type=float, default=1e-4,
                        help='ODE solver error tolerance.')
    parser.add_argument('--eval_ode_eps', type=float, default=1e-5,
                        help='ODE can only be integrated up to some epsilon > 0.')
    parser.add_argument('--eval_ode_solver_tol', type=float, default=1e-5,
                        help='ODE solver error tolerance.')
    parser.add_argument('--time_eps', type=float, default=1e-2,
                        help='During training, t is sampled in [time_eps, 1.].')
    parser.add_argument('--denoising_stddevs', type=str, default='beta', choices=['learn', 'beta', 'beta_post'],
                        help='enables learning the conditional VAE decoder distribution standard deviations')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout probability applied to the denosing model')

    parser.add_argument('--mixing_logit_init', type=float, default=-3,
                        help='The initial logit for mixing coefficient.')
    parser.add_argument('--embedding_type', type=str, choices=['positional', 'fourier'], default='positional',
                        help='Type of time embedding')
    parser.add_argument('--embedding_scale', type=float, default=1.,
                        # 'fourier':16, 'positional':1000, backward compatible: 1.
                        help='Embedding scale that is used for rescaling time')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--num_scales_dae', type=int, default=2,
                        help='number of spatial scales in denosing model')
    parser.add_argument('--num_cell_per_scale_dae', type=int, default=2,
                        help='number of cells per scale')

    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')

    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')

    parser.add_argument('--iw_subvp_like_vp_sde', action='store_true', default=False,
                        help='Only relevant when using Sub-VPSDE. When true, use VPSDE-based IW distributions.')


    return parser




def main(args):

    t1 = dt.now()
    # common initialization copied from LSGM
    logging, writer = utils.common_init(args.global_rank, args.seed, args.save)

    # data preprocess following the formats in CryoDRGN
    logging.info('loading datasets')
    data, posetracker, ctf_params = datasets.get_loaders(args)
    pose_optimizer = torch.optim.SparseAdam(list(posetracker.parameters()), lr=args.pose_lr) if args.do_pose_sgd else None
    train_queue = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs

    # load a pretrained vae or lsgm
    load_vae, load_dae = False, False
    if args.vae_checkpoint != '' or args.vada_checkpoint != '':
        assert not (args.vae_checkpoint != '' and args.vada_checkpoint != ''), 'provide only 1 checkpoint'
        checkpoint_path = args.vada_checkpoint if args.vada_checkpoint != '' else args.vae_checkpoint
        logging.info('loading pretrained vae checkpoint:')
        logging.info(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # stored_args = checkpoint['args']
        # utils.override_architecture_fields(args, stored_args, logging)
        load_vae = True and not args.discard_vae_weights  # true & true
        load_dae = args.vada_checkpoint != '' and not args.discard_dae_weights  # ? & true


    #####  vae part
    device = torch.device('cuda')
    Nimg = data.N
    D = data.D
    # instantiate model
    lattice = Lattice(D, extent=0.5, device=device)
    if args.enc_mask is None:
        args.enc_mask = D // 2
    if args.enc_mask > 0:
        assert args.enc_mask <= D // 2
        enc_mask = lattice.get_circular_mask(args.enc_mask)
        in_dim = enc_mask.sum()
    elif args.enc_mask == -1:
        enc_mask = None
        in_dim = lattice.D ** 2 if not args.use_real else (lattice.D - 1) ** 2
    else:
        raise RuntimeError("Invalid argument for encoder mask radius {}".format(args.enc_mask))
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    vae = HetOnlyVAE(lattice, args.qlayers, args.qdim, args.players, args.pdim,
                     in_dim, args.zdim, encode_mode=args.encode_mode, enc_mask=enc_mask,
                     enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
                     activation=activation, feat_sigma=args.feat_sigma)
    vae.to(device)
    if load_vae:
        logging.info('loading weights from vae checkpoint {}'.format(checkpoint_path))
        vae.load_state_dict(checkpoint['model_state_dict'])
    logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(vae.parameters(), args.distributed)
    # optim = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.wd)
    vae_optimizer = FusedAdam(vae.parameters(), args.learning_rate_vae, weight_decay=args.weight_decay, eps=1e-3)
    # vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    # vae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_vae)
    vae_scheduler = vae_optimizer

    #####  dae part
    args.mixed_prediction = True
    # num_input_channels = vae.latent_structure()[0]
    num_input_channels = args.zdim
    dae = utils.get_dae_model(args, num_input_channels, D)
    if load_dae:
        logging.info('loading weights from dae checkpoint')
        dae.load_state_dict(checkpoint['dae_state_dict'])
    dae = dae.cuda()

    # for VESDE, run one epoch over data and get encodings and estimate sigma2_max based on Song's/Ermon's techniques.
    if args.sde_type == 'vesde':
        assert args.sigma2_min == args.sigma2_0, "VESDE was proposed implicitly assuming sigma2_min = sigma2_0!"
        args = utils.set_vesde_sigma_max(args, vae, train_queue, logging, args.distributed)

    diffusion_cont = make_diffusion(args)
    diffusion_disc = DiffusionDiscretized(args, diffusion_cont.var)

    logging.info('DAE: param size = %fM ', utils.count_parameters_in_M(dae))
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(dae.parameters(), args.distributed)

    dae_optimizer = FusedAdam(dae.parameters(), args.learning_rate_dae, weight_decay=args.weight_decay, eps=1e-4)
    # add EMA functionality to the optimizer
    dae_optimizer = EMA(dae_optimizer, ema_decay=args.ema_decay)
    dae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_dae)

    grad_scalar = GradScaler()
    bpd_coeff = utils.get_bpd_coeff(D)


    # loading from checkpoints
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        dae.load_state_dict(checkpoint['dae_state_dict'])
        # load dae
        dae = dae.cuda()
        dae_optimizer.load_state_dict(checkpoint['dae_optimizer'])
        dae_scheduler.load_state_dict(checkpoint['dae_scheduler'])
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae = vae.cuda()
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        global_step = checkpoint['global_step']
        logging.info('loaded the model at epoch %d.', init_epoch)
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    # training
    for epoch in range(init_epoch, args.epochs):
        # if args.distributed:
        #     train_queue.sampler.set_epoch(global_step + args.seed)
        if epoch > args.warmup_epochs:
            dae_scheduler.step()
            # vae_scheduler.step()

        # # remove disabled latent variables by setting their mixing component to a small value
        # if epoch == 0 and args.mixed_prediction and args.drop_inactive_var:
        #     logging.info('inferring active latent variables.')
        #     is_active = infer_active_variables(train_queue, vae, args, max_iter=1000)
        #     dae.mixing_logit.data[0, torch.logical_not(is_active), 0, 0] = -15
        #     dae.is_active = is_active.float().view(1, -1, 1, 1)


        if args.disjoint_training:
            # we may use disjoint training for update q with ema
            assert args.iw_sample_p != args.iw_sample_q or args.update_q_ema, \
                'disjoint training is for the case training objective of p and q are not the same unless q is ' \
                'updated with the EMA parameters.'
            assert args.iw_sample_q in ['ll_uniform', 'll_iw']
            assert args.train_vae, 'disjoint training is used when training both VAE and prior.'

            # train_obj, global_step = train_vada_disjoint(train_queue, diffusion_cont, dae, dae_optimizer, vae, vae_optimizer,
            #                                              grad_scalar, global_step, warmup_iters, writer, logging, args)
            train_obj, global_step = train_vada_disjoint(train_queue, diffusion_cont, dae, dae_optimizer, vae, vae_optimizer,
                                                         lattice, posetracker, ctf_params, device,
                                                         grad_scalar, global_step, warmup_iters, writer, logging, args)
        else:
            assert not args.update_q_ema, 'q can be training with EMA parameters of prior in disjoint training only.'
            train_obj, global_step = train_vada_joint(train_queue, diffusion_cont, dae, dae_optimizer, vae, vae_optimizer,
                                                      lattice, posetracker, ctf_params, device,
                                                      grad_scalar, global_step, warmup_iters, writer, logging, args)

        logging.info('epoch {}'.format(epoch))
        logging.info('train_loss %f', train_obj)
        writer.add_scalar('train/loss_epoch', train_obj, global_step)

        if args.global_rank == 0:
            logging.info('saving the model.')
            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                       'grad_scalar': grad_scalar.state_dict(),
                       'dae_state_dict': dae.state_dict(), 'dae_optimizer': dae_optimizer.state_dict(),
                       'dae_scheduler': dae_scheduler.state_dict(), 'model_state_dict': vae.state_dict(),
                       'vae_optimizer': vae_optimizer.state_dict(), 'vae_scheduler': vae_scheduler.state_dict()}
            torch.save(content, checkpoint_file)

        if epoch % 5 == 0:
            out_z = '{}/zz.{}.pkl'.format(args.save, epoch)
            vae.eval()
            with torch.no_grad():
                z_mu, z_logvar = eval_z(vae, lattice, data, args.batch_size, device, posetracker.trans,
                                        ctf_params, args.use_real)
                save_z(vae, z_mu, z_logvar, out_z)



def save_z(model, z_mu, z_logvar, out_z):
    '''Save model weights, latent encoding z, and decoder volumes'''
    # save model weights
    with open(out_z, 'wb') as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='__doc__')
    args = add_args(parser).parse_args()

    args.save = args.root + '/' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=utils.init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        utils.init_processes(0, size, main, args)

