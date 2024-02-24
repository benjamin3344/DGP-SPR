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

from cryodrgn.model2 import HetOnlyVAE, VAEvampprior
from cryodrgn.lattice import Lattice
import cryodrgn.ctf as ctf
from score_sde.ncsnpp import NCSNpp
import utils
import datasets

from vampprior.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256

try:
    import apex.amp as amp
except:
    pass



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
    group.add_argument('-b','--batch-size', type=int, default=8, help='Minibatch size (default: %(default)s)')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer (default: %(default)s)')
    group.add_argument('--lr', type=float, default=1e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--amp', action='store_true', help='Accelerate training speed with mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')
    group.add_argument('--beta', default=None,
                       help='Choice of beta schedule or a constant for KLD weight (default: 1/zdim)')

    group.add_argument('--beta-control', type=float, help='KL-Controlled VAE gamma. Beta is KL target. (default: %(default)s)')

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

    # vampprior
    parser.add_argument('--prior', type=str, default='vampprior', metavar='P',
                        help='prior: standard, vampprior')
    parser.add_argument('--number-components', type=int, default=500, metavar='NC',
                        help='number of pseudo-inputs')
    parser.add_argument('--pseudoinputs-mean', type=float, default=-0.05, metavar='PM',
                        help='mean for init pseudo-inputs')
    parser.add_argument('--pseudoinputs-std', type=float, default=0.01, metavar='PS',
                        help='std for init pseudo-inputs')
    parser.add_argument('--use_training_data_init', action='store_true', default=False,
                        help='initialize pseudo-inputs with randomly chosen training data')



    return parser


def train_batch(model, lattice, y, rot, trans, optim, beta, beta_control=None, ctf_params=None, yr=None,
                use_amp=False, scaler=None):
    optim.zero_grad()
    model.train()
    if trans is not None:
        y = preprocess_input(y, lattice, trans)
    # Cast operations to mixed precision if using torch.cuda.amp.GradScaler()
    if scaler is not None:
        with torch.cuda.amp.autocast():
            z_mu, z_logvar, z, log_p_z, y_recon, mask = run_batch(model, lattice, y, rot, ctf_params, yr)
            loss, gen_loss, kld = loss_function(z_mu, z_logvar, z, log_p_z, y, y_recon, mask, beta, beta_control)
    else:
        z_mu, z_logvar, z, log_p_z, y_recon, mask = run_batch(model, lattice, y, rot, ctf_params, yr)
        loss, gen_loss, kld = loss_function(z_mu, z_logvar, z, log_p_z, y, y_recon, mask, beta, beta_control)
    if use_amp:
        if scaler is not None:  # torch mixed precision
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:  # apex.amp mixed precision
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
            optim.step()
    else:
        loss.backward()
        optim.step()
    return loss.item(), gen_loss.item(), kld.item()


def preprocess_input(y, lattice, trans):
    # center the image
    B = y.size(0)
    D = lattice.D
    y = lattice.translate_ht(y.view(B, -1), trans.unsqueeze(1)).view(B, D, D)
    return y


def _unparallelize(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def run_batch(model, lattice, y, rot, ctf_params=None, yr=None):
    use_ctf = ctf_params is not None
    B = y.size(0)
    D = lattice.D
    if use_ctf:
        freqs = lattice.freqs2d.unsqueeze(0).expand(B, *lattice.freqs2d.shape) / ctf_params[:, 0].view(B, 1, 1)
        c = ctf.compute_ctf(freqs, *torch.split(ctf_params[:, 1:], 1, 1)).view(B, D, D)

    # encode
    if yr is not None:
        input_ = (yr,)
    else:
        input_ = (y,)
        if use_ctf: input_ = (x * c.sign() for x in input_)  # phase flip by the ctf
    z_mu, z_logvar = _unparallelize(model).encode(*input_)
    z = _unparallelize(model).reparameterize(z_mu, z_logvar)
    log_p_z = _unparallelize(model).log_p_z(z)

    # decode
    mask = lattice.get_circular_mask(D // 2)  # restrict to circular mask
    y_recon = model(lattice.coords[mask] / lattice.extent / 2 @ rot, z).view(B, -1)
    if use_ctf: y_recon *= c.view(B, -1)[:, mask]

    # decode the tilt series
    return z_mu, z_logvar, z, log_p_z, y_recon, mask


def loss_function(z_mu, z_logvar, z, log_p_z, y, y_recon, mask, beta, beta_control=None):
    # reconstruction error
    B = y.size(0)
    gen_loss = F.mse_loss(y_recon, y.view(B, -1)[:, mask])
    # latent loss

    vae_neg_entropy =  log_Normal_diag(z, z_mu, z_logvar, dim=1)
    cross_entropy = log_p_z
    kld = -(cross_entropy - vae_neg_entropy)
    kld = torch.mean(kld)
    # total loss
    if beta_control is None:
        loss = gen_loss + beta * kld / mask.sum().float()
    else:
        loss = gen_loss + args.beta_control * (beta - kld) ** 2 / mask.sum().float()
    return loss, gen_loss, kld


def eval_z(model, lattice, data, batch_size, device, trans=None, ctf_params=None, use_real=False):
    assert not model.training
    z_mu_all = []
    z_logvar_all = []
    data_generator = DataLoader(data, batch_size=batch_size, shuffle=False)
    for minibatch in data_generator:
        ind = minibatch[-1]
        y = minibatch[0].to(device)
        B = len(ind)
        D = lattice.D
        if ctf_params is not None:
            freqs = lattice.freqs2d.unsqueeze(0).expand(B, *lattice.freqs2d.shape) / ctf_params[ind, 0].view(B, 1, 1)
            c = ctf.compute_ctf(freqs, *torch.split(ctf_params[ind, 1:], 1, 1)).view(B, D, D)
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
    z_mu_all = np.vstack(z_mu_all)
    z_logvar_all = np.vstack(z_logvar_all)
    return z_mu_all, z_logvar_all


def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z):
    '''Save model weights, latent encoding z, and decoder volumes'''
    # save model weights
    torch.save({
        'epoch': epoch,
        'model_state_dict': _unparallelize(model).state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }, out_weights)
    # save z
    with open(out_z, 'wb') as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)


def save_config(args, dataset, lattice, model, out_config):
    dataset_args = dict(particles=args.particles,
                        norm=dataset.norm,
                        invert_data=args.invert_data,
                        ind=args.ind,
                        keepreal=args.use_real,
                        window=args.window,
                        window_r=args.window_r,
                        datadir=args.datadir,
                        ctf=args.ctf,
                        poses=args.poses,
                        do_pose_sgd=args.do_pose_sgd)
    lattice_args = dict(D=lattice.D,
                        extent=lattice.extent,
                        ignore_DC=lattice.ignore_DC)
    model_args = dict(qlayers=args.qlayers,
                      qdim=args.qdim,
                      players=args.players,
                      pdim=args.pdim,
                      zdim=args.zdim,
                      encode_mode=args.encode_mode,
                      enc_mask=args.enc_mask,
                      pe_type=args.pe_type,
                      feat_sigma=args.feat_sigma,
                      pe_dim=args.pe_dim,
                      domain=args.domain,
                      activation=args.activation)
    config = dict(dataset_args=dataset_args,
                  lattice_args=lattice_args,
                  model_args=model_args)
    config['seed'] = args.seed
    with open(out_config, 'wb') as f:
        pickle.dump(config, f)
        meta = dict(time=dt.now(),
                    cmd=sys.argv)
 
        pickle.dump(meta, f)


def main(args):

    t1 = dt.now()
    # common initialization copied from LSGM
    logging, writer = utils.common_init(args.global_rank, args.seed, args.save)

    # Data preprocess following the formats in CryoDRGN
    logging.info('loading datasets')
    data, posetracker, ctf_params = datasets.get_loaders(args)
    pose_optimizer = torch.optim.SparseAdam(list(posetracker.parameters()), lr=args.pose_lr) if args.do_pose_sgd else None
    train_queue = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # instantiate model
    device = torch.device('cuda')
    Nimg = data.N
    D = data.D
    # instantiate model
    lattice = Lattice(D, extent=0.5, device=device)
    if args.enc_mask is None:
        args.enc_mask = D//2
    if args.enc_mask > 0:
        assert args.enc_mask <= D//2
        enc_mask = lattice.get_circular_mask(args.enc_mask)
        in_dim = enc_mask.sum()
    elif args.enc_mask == -1:
        enc_mask = None
        in_dim = lattice.D**2 if not args.use_real else (lattice.D-1)**2
    else:
        raise RuntimeError("Invalid argument for encoder mask radius {}".format(args.enc_mask))
    activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]

    model = VAEvampprior(lattice, args.qlayers, args.qdim, args.players, args.pdim,
                       in_dim, args.zdim, encode_mode=args.encode_mode, enc_mask=enc_mask,
                       enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
                       activation=activation, feat_sigma=args.feat_sigma,  number_components=args.number_components,
                       pseudoinputs_mean=args.pseudoinputs_mean, pseudoinputs_std=args.pseudoinputs_std)

    model.to(device)
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(model.parameters(), args.distributed)

    # logging_info
    logging.info('args = %s', args)
    logging.info(model)
    logging.info('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logging.info('{} parameters in encoder'.format(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)))
    logging.info('{} parameters in deoder'.format(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)))
    logging.info('VAE_Cryodrgn: param size = %fM ', utils.count_parameters_in_M(model))

    # save configuration
    out_config = '{}/config.pkl'.format(args.save)
    save_config(args, data, lattice, model, out_config)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Mixed precision training
    scaler = None
    if args.amp:
        assert args.batch_size % 8 == 0, "Batch size must be divisible by 8 for AMP training"
        assert (D-1) % 8 == 0, "Image size must be divisible by 8 for AMP training"
        assert args.pdim % 8 == 0, "Decoder hidden layer dimension must be divisible by 8 for AMP training"
        assert args.qdim % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        # Also check zdim, enc_mask dim? Add them as warnings for now.
        if args.zdim % 8 != 0:
            logging.info('Warning: z dimension is not a multiple of 8 -- AMP training speedup is not optimized')
        if in_dim % 8 != 0:
            logging.info('Warning: Masked input image dimension is not a mutiple of 8 -- AMP training speedup is not optimized')
        try: # Mixed precision with apex.amp
            model, optim = amp.initialize(model, optim, opt_level='O1')
        except: # Mixed precision with pytorch (v1.6+)
            scaler = torch.cuda.amp.GradScaler()

    # restart from checkpoint
    if args.load:
        logging.info('loading the model from {}'.format(args.load))
        checkpoint_file = os.path.join(args.save, args.load)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        model.train()
    else:
        start_epoch = 0

    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        logging.info(f'Using {torch.cuda.device_count()} GPUs!')
        args.batch_size *= torch.cuda.device_count()
        logging.info(f'Increasing batch size to {args.batch_size}')
        model = nn.DataParallel(model)
    elif args.multigpu:
        logging.info(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')


    # Training
    num_epochs = args.num_epochs
    for epoch in range(start_epoch, num_epochs):
        t2 = dt.now()
        gen_loss_accum = 0
        loss_accum = 0
        kld_accum = 0
        eq_loss_accum = 0
        batch_it = 0
        for minibatch in train_queue:
            ind = minibatch[-1].to(device)
            y = minibatch[0].to(device)
            B = len(ind)
            batch_it += B
            global_it = Nimg * epoch + batch_it

            if args.beta is None:
                args.beta = 1. / args.zdim
            try:
                args.beta = float(args.beta)
            except ValueError:
                assert args.beta_control, "Need to set beta control weight for schedule {}".format(args.beta)
            beta = args.beta  #beta = beta_schedule(global_it)

            yr = torch.from_numpy(data.particles_real[ind.numpy()]).to(device) if args.use_real else None
            if args.do_pose_sgd:
                pose_optimizer.zero_grad()
            rot, tran = posetracker.get_pose(ind)
            ctf_param = ctf_params[ind] if ctf_params is not None else None
            loss, gen_loss, kld = train_batch(model, lattice, y, rot, tran, optim, beta, args.beta_control,
                                              ctf_params=ctf_param, yr=yr, use_amp=args.amp, scaler=scaler)
            if args.do_pose_sgd and epoch >= args.pretrain:
                pose_optimizer.step()

            # logging
            gen_loss_accum += gen_loss * B
            kld_accum += kld * B
            loss_accum += loss * B

            if batch_it % args.log_interval == 0:
                logging.info(
                    '# [Train Epoch: {}/{}] [{}/{} images] gen loss={:.6f}, kld={:.6f}, beta={:.6f}, loss={:.6f}'.format(
                        epoch + 1, num_epochs, batch_it, Nimg, gen_loss, kld, beta, loss))
        logging.info('# =====> Epoch: {} Average gen loss = {:.6}, KLD = {:.6f}, total loss = {:.6f}; Finished in {}'.format(
            epoch + 1, gen_loss_accum / Nimg, kld_accum / Nimg, loss_accum / Nimg, dt.now() - t2))

        if args.checkpoint and epoch % args.checkpoint == 0:
            out_weights = '{}/weights.{}.pkl'.format(args.save, epoch)
            out_z = '{}/z.{}.pkl'.format(args.save, epoch)
            model.eval()
            with torch.no_grad():
                z_mu, z_logvar = eval_z(model, lattice, data, args.batch_size, device, posetracker.trans,
                                        ctf_params, args.use_real)
                save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z)
            if args.do_pose_sgd and epoch >= args.pretrain:
                out_pose = '{}/pose.{}.pkl'.format(args.save, epoch)
                posetracker.save(out_pose)

    # save model weights, latent encoding, and evaluate the model on 3D lattice
    out_weights = '{}/weights.pkl'.format(args.save)
    out_z = '{}/z.pkl'.format(args.save)
    model.eval()
    with torch.no_grad():
        z_mu, z_logvar = eval_z(model, lattice, data, args.batch_size, device, posetracker.trans,
                                ctf_params, args.use_real)
        save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z)

    if args.do_pose_sgd and epoch >= args.pretrain:
        out_pose = '{}/pose.pkl'.format(args.save)
        posetracker.save(out_pose)
    td = dt.now() - t1
    logging.info('Finished in {} ({} per epoch)'.format(td, td / (num_epochs - start_epoch)))



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
