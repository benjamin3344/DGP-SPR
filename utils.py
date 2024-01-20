import os
import sys
import time
import logging
from tensorboardX import SummaryWriter
import torch
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast
import types

def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))


class Logger(object):
    def __init__(self, rank, save):
        # other libraries may set logging before arriving at this line.
        # by reloading logging, we can get rid of previous configs set by other libraries.
        from importlib import reload
        reload(logging)
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:  # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.close()


def common_init(rank, seed, save_dir):
    # we use different seeds per gpu. But we sync the weights after model initialization.
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True

    # prepare logging and tensorboard summary
    logging = Logger(rank, save_dir)
    writer = Writer(rank, save_dir)

    return logging, writer


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def broadcast_params(params, is_distributed):
    if is_distributed:
        for param in params:
            dist.broadcast(param.data, src=0)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    dist.barrier()
    dist.destroy_process_group()


def get_bpd_coeff(D):
    n = D * D
    return 1. / np.log(2.) / n


def get_dae_model(args, num_input_channels, Dimg):
    if args.dae_arch == 'ncsnpp':
        # we need to import NCSNpp after processes are launched on the multi gpu training.
        from score_sde.ncsnpp import NCSNpp
        dae = NCSNpp(args, num_input_channels, Dimg)
    else:
        raise NotImplementedError

    return dae


def set_vesde_sigma_max(args, vae, train_queue, logging, is_distributed):
    logging.info('')
    logging.info('Calculating max. pairwise distance in latent space to set sigma2_max for VESDE...')

    eps_list = []
    vae.eval()
    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()
        x = symmetrize_image_data(x)

        # run vae
        with autocast(enabled=args.autocast_train):
            with torch.set_grad_enabled(False):
                logits, all_log_q, all_eps = vae(x)
                eps = torch.cat(all_eps, dim=1)

        eps_list.append(eps.detach())

    # concat eps tensor on each GPU and then gather all on all GPUs
    eps_this_rank = torch.cat(eps_list, dim=0)
    if is_distributed:
        eps_all_gathered = [torch.zeros_like(eps_this_rank)] * dist.get_world_size()
        dist.all_gather(eps_all_gathered, eps_this_rank)
        eps_full = torch.cat(eps_all_gathered, dim=0)
    else:
        eps_full = eps_this_rank

    # max pairwise distance squared between all latent encodings, is computed on CPU
    eps_full = eps_full.cpu().float()
    eps_full = eps_full.flatten(start_dim=1).unsqueeze(0)
    max_pairwise_dist_sqr = torch.cdist(eps_full, eps_full).square().max()
    max_pairwise_dist_sqr = max_pairwise_dist_sqr.cuda()

    # to be safe, we broadcast to all GPUs if we are in distributed environment. Shouldn't be necessary in principle.
    if is_distributed:
        dist.broadcast(max_pairwise_dist_sqr, src=0)

    args.sigma2_max = max_pairwise_dist_sqr.item()

    logging.info('Done! Set args.sigma2_max set to {}'.format(args.sigma2_max))
    logging.info('')
    return args


def symmetrize_image_data(images):
    return 2.0 * images - 1.0


def cross_entropy_normal(all_eps):
    from score_sde.distributions import log_p_standard_normal

    cross_entropy = 0.
    neg_log_p_per_group = []
    for eps in all_eps:
        neg_log_p_conv = - log_p_standard_normal(eps)
        neg_log_p = torch.sum(neg_log_p_conv, dim=[1, 2, 3])
        cross_entropy += neg_log_p
        neg_log_p_per_group.append(neg_log_p_conv)

    return cross_entropy, neg_log_p_per_group


def common_x_operations(x, num_x_bits):
    x = x[0] if len(x) > 1 else x
    x = x.cuda()

    # change bit length
    x = change_bit_length(x, num_x_bits)
    x = symmetrize_image_data(x)

    return x


def change_bit_length(x, num_bits):
    if num_bits != 8:
        x = torch.floor(x * 255 / 2 ** (8 - num_bits))
        x /= (2 ** num_bits - 1)
    return x


def sum_log_q(all_log_q):
    log_q = 0.
    all_log_q = all_log_q.unsqueeze(0) # bs_2 expand
    for log_q_conv in all_log_q:
        log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
        # log_q += torch.sum(log_q_conv, dim=[0])

    return log_q


def get_mixed_prediction(mixed_prediction, param, mixing_logit, mixing_component=None):
    if mixed_prediction:
        assert mixing_component is not None, 'Provide mixing component when mixed_prediction is enabled.'
        coeff = torch.sigmoid(mixing_logit)
        param = (1 - coeff) * mixing_component + coeff * param

    return param


def average_gradients(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        if isinstance(params, types.GeneratorType):
            params = [p for p in params]

        size = float(dist.get_world_size())
        grad_data = []
        grad_size = []
        grad_shapes = []
        # Gather all grad values
        for param in params:
            if param.requires_grad:
                grad_size.append(param.grad.data.numel())
                grad_shapes.append(list(param.grad.data.shape))
                grad_data.append(param.grad.data.flatten())
        grad_data = torch.cat(grad_data).contiguous()

        # All-reduce grad values
        grad_data /= size
        dist.all_reduce(grad_data, op=dist.ReduceOp.SUM)

        # Put back the reduce grad values to parameters
        base = 0
        for i, param in enumerate(params):
            if param.requires_grad:
                param.grad.data = grad_data[base:base + grad_size[i]].view(grad_shapes[i])
                base += grad_size[i]


def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
    # return max(min(max_kl_coeff * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)


def kl_per_group_vada(all_log_q, all_neg_log_p):
    assert len(all_log_q) == len(all_neg_log_p)

    kl_all_list = []
    kl_diag = []
    all_log_q = all_log_q.unsqueeze(0) # bs_2 expand
    all_neg_log_p = all_neg_log_p.unsqueeze(0)
    for log_q, neg_log_p in zip(all_log_q, all_neg_log_p):
        kl_diag.append(torch.mean(torch.sum(neg_log_p + log_q, dim=[2, 3]), dim=0))
        kl_all_list.append(torch.sum(neg_log_p + log_q, dim=[1, 2, 3]))

    # kl_all = torch.stack(kl_all, dim=1)   # batch x num_total_groups
    kl_vals = torch.mean(torch.stack(kl_all_list, dim=1), dim=0)   # mean per group

    return kl_all_list, kl_vals, kl_diag


def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    if kl_balance and kl_coeff < 1.0:
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=1)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=1)
        kl_coeffs = torch.ones(size=(len(kl_vals),))

    return kl_coeff * kl, kl_coeffs, kl_vals


def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAvgrageMeter(object):

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size

def different_p_q_objectives(iw_sample_p, iw_sample_q):
    assert iw_sample_p in ['ll_uniform', 'drop_all_uniform', 'll_iw', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw',
                           'drop_sigma2t_uniform']
    assert iw_sample_q in ['reweight_p_samples', 'll_uniform', 'll_iw']
    # In these cases, we reuse the likelihood-based p-objective (either the uniform sampling version or the importance
    # sampling version) also for q.
    if iw_sample_p in ['ll_uniform', 'll_iw'] and iw_sample_q == 'reweight_p_samples':
        return False
    # In these cases, we are using a non-likelihood-based objective for p, and hence definitly need to use another q
    # objective.
    else:
        return True
