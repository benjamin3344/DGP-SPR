import numpy as np
import torch
import torch.nn as nn
import math
from . import fft
from . import utils
from . import lattice

from torch.autograd import Variable
from vampprior.nn import normal_init, NonLinear, he_init
from vampprior.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from exemplar.utils.distributions import log_normal_diag_vectorized, pairwise_distance




class HetOnlyVAE(nn.Module):
    def __init__(self, lattice,  # Lattice object
                 qlayers, qdim,
                 players, pdim,
                 in_dim, zdim=1,
                 encode_mode='resid',
                 enc_mask=None,
                 enc_type='linear_lowf',
                 enc_dim=None,
                 domain='fourier',
                 activation=nn.ReLU,
                 feat_sigma=None):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        assert encode_mode == 'resid'
        self.encoder = ResidLinearMLP(in_dim,
                                      qlayers,  # nlayers
                                      qdim,  # hidden_dim
                                      zdim * 2,  # out_dim
                                      activation)
        self.encode_mode = encode_mode
        self.decoder = get_decoder(3 + zdim, lattice.D, players, pdim, domain, enc_type, enc_dim, activation,
                                   feat_sigma)


    @classmethod
    def load(self, config, weights=None, device=None):
        '''Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        '''
        cfg = utils.load_pkl(config) if type(config) is str else config
        c = cfg['lattice_args']
        lat = lattice.Lattice(c['D'], extent=c['extent'], device=device)
        c = cfg['model_args']
        if c['enc_mask'] > 0:
            enc_mask = lat.get_circular_mask(c['enc_mask'])
            in_dim = int(enc_mask.sum())
        else:
            assert c['enc_mask'] == -1
            enc_mask = None
            in_dim = lat.D ** 2
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
        model = HetOnlyVAE(lat,
                           c['qlayers'], c['qdim'],
                           c['players'], c['pdim'],
                           in_dim, c['zdim'],
                           encode_mode=c['encode_mode'],
                           enc_mask=enc_mask,
                           enc_type=c['pe_type'],
                           enc_dim=c['pe_dim'],
                           domain=c['domain'],
                           activation=activation,
                           feat_sigma=c['feat_sigma'])
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt['model_state_dict'])
        if device is not None:
            model.to(device)
        return model, lat

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, *img):
        img = (x.view(x.shape[0], -1) for x in img)
        if self.enc_mask is not None:
            img = (x[:, self.enc_mask] for x in img)
        z = self.encoder(*img)
        return z[:, :self.zdim], z[:, self.zdim:]


    def cat_z(self, coords, z):
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat((coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, z, mask=None):
        '''
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords, z))

    def forward(self, *args, **kwargs):
        return self.decode(*args, **kwargs)

    def latent_structure(self):
        structure = self.zdim
        return structure


def get_decoder(in_dim, D, layers, dim, domain, enc_type, enc_dim=None, activation=nn.ReLU, feat_sigma=None):
    assert enc_type != 'none'
    assert domain != 'hartley'
    model = FTPositionalDecoder
    return model(in_dim, D, layers, dim, activation, enc_type=enc_type, enc_dim=enc_dim, feat_sigma=feat_sigma)


class FTPositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None,
                 feat_sigma=None):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)

        if enc_type == "gaussian":
            # We construct 3 * self.enc_dim random vector frequences, to match the original positional encoding:
            # In the positional encoding we produce self.enc_dim features for each of the x,y,z dimensions,
            # whereas in gaussian encoding we produce self.enc_dim features each with random x,y,z components
            #
            # Each of the random feats is the sine/cosine of the dot product of the coordinates with a frequency
            # vector sampled from a gaussian with std of feat_sigma
            rand_freqs = torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * feat_sigma
            # make rand_feats a parameter so it is saved in the checkpoint, but do not perform SGD on it
            self.rand_freqs = nn.Parameter(rand_freqs, requires_grad=False)
        else:
            self.rand_feats = None

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        if self.enc_type == "gaussian":
            return self.random_fourier_encoding(coords)
        freqs = torch.arange(self.enc_dim, dtype=torch.float, device=coords.device)
        if self.enc_type == 'geom_ft':
            freqs = self.DD * np.pi * (2. / self.DD) ** (freqs / (self.enc_dim - 1))  # option 1: 2/D to 1
        elif self.enc_type == 'geom_full':
            freqs = self.DD * np.pi * (1. / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))  # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2 * (1. / self.D2) ** (freqs / (self.enc_dim - 1))  # option 3: 2/D*2pi to 2pi
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2 * (2. * np.pi / self.D2) ** (freqs / (self.enc_dim - 1))  # option 4: 2/D*2pi to 1
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def random_fourier_encoding(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2

        kxkykz = (coords[..., None, 0:3] * freqs)  # compute the x,y,z components of k
        k = kxkykz.sum(-1)  # compute k
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float, device=coords.device)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        # if ignore_DC = False, then the size of the lattice will be odd (since it
        # includes the origin), so we need to evaluate one additional pixel
        c = lattice.shape[-2] // 2  # top half
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c  # include the origin
        assert abs(lattice[..., 0:3].mean()) < 1e-4, '{} != 0.0'.format(lattice[..., 0:3].mean())
        image = torch.empty(lattice.shape[:-1], device=lattice.device)
        top_half = self.decode(lattice[..., 0:cc, :])
        image[..., 0:cc] = top_half[..., 0] - top_half[..., 1]
        # the bottom half of the image is the complex conjugate of the top half
        image[..., cc:] = (top_half[..., 0] + top_half[..., 1])[..., np.arange(c - 1, -1, -1)]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        assert (lattice[..., 0:3].abs() - 0.5 < 1e-4).all()
        # convention: only evalute the -z points
        w = lattice[..., 2] > 0.0
        lattice[..., 0:3][w] = -lattice[..., 0:3][w]  # negate lattice coordinates where z > 0
        result = self.decoder(self.positional_encoding_geom(lattice))
        result[..., 1][w] *= -1  # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        assert extent <= 0.5
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32, device=coords.device)

        vol_f = np.zeros((D, D, D), dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)):
            x = coords + torch.tensor([0, 0, dz], device=coords.device)
            keep = x.pow(2).sum(dim=1) <= extent ** 2
            x = x[keep]
            if zval is not None:
                x = torch.cat((x, z.expand(x.shape[0], zdim)), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[..., 0] - y[..., 1]
                slice_ = torch.zeros(D ** 2, device='cpu')
                slice_[keep] = y.cpu()
                slice_ = slice_.view(D, D).numpy()
            vol_f[i] = slice_
        vol_f = vol_f * norm[1] + norm[0]
        vol = fft.ihtn_center(vol_f[:-1, :-1, :-1])  # remove last +k freq for inverse FFT
        return vol


class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(ResidLinearMLP, self).__init__()
        layers = [ResidLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim),
                  activation()]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(ResidLinear(hidden_dim, out_dim) if out_dim == hidden_dim else nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(nin, nout)

    def forward(self, x):
        z = self.linear(x) + x
        return z


class VAEvampprior(HetOnlyVAE):
    # # AUXILIARY METHODS
    def __init__(self,lattice,  # Lattice object
                     qlayers, qdim,
                     players, pdim,
                     in_dim, zdim=1,
                     encode_mode='resid',
                     enc_mask=None,
                     enc_type='linear_lowf',
                     enc_dim=None,
                     domain='fourier',
                     activation=nn.ReLU,
                     feat_sigma=None,
                     number_components=500,
                     pseudoinputs_mean=-0.05,
                     pseudoinputs_std=0.01):
        super(VAEvampprior, self).__init__(lattice,  # Lattice object
                     qlayers, qdim,
                     players, pdim,
                     in_dim, zdim,
                     encode_mode,
                     enc_mask,
                     enc_type,
                     enc_dim,
                     domain,
                     activation,
                     feat_sigma)
        self.number_components = number_components
        self.pseudoinputs_mean = pseudoinputs_mean
        self.pseudoinputs_std = pseudoinputs_std
        self.add_pseudoinputs()

    def add_pseudoinputs(self):
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)

        # self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False, activation=nonlinearity)
        self.means = NonLinear(self.number_components, np.prod([1, self.lattice.D, self.lattice.D]), bias=False, activation=nonlinearity)
        # self.means = NonLinear(self.number_components, self.in_dim, bias=False, activation=nonlinearity)


        # init pseudo-inputs
        # if self.args.use_training_data_init:
        #     self.means.linear.weight.data = self.args.pseudoinputs_mean
        # else:
        #     normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)
        normal_init(self.means.linear, self.pseudoinputs_mean, self.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = Variable(torch.eye(self.number_components, self.number_components), requires_grad=False)
        self.idle_input = self.idle_input.cuda()


    def log_p_z(self, z):

        # z - MB x M
        # C = self.args.number_components
        C= self.number_components

        # calculate params
        X = self.means(self.idle_input)

        # calculate params for given data
        # z_p_mean, z_p_logvar = self.q_z(X)  # C x M
        z_p_mean, z_p_logvar = self.encode(X)

        # expand z
        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)

        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1


    # def log_p_z(self, z):
    #     if self.args.prior == 'standard':
    #         log_prior = log_Normal_standard(z, dim=1)
    #
    #     elif self.args.prior == 'vampprior':
    #         # z - MB x M
    #         C = self.args.number_components
    #
    #         # calculate params
    #         X = self.means(self.idle_input)
    #
    #         # calculate params for given data
    #         # z_p_mean, z_p_logvar = self.q_z(X)  # C x M
    #         z_p_mean, z_p_logvar = self.encode(X)
    #
    #         # expand z
    #         z_expand = z.unsqueeze(1)
    #         means = z_p_mean.unsqueeze(0)
    #         logvars = z_p_logvar.unsqueeze(0)
    #
    #         a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
    #         a_max, _ = torch.max(a, 1)  # MB x 1
    #
    #         # calculte log-sum-exp
    #         log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
    #
    #     else:
    #         raise Exception('Wrong name of the prior!')
    @classmethod
    def load(self, config, weights=None, device=None):
        '''Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        '''
        cfg = utils.load_pkl(config) if type(config) is str else config
        c = cfg['lattice_args']
        lat = lattice.Lattice(c['D'], extent=c['extent'], device=device)
        c = cfg['model_args']
        if c['enc_mask'] > 0:
            enc_mask = lat.get_circular_mask(c['enc_mask'])
            in_dim = int(enc_mask.sum())
        else:
            assert c['enc_mask'] == -1
            enc_mask = None
            in_dim = lat.D ** 2
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
        model = VAEvampprior(lat,
                           c['qlayers'], c['qdim'],
                           c['players'], c['pdim'],
                           in_dim, c['zdim'],
                           encode_mode=c['encode_mode'],
                           enc_mask=enc_mask,
                           enc_type=c['pe_type'],
                           enc_dim=c['pe_dim'],
                           domain=c['domain'],
                           activation=activation,
                           feat_sigma=c['feat_sigma'])
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt['model_state_dict'])
        if device is not None:
            model.to(device)
        return model, lat

        return log_prior

class VAEexemplar(HetOnlyVAE):
    # # AUXILIARY METHODS
    def __init__(self,
                     lattice,  # Lattice object
                     qlayers, qdim,
                     players, pdim,
                     in_dim, zdim=1,
                     encode_mode='resid',
                     enc_mask=None,
                     enc_type='linear_lowf',
                     enc_dim=None,
                     domain='fourier',
                     activation=nn.ReLU,
                     feat_sigma=None,
                     number_cachecomponents=5000,
                     approximate_k=10):
        super(VAEexemplar, self).__init__(lattice,  # Lattice object
                     qlayers, qdim,
                     players, pdim,
                     in_dim, zdim,
                     encode_mode,
                     enc_mask,
                     enc_type,
                     enc_dim,
                     domain,
                     activation,
                     feat_sigma)
        self.number_cachecomponents = number_cachecomponents
        self.approximate_k = approximate_k
        self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))
        # self.he_initializer()

    def he_initializer(self):
        print("he initializer")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)


    def q_z(self, *img, prior=False):
        img = (x.view(x.shape[0], -1) for x in img)
        if self.enc_mask is not None:
            img = (x[:, self.enc_mask] for x in img)
        z = self.encoder(*img)
        if prior is True:
            return z[:, :self.zdim], self.prior_log_variance * torch.ones_like(z[:, self.zdim:])
        else:
            return z[:, :self.zdim], z[:, self.zdim:]

    def log_p_z_exemplar(self, z, z_indices, exemplars_embedding):
        centers, center_log_variance, center_indices = exemplars_embedding
        denominator = torch.tensor(len(centers)).expand(len(z)).float().cuda()
        center_log_variance = center_log_variance[0, :].unsqueeze(0)
        prob, _ = log_normal_diag_vectorized(z, centers, center_log_variance)  # MB x C
        prob -= torch.log(denominator).unsqueeze(1)
        return prob

    def log_p_z(self, z, exemplars_embedding):
        z, z_indices = z
        prob = self.log_p_z_exemplar(z, z_indices, exemplars_embedding)
        prob_max, _ = torch.max(prob, 1)  # MB x 1
        log_prior = prob_max + torch.log(torch.sum(torch.exp(prob - prob_max.unsqueeze(1)), 1))  # MB x 1
        return log_prior


    # def get_exemplar_set(self, z_mean, z_log_var, dataset, cache, x_indices):
    #     exemplar_set = self.get_approximate_nearest_exemplars(
    #         z=(z_mean, z_log_var, x_indices),
    #         dataset=dataset,
    #         cache=cache)
    #     return exemplar_set
    #
    # def get_approximate_nearest_exemplars(self, z, cache, dataset):
    #     z, _, indices = z
    #     exemplars_indices = torch.randint(low=0, high=cache.shape[0],
    #                                       size=(self.args.number_cachecomponents, )).cuda()
    #
    #     # high = self.args.training_set_size
    #     cached_z, cached_log_variance = cache
    #     cached_z[indices.reshape(-1)] = z
    #     sub_cache = cached_z[exemplars_indices, :]
    #     _, nearest_indices = pairwise_distance(z, sub_cache) \
    #         .topk(k=self.args.approximate_k, largest=False, dim=1)
    #     nearest_indices = torch.unique(nearest_indices.view(-1))
    #     exemplars_indices = exemplars_indices[nearest_indices].view(-1)
    #     exemplars = dataset.tensors[0][exemplars_indices].cuda()
    #     exemplars_z, log_variance = self.q_z(exemplars, prior=True)
    #     cached_z[exemplars_indices] = exemplars_z
    #     exemplar_set = (exemplars_z, log_variance, exemplars_indices)
    #     return exemplar_set

    def get_exemplar_set(self, z_mean, z_log_var, cache, x_indices):
        exemplar_set = self.get_approximate_nearest_exemplars(
            z=(z_mean, z_log_var, x_indices),
            cache=cache)
        return exemplar_set

    def get_approximate_nearest_exemplars(self, z, cache):
        z, _, indices = z
        cached_z, cached_log_variance = cache
        device = torch.device('cuda')
        exemplars_indices = torch.randint(low=0, high=cached_z.shape[0],
                                          size=(self.number_cachecomponents, )).to(device)
        cached_z[indices.reshape(-1)] = z
        sub_cache = cached_z[exemplars_indices, :]
        _, nearest_indices = pairwise_distance(z, sub_cache) \
            .topk(k=self.approximate_k, largest=False, dim=1)
        nearest_indices = torch.unique(nearest_indices.view(-1))
        exemplars_indices = exemplars_indices[nearest_indices].view(-1)
        exemplars_z = cached_z[exemplars_indices]
        log_variance = cached_log_variance[exemplars_indices]
        exemplar_set = (exemplars_z, log_variance, exemplars_indices)
        return exemplar_set


    @classmethod
    def load(self, config, weights=None, device=None):
        '''Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        '''
        cfg = utils.load_pkl(config) if type(config) is str else config
        c = cfg['lattice_args']
        lat = lattice.Lattice(c['D'], extent=c['extent'], device=device)
        c = cfg['model_args']
        if c['enc_mask'] > 0:
            enc_mask = lat.get_circular_mask(c['enc_mask'])
            in_dim = int(enc_mask.sum())
        else:
            assert c['enc_mask'] == -1
            enc_mask = None
            in_dim = lat.D ** 2
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
        model = VAEexemplar(lat,
                           c['qlayers'], c['qdim'],
                           c['players'], c['pdim'],
                           in_dim, c['zdim'],
                           encode_mode=c['encode_mode'],
                           enc_mask=enc_mask,
                           enc_type=c['pe_type'],
                           enc_dim=c['pe_dim'],
                           domain=c['domain'],
                           activation=activation,
                           feat_sigma=c['feat_sigma'],
                           number_cachecomponents=c['number_cachecomponents'],
                           approximate_k=c['approximate_k'])
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt['model_state_dict'])
        if device is not None:
            model.to(device)
        return model, lat

    # def cache_z(self, dataset, prior=True, cuda=True):
    #     cached_z = []
    #     cached_log_var = []
    #     caching_batch_size = 10000
    #     num_batchs = math.ceil(len(dataset) / caching_batch_size)
    #     for i in range(num_batchs):
    #         if len(dataset[0]) == 3:
    #             batch_data, batch_indices, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]
    #         else:
    #             batch_data, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]
    #
    #         # exemplars_embedding, log_variance_z = self.q_z(batch_data.to(self.args.device), prior=prior)
    #         exemplars_embedding, log_variance_z = self.q_z(batch_data.cuda, prior=prior)
    #         cached_z.append(exemplars_embedding)
    #         cached_log_var.append(log_variance_z)
    #     cached_z = torch.cat(cached_z, dim=0)
    #     cached_log_var = torch.cat(cached_log_var, dim=0)
    #     cached_z = cached_z.cuda()
    #     cached_log_var = cached_log_var.cuda()
    #     return cached_z, cached_log_var



