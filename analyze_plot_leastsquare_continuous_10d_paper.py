'''
Codes for analyzing continuous datasets, e.g., CCMV:
Input: evaluated latent encodings z.100.pkl
Output: PCA-map, U-map, fit.txt, z_fit.png

Function 'analyze_zN' adapted from CryoDRGN
'''

import argparse
import os
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import utils
log = utils.log
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from cryodrgn import analysis
from cryodrgn.analysis import run_pca
import torch, io
import sklearn.metrics


def add_args(parser):
    parser.add_argument("zdir", type=os.path.abspath, help="Directory with ground-truth latent encodings")
    parser.add_argument("--mrcdir", type=os.path.abspath, help="Directory with ground-truth mrcs density maps")
    parser.add_argument('--datasets', type=str,  default='One$\_$rect', choices=('One$\_$rect','Two$\_$circles','Three$\_$rect','continuous'))
    parser.add_argument('--models', type=str,  default='cryodrgn', choices=('cryodrgn','lsgmprior','vampprior','exemplar'))
    parser.add_argument('--methods', type=str,  default='ls_10d', choices=('ls_10d','ls_2d'))

    return parser



def analyze_zN(z, z_gt, outdir, skip_umap=False, num_pcs=2, num_ksamples=20):
    zdim = z.shape[1]

    # Principal component analysis
    log('Perfoming principal component analysis...')
    pc, pca = run_pca(z)

    # kmeans clustering
    log('K-means clustering...')
    K = num_ksamples
    kmeans_labels, centers = analysis.cluster_kmeans(z, K)
    centers, centers_ind = analysis.get_nearest_point(z, centers)
    if not os.path.exists(f'{outdir}/kmeans{K}'):
        os.mkdir(f'{outdir}/kmeans{K}')
    utils.save_pkl(kmeans_labels, f'{outdir}/kmeans{K}/labels.pkl')
    np.savetxt(f'{outdir}/kmeans{K}/centers.txt', centers)
    np.savetxt(f'{outdir}/kmeans{K}/centers_ind.txt', centers_ind, fmt='%d')


    # UMAP -- slow step
    if zdim > 2 and not skip_umap:
        log('Running UMAP...')
        umap_emb = analysis.run_umap(z)
        utils.save_pkl(umap_emb, f'{outdir}/umap.pkl')


    log('Generating plots...')
    plt.figure(1)
    sns.set(rc={'figure.facecolor': '#ffffff'})
    sns.set(rc={'axes.grid': True}, style='white', font_scale=1.5)
    g = sns.jointplot(x=pc[:, 0], y=pc[:, 1], alpha=.1, s=2)
    g.set_axis_labels('PC1', 'PC2', fontsize=20)

    # scatter plot with the same color
    # g.ax_joint.scatter(pc[:, 0], pc[:, 1], alpha=.1, s=2)
    # scatter plot with its ground-truth labelling color
    index = np.loadtxt('three_ovals_index_reverse.txt').astype(int)
    index1 = index[0:10000]
    g.ax_joint.scatter(pc[index1, 0], pc[index1, 1], alpha=.1, s=2)
    index2 = index[10000:25000]
    g.ax_joint.scatter(pc[index2, 0], pc[index2, 1], alpha=.1, s=2)
    index3 = index[25000:50000]
    g.ax_joint.scatter(pc[index3, 0], pc[index3, 1], alpha=.1, s=2)


    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{outdir}/z_pca.png')

    plt.figure(2)
    g = sns.jointplot(x=pc[:, 0], y=pc[:, 1], kind='hex')
    g.set_axis_labels('PC1', 'PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/z_pca_hexbin.png')

    out_z = '{}/z_pc.pkl'.format(outdir)
    with open(out_z, 'wb') as f:
        pickle.dump(pc, f)
        pickle.dump(z_gt, f)


    if zdim > 2 and not skip_umap:
        plt.figure(3)
        sns.set(rc={'axes.grid': True}, style='white', font_scale=1.5)
        g = sns.jointplot(x=umap_emb[:, 0], y=umap_emb[:, 1], alpha=.1, s=2)
        g.ax_joint.scatter(umap_emb[:, 0], umap_emb[:, 1], alpha=.1, s=2)
        # g.ax_joint.scatter(umap_emb[:, 0], umap_emb[:, 1], c=cmap[np.rint(z_color[:]).astype('int'), 0], alpha=.1, s=2)
        g.set_axis_labels('UMAP1', 'UMAP2', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{outdir}/umap.png')

        plt.figure(4)
        g = sns.jointplot(x=umap_emb[:, 0], y=umap_emb[:, 1], kind='hex')
        g.set_axis_labels('UMAP1', 'UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/umap_hexbin.png')

    analysis.scatter_annotate(pc[:, 0], pc[:, 1], centers_ind=centers_ind, annotate=True)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(f'{outdir}/kmeans{K}/z_pca.png')

    g = analysis.scatter_annotate_hex(pc[:, 0], pc[:, 1], centers_ind=centers_ind, annotate=True)
    g.set_axis_labels('PC1', 'PC2')
    plt.tight_layout()
    plt.savefig(f'{outdir}/kmeans{K}/z_pca_hex.png')

    if zdim > 2 and not skip_umap:
        analysis.scatter_annotate(umap_emb[:, 0], umap_emb[:, 1], centers_ind=centers_ind, annotate=True)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.savefig(f'{outdir}/kmeans{K}/umap.png')

        g = analysis.scatter_annotate_hex(umap_emb[:, 0], umap_emb[:, 1], centers_ind=centers_ind, annotate=True)
        g.set_axis_labels('UMAP1', 'UMAP2')
        plt.tight_layout()
        plt.savefig(f'{outdir}/kmeans{K}/umap_hex.png')

    for i in range(num_pcs):
        if zdim > 2 and not skip_umap:
            analysis.scatter_color(umap_emb[:, 0], umap_emb[:, 1], pc[:, i], label=f'PC{i + 1}')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.tight_layout()

    return pca



def analyze_zfit_leastsquare_10d(z, z_gt, outdir, index=None):
    # initial condition
    Ax = z[:, 0:10]
    z_gt = np.c_[z_gt, np.zeros((z_gt.shape[0],8))]


    if index is not None:
        # index_fit = index[10000:]
        index_fit = index
        Ax2 = z[index_fit, :]
        Ax2 = np.c_[Ax2, np.ones(Ax2.shape[0])]
        M = np.linalg.inv(Ax2.T @ Ax2) @ Ax2.T @ z_gt[index_fit, :]
        Ax = np.c_[Ax, np.ones(Ax.shape[0])]
    else:
        Ax = np.c_[Ax, np.ones(Ax.shape[0])]
        M = np.linalg.inv(Ax.T @ Ax) @ Ax.T @ z_gt
    trans_zmaps = Ax @ M
    print(z.shape, z_gt.shape, Ax.shape, trans_zmaps.shape)
    print(z[:, 0:10], z_gt, trans_zmaps)


    states_rmsd = np.sqrt(np.mean(np.square(trans_zmaps - z_gt)) * 10)  # *10 for x, y, 10-d seperation
    pd = sklearn.metrics.pairwise_distances(trans_zmaps) # pairwise distances
    pd_gt = sklearn.metrics.pairwise_distances(z_gt)
    print(pd.shape, pd_gt.shape, pd, pd_gt)
    delta_pd = pd-pd_gt
    distance = np.linalg.norm(delta_pd, 'fro')

    with open(f'{outdir}/fit.txt', "a") as zfit:
        zfit.write('least square coefficients\n')
        zfit.write(str(M))
        zfit.write('states_rmsd\n')
        zfit.write(str(states_rmsd))
        zfit.write('pairwise distance difference fro norm\n')
        zfit.write(str(distance))
        zfit.write('\n')

    plt.figure('fitted_z_map')
    g = plt.subplots()
    sns.set(rc={'axes.grid': True}, style='white', font_scale=1.5)
    g = sns.jointplot(z_gt[:, 1], z_gt[:, 0], kind='kde', xlim=(-10, 100), ylim=(0, 70), shade=True, thresh=0.05,
                      fill=True, color='gray', marginal_kws={"color": "gray", "alpha": .2}, alpha=.5)

    # random permutation of scatter points
    # otherwise points with smaller index, e.g.,0:10000 will be covered by those with larger index, e.g., 25000:50000
    indd = np.random.permutation(50000)
    g.x = trans_zmaps[index[indd], 1]
    g.y = trans_zmaps[index[indd], 0]
    classes = np.zeros(50000).astype(int)
    classes[0:10000] = 0
    classes[10000:25000] = 1
    classes[25000:50000] = 2

    colormap = np.array(['#1f77b4', '#ff7f0e', '#2ca02c'])
    g.plot_joint(sns.scatterplot, alpha=.1, s=2, c=colormap[classes[indd]])
    g.plot_marginals(sns.kdeplot, color='b', fill=True)
    g.set_axis_labels('DoF2 ($^\circ$)', 'DoF1 ($^\circ$)')
    g.ax_joint.text(50, 61, 'ExemplarPrior-SPR', fontsize=12)


    # g.ax_joint.scatter(trans_zmaps[:, 1], trans_zmaps[:, 0], alpha=.1, s=2)

    # index = np.loadtxt('three_ovals_index_reverse.txt').astype(int)
    # index1 = index[0:10000]
    # g.ax_joint.scatter(trans_zmaps[index1, 1], trans_zmaps[index1, 0], alpha=.1, s=2)
    # index2 = index[10000:25000]
    # g.ax_joint.scatter(trans_zmaps[index2, 1], trans_zmaps[index2, 0], alpha=.1, s=2)
    # index3 = index[25000:50000]
    # g.ax_joint.scatter(trans_zmaps[index3, 1], trans_zmaps[index3, 0], alpha=.1, s=2)


    # classes = np.zeros(50000)
    # classes[0:10000] = 0
    # classes[10000:25000] = 2
    # classes[25000:50000] = 1
    # g = sns.jointplot(z_gt[:, 1], z_gt[:, 0], alpha=.1, s=2)
    # g.set_axis_labels('PC1', 'PC2')

    # g.ax_joint.scatter(z_gt[:, 1], z_gt[:, 0], c='k', alpha=.02, s=2)


    # plt.xlim(-10, 70)
    # plt.ylim(-10, 90)

    plt.tight_layout()
    plt.savefig(f'{outdir}/z_fit2.png')






class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def main(args):

    # 1. latent vector encoding z: pca and visualization
    zdir = args.zdir
    z_gt = np.loadtxt('three_oval_formal.txt')
    z_gt[15253, :] = z_gt[15254, :]
    z_gt[:, [0, 1]] = z_gt[:, [1, 0]]

    index = np.loadtxt('three_ovals_index_reverse.txt').astype(int)
    print(z_gt[15252:15255, :], z_gt.shape)

    if args.methods == 'cryodrgn' or args.methods == 'vampprior':
        zfile = os.path.join(args.zdir, 'z.100.pkl')
        with open(zfile, "rb") as zz:
            z_mean = pickle.load(zz)
            z_var = pickle.load(zz)
    elif args.methods == 'exemplar':
        zfile = os.path.join(args.zdir, 'z.100.pkl')
        with open(zfile, "rb") as zz:
            z_mean = CPU_Unpickler(zz).load().numpy()
    elif args.methods == 'lsgmprior':
        zfile = os.path.join(args.zdir, 'zz.100.pkl')
        with open(zfile, "rb") as zz:
            z_mean = pickle.load(zz)
    else:
        raise Exception("Unknown models claimed")

    log('Latent encoding pca and visualization...')
    pca = analyze_zN(z_mean, z_gt, zdir, skip_umap=False, num_pcs=2, num_ksamples=20)


    # 2. fitting z to the ground-truth rotational angles using an affine transformation
    # z_pcfile = os.path.join(args.zdir, 'z_pc.pkl')  #
    # with open(z_pcfile, "rb") as zz:
    #     z_pc = pickle.load(zz)
    #     z_gt = pickle.load(zz)
    log('Latent encoding fitting...')
    if not os.path.exists(f'{zdir}/z_fitting'):
        os.mkdir(f'{zdir}/z_fitting')

    analyze_zfit_leastsquare_10d(z_mean, z_gt, f'{zdir}/z_fitting', index=index)


    # 3. eval_vol and fsc calculation
    log('FSC calculation...')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='__doc__')
    args = add_args(parser).parse_args()
    main(args)

