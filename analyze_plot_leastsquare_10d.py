import argparse
import os
import numpy as np
import os
import utils
log = utils.log
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy
import scipy.optimize
from matplotlib import cm
from cryodrgn import analysis
from cryodrgn.analysis import run_pca
import sklearn.metrics



def add_args(parser):
    parser.add_argument("zdir", type=os.path.abspath, help="Directory with ground-truth latent encodings")
    parser.add_argument("--mrcdir", type=os.path.abspath, help="Directory with ground-truth mrcs density maps")
    parser.add_argument('--mirror', action='store_true', help='Mirror for z-fitting')
    parser.add_argument('--datasets', type=str,  default='One$\_$rect', choices=('One$\_$rect','Two$\_$circles','Three$\_$rect'))
    return parser



def analyze_zN(z, z_color, outdir, skip_umap=False, num_pcs=2, num_ksamples=20): #vg
    zdim = z.shape[1]

    z_gt = z_color[0:2, :]
    z_color = z_color[2, :].astype('int')

    # Principal component analysis
    log('Perfoming principal component analysis...')
    # pc, pca, z_gt_reduced, z_gt_inv = run_pca(z, num_pcs, z_gt=z_gt)
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

    # average z
    states = np.unique(z_gt.T, axis=0)  # ...reordered
    num_states = states.shape[0]
    z_avg = np.mean(z.reshape((num_states, np.int(z.shape[0] / num_states), 10)), axis=1)
    log('average z in each cluster for volume generation and fsc calculation')
    np.savetxt(f'{outdir}/z_avg.txt', z_avg)

    z_gt_idx = np.mean(z_gt.T.reshape((num_states, np.int(z_gt.T.shape[0] / num_states), 2)), axis=1).astype(int)
    np.savetxt(f'{outdir}/z_idx.txt', z_gt_idx, fmt='%i')
    # in the case not int
    # z_gt_idx = np.mean(z_gt.T.reshape((num_states, np.int(z_gt.T.shape[0] / num_states), 2)), axis=1)
    # np.savetxt(f'{outdir}/z_idx.txt', z_gt_idx, fmt='%1.1f')


    # UMAP -- slow step
    if zdim > 2 and not skip_umap:
        log('Running UMAP...')
        umap_emb = analysis.run_umap(z)
        utils.save_pkl(umap_emb, f'{outdir}/umap.pkl')


    cmap_array = np.array([[86, 180, 233],  [166, 6, 40],    [122, 104, 166], [240, 228, 66],  [0, 158, 115],
                           [213, 94, 0],    [204, 121, 167], [155, 155, 41] ,[0, 114, 178], [52, 138, 189]])/255.0
    cmap = matplotlib.colors.ListedColormap(cmap_array)


    log('Generating plots...')
    plt.figure(1)
    sns.set(rc={'figure.facecolor': '#ffffff'})
    sns.set(rc={'axes.grid': True}, style='white', font_scale=1.5)
    g = sns.jointplot(x=pc[:, 0], y=pc[:, 1], alpha=.1, s=2)
    # g = sns.jointplot(x=pc[:, 0], y=pc[:, 1], alpha=.1, s=2, xlim=(-10,10), ylim=(-7,7))
    # g = sns.jointplot(x=pc[:, 0], y=pc[:, 1], alpha=.1, s=2, ylim=(-6,6.5))
    g.set_axis_labels('PC1', 'PC2', fontsize=20)
    # g.set_axis_labels('Principal Component 1', 'Principal Component 2', fontsize=20)
    g.ax_joint.scatter(pc[:, 0], pc[:, 1], c=np.squeeze(cmap(z_color[:])), alpha=.1, s=2)
    # g.ax_joint.set_xticklabels([])
    # g.ax_joint.set_yticklabels([])
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
        g.ax_joint.scatter(umap_emb[:, 0], umap_emb[:, 1], c=np.squeeze(cmap(z_color[:])), alpha=.1, s=2)
        # g.ax_joint.scatter(umap_emb[:, 0], umap_emb[:, 1], c=cmap[np.rint(z_color[:]).astype('int'), 0], alpha=.1, s=2)
        g.set_axis_labels('UMAP1', 'UMAP2', fontsize=20)
        # g.ax_joint.set_xticklabels([])
        # g.ax_joint.set_yticklabels([])
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


def zmap_error(x, grounds, zmaps):
    scale_x, scale_y, theta, xc, yc = x
    # apply affine matrix to each coordinate in grounds
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    trans_zmaps = (zmaps - np.array([xc, yc])) @ R * np.array([scale_x, scale_y])
    zmap_error = np.sqrt(np.sum(np.square(trans_zmaps - grounds), axis=1)).sum() / grounds.shape[0]
    return zmap_error


def analyze_zfit(z_pc, z_gt, z_color, mirror_flag, outdir):
    # initial condition
    x0 = 1.0, 1.0, math.pi / 2, 0, 0
    if mirror_flag:
        optimize = scipy.optimize.minimize(zmap_error, x0, args=(z_gt.T, z_pc[:, 0:2] * np.array([1, -1])),
                                           method='L-BFGS-B')
    else:
        optimize = scipy.optimize.minimize(zmap_error, x0, args=(z_gt.T, z_pc[:, 0:2] * np.array([1, 1])),
                                           method='L-BFGS-B')

    x = optimize.x
    scale_x, scale_y, theta, xc, yc = x
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    if mirror_flag:
        trans_zmaps = (z_pc[:, 0:2] * np.array([1, -1]) - np.array([xc, yc])) @ R * np.array([scale_x, scale_y])
    else:
        trans_zmaps = (z_pc[:, 0:2] * np.array([1, 1]) - np.array([xc, yc])) @ R * np.array([scale_x, scale_y])

    states = np.unique(z_gt.T, axis=0)  # ...reordered
    num_states = states.shape[0]

    states2 = np.mean(z_gt.T.reshape((num_states, np.int(z_gt.shape[1] / num_states), 2)), axis=1)
    states_mean = np.mean(trans_zmaps.reshape((num_states, np.int(z_gt.shape[1] / num_states), 2)), axis=1)
    # print(states2.shape, states_mean.shape, states2 + 1j * states_mean)

    states_errors = np.sqrt(np.sum(np.square(trans_zmaps - z_gt.T), axis=1))
    states_errors = np.mean(states_errors.reshape((num_states, np.int(z_gt.shape[1] / num_states))), axis=1)
    states_errors = states_errors

    with open(f'{outdir}/fit.txt', "a") as zfit:
        zfit.write('scale_x, scale_y, theta, xc, yc, errors')
        zfit.write(str(x))
        zfit.write(str(np.mean(states_errors)))


    # plotting fitting quiver plots
    sns.set(style='white')
    z_color = z_color[2, :].astype('int')
    cmap_array = np.array([[86, 180, 233],  [166, 6, 40],    [122, 104, 166], [240, 228, 66],  [0, 158, 115],
                           [213, 94, 0],    [204, 121, 167], [155, 155, 41] ,[0, 114, 178], [52, 138, 189]])/255.0
    cmap = matplotlib.colors.ListedColormap(cmap_array)


    fig, ax = plt.subplots(dpi=300)
    fig_qv = ax.quiver(states2[:, 0], states2[:, 1], states_mean[:, 0] - states2[:, 0],
                       states_mean[:, 1] - states2[:, 1], states_errors,
                       angles='xy', scale_units='xy', scale=1.0, cmap='viridis')
    ax.tick_params(labelsize=12)
    ax.locator_params(nbins=5)
    ax.set_aspect('equal', 'box')
    plt.xlim(-10, 90)
    plt.ylim(-10, 70)
    fig_qv.set_clim(0, 10)
    cbar = plt.colorbar(fig_qv, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    plt.scatter(trans_zmaps[:, 0], trans_zmaps[:, 1], s=1, alpha=0.05, c='gray')
    plt.scatter(z_gt.T[:, 0], z_gt.T[:, 1], s=5, alpha=0.5)
    # plt.scatter(states_mean[:,0], states_mean[:,1], s=5, alpha=0.5)
    # plt.grid()
    # plt.show()

    plt.xlabel('DoF2 ($^\circ$)', fontsize=16)
    plt.ylabel('DoF1 ($^\circ$)', fontsize=16)
    # plt.title('Three$\_$rect: 12.7924', fontsize=16)
    plt.title('One$\_$rect: 3.2598', fontsize=16)
    # plt.title('Two$\_$circles: 2.3925', fontsize=16)
    # plt.title('Three$\_$rect: 4.3750', fontsize=16)


    plt.savefig(f'{outdir}/z_fit.png')



    plt.figure('fitted_z_map')
    plt.xlim(-10, 90)
    plt.ylim(-10, 70)
    sns.set(rc={'figure.facecolor': '#ffffff'})
    g = sns.jointplot(trans_zmaps[:, 0], trans_zmaps[:, 1], alpha=.1, s=2)
    g.set_axis_labels('PC1', 'PC2')
    g.ax_joint.scatter(trans_zmaps[:, 0], trans_zmaps[:, 1], c=np.squeeze(cmap(z_color[:])), alpha=.1, s=2)

    plt.tight_layout()
    plt.savefig(f'{outdir}/z_fit2.png')

def analyze_zfit_leastsquare(z_mean, z_gt, z_color, mirror_flag, outdir):
    # initial condition

    z_gt = z_gt.T
    print(z_mean.shape, z_gt.shape)

    Ax = z_mean[:, 0:10]
    Ax = np.c_[Ax, np.ones(Ax.shape[0])]
    z_gt = np.c_[z_gt, np.zeros((z_gt.shape[0],8))]
    M = np.linalg.inv(Ax.T @ Ax) @ Ax.T @ z_gt
    trans_zmaps = Ax @ M


# pairwise distance matrix (20240111)
    pd = sklearn.metrics.pairwise_distances(trans_zmaps) # pairwise distances
    pd_gt = sklearn.metrics.pairwise_distances(z_gt)
    print(pd.shape, pd_gt.shape, pd, pd_gt)
    delta_pd = pd-pd_gt
    distance = np.linalg.norm(delta_pd, 'fro')


    z_gt = z_gt.T
    states = np.unique(z_gt.T, axis=0)  # ...reordered
    num_states = states.shape[0]

    states2 = np.mean(z_gt.T.reshape((num_states, np.int(z_gt.shape[1] / num_states), 10)), axis=1)
    states_mean = np.mean(trans_zmaps.reshape((num_states, np.int(z_gt.shape[1] / num_states), 10)), axis=1) # reshape to (ns, ni, 2)


    states_error_per_image = np.sum(np.square(trans_zmaps - z_gt.T), axis=1)
    states_variances = np.sum(
        np.square(trans_zmaps - np.repeat(states_mean, np.int(z_gt.shape[1] / num_states), axis=0)), axis=1)
    states_rmsd = np.sqrt(np.mean(np.square(trans_zmaps - z_gt.T))*10) #*2 for x, y seperation
    states_variances_per_state = np.mean(states_variances.reshape((num_states, np.int(z_gt.shape[1] / num_states))), axis=1)
    states_error_per_state = np.mean(states_error_per_image.reshape((num_states, np.int(z_gt.shape[1] / num_states))), axis=1)
    # states_errors = np.mean(states_errors.reshape((num_states, np.int(z_gt.shape[1] / num_states))), axis=1)
    states_rmsd_per_state = np.sqrt(states_error_per_state)
    states_errors = states_rmsd
    states_rmsd_variances = np.sqrt(np.mean(states_variances))
    states_rmsd_variances2 = np.sqrt(np.mean(np.square(trans_zmaps - np.repeat(states_mean, np.int(z_gt.shape[1] / num_states), axis=0)))*10)
    # examine equality bw variances and variance2 computed in different mathods
    print(states_rmsd_variances2, states_rmsd_variances)
    # states_rmsd_variances = np.sqrt(np.mean(states_variances))

    grid_deformation_rmsd = np.sqrt(np.mean(np.square(states_mean -  states2)*10))

    with open(f'{outdir}/fit.txt', "a") as zfit:
        zfit.write('least square coefficients\n')
        zfit.write(str(M))
        zfit.write('states_rmsd\n')
        zfit.write(str(states_rmsd))
        zfit.write('states_rmsd_variances\n')
        zfit.write(str(states_rmsd_variances))
        zfit.write('grid_deformation_rmsd\n')
        zfit.write(str(grid_deformation_rmsd))
        zfit.write('pairwise distance difference fro norm\n')
        zfit.write(str(distance))
        zfit.write('\n')


    # plotting fitting quiver plots
    sns.set(style='white')
    z_color = z_color[2, :].astype('int')
    cmap_array = np.array([[86, 180, 233],  [166, 6, 40],    [122, 104, 166], [240, 228, 66],  [0, 158, 115],
                           [213, 94, 0],    [204, 121, 167], [155, 155, 41] ,[0, 114, 178], [52, 138, 189]])/255.0
    cmap = matplotlib.colors.ListedColormap(cmap_array)


    fig, ax = plt.subplots(dpi=300)
    fig_qv = ax.quiver(states2[:, 0], states2[:, 1], states_mean[:, 0] - states2[:, 0],
                       states_mean[:, 1] - states2[:, 1], np.sqrt(states_variances_per_state),
                       angles='xy', scale_units='xy', scale=1.0, cmap='viridis')
    ax.tick_params(labelsize=12)
    ax.locator_params(nbins=5)
    ax.set_aspect('equal', 'box')
    plt.xlim(-10, 90)
    plt.ylim(-10, 70)
    fig_qv.set_clim(0, 10)
    cbar = plt.colorbar(fig_qv, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    plt.scatter(trans_zmaps[:, 0], trans_zmaps[:, 1], s=1, alpha=0.05, c='gray')
    plt.scatter(z_gt.T[:, 0], z_gt.T[:, 1], s=5, alpha=0.5)
    # plt.scatter(states_mean[:,0], states_mean[:,1], s=5, alpha=0.5)
    # plt.grid()
    # plt.show()

    plt.xlabel('DoF2 ($^\circ$)', fontsize=16)
    plt.ylabel('DoF1 ($^\circ$)', fontsize=16)
    # plt.title('Three$\_$rect: 12.7924', fontsize=16)
    plt.title( args.datasets + ": " + "{:.4f}".format(grid_deformation_rmsd) + '/' + "{:.4f}$^2$".format(states_rmsd_variances), fontsize=16)
    # plt.title('Two$\_$circles: 2.3925', fontsize=16)
    # plt.title('Three$\_$rect: 4.3750', fontsize=16)


    plt.savefig(f'{outdir}/z_fit.png')



    plt.figure('fitted_z_map')
    plt.xlim(-10, 90)
    plt.ylim(-10, 70)
    sns.set(rc={'figure.facecolor': '#ffffff'})
    g = sns.jointplot(trans_zmaps[:, 0], trans_zmaps[:, 1], alpha=.1, s=2)
    g.set_axis_labels('PC1', 'PC2')
    g.ax_joint.scatter(trans_zmaps[:, 0], trans_zmaps[:, 1], c=np.squeeze(cmap(z_color[:])), alpha=.1, s=2)

    plt.tight_layout()
    plt.savefig(f'{outdir}/z_fit2.png')


def analyze_zfit2(z_pc, z_gt, z_color, mirror_flag, outdir):
    # for 3dflex
    # initial condition
    x0 = 1.0, 1.0, math.pi / 2, 0, 0
    if mirror_flag:
        optimize = scipy.optimize.minimize(zmap_error, x0, args=(z_gt.T, z_pc[:, 0:2] * np.array([1, -1])),
                                           method='L-BFGS-B')
    else:
        optimize = scipy.optimize.minimize(zmap_error, x0, args=(z_gt.T, z_pc[:, 0:2] * np.array([1, 1])),
                                           method='L-BFGS-B')

    x = optimize.x
    scale_x, scale_y, theta, xc, yc = x
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    if mirror_flag:
        trans_zmaps = (z_pc[:, 0:2] * np.array([1, -1]) - np.array([xc, yc])) @ R * np.array([scale_x, scale_y])
    else:
        trans_zmaps = (z_pc[:, 0:2] * np.array([1, 1]) - np.array([xc, yc])) @ R * np.array([scale_x, scale_y])

    states = np.unique(z_gt.T, axis=0)  # ...reordered
    num_states = states.shape[0]

    states2 = np.mean(z_gt.T.reshape((num_states, np.int(z_gt.shape[1] / num_states), 2)), axis=1)
    states_mean = np.mean(trans_zmaps.reshape((num_states, np.int(z_gt.shape[1] / num_states), 2)), axis=1)
    # print(states2.shape, states_mean.shape, states2 + 1j * states_mean)

    states_errors = np.sqrt(np.sum(np.square(trans_zmaps - z_gt.T), axis=1))
    states_errors = np.mean(states_errors.reshape((num_states, np.int(z_gt.shape[1] / num_states))), axis=1)
    states_errors = states_errors

    with open(f'{outdir}/fit.txt', "a") as zfit:
        zfit.write('scale_x, scale_y, theta, xc, yc, errors')
        zfit.write(str(x))
        zfit.write(str(np.mean(states_errors)))


    # plotting fitting quiver plots
    sns.set(style='white')
    z_color = z_color[2, :].astype('int')
    cmap_array = np.array([[86, 180, 233],  [166, 6, 40],    [122, 104, 166], [240, 228, 66],  [0, 158, 115],
                           [213, 94, 0],    [204, 121, 167], [155, 155, 41] ,[0, 114, 178], [52, 138, 189]])/255.0
    cmap = matplotlib.colors.ListedColormap(cmap_array)


    fig, ax = plt.subplots(dpi=300)
    fig_qv = ax.quiver(states2[:, 0], states2[:, 1], states_mean[:, 0] - states2[:, 0],
                       states_mean[:, 1] - states2[:, 1], states_errors,
                       angles='xy', scale_units='xy', scale=1.0, cmap='viridis')
    ax.tick_params(labelsize=12)
    ax.locator_params(nbins=5)
    ax.set_aspect('equal', 'box')
    plt.xlim(-10, 90)
    plt.ylim(-10, 70)
    fig_qv.set_clim(0, 10)
    cbar = plt.colorbar(fig_qv, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    plt.scatter(trans_zmaps[:, 0], trans_zmaps[:, 1], s=1, alpha=0.05, c='gray')
    plt.scatter(z_gt.T[:, 0], z_gt.T[:, 1], s=5, alpha=0.5)
    # plt.scatter(states_mean[:,0], states_mean[:,1], s=5, alpha=0.5)
    # plt.grid()
    # plt.show()

    plt.xlabel('DoF2 ($^\circ$)', fontsize=16)
    plt.ylabel('DoF1 ($^\circ$)', fontsize=16)
    plt.title('One$\_$rect: 3.2598', fontsize=16)
    plt.savefig(f'{outdir}/z_fit.png')



    plt.figure('fitted_z_map')
    plt.xlim(-10, 90)
    plt.ylim(-10, 70)
    sns.set(rc={'figure.facecolor': '#ffffff'})
    g = sns.jointplot(trans_zmaps[:, 0], trans_zmaps[:, 1], alpha=.1, s=2)
    g.set_axis_labels('PC1', 'PC2')
    g.ax_joint.scatter(trans_zmaps[:, 0], trans_zmaps[:, 1], c=np.squeeze(cmap(z_color[:])), alpha=.1, s=2)

    plt.tight_layout()
    plt.savefig(f'{outdir}/z_fit2.png')


def main(args):

    # 1. latent vector encoding z: pca and visualization
    zdir = args.zdir
    zfile = os.path.join(args.zdir, 'zzcolor.pkl')

    with open(zfile, "rb") as zz:
        z_mean = pickle.load(zz)
        z_var = pickle.load(zz)
        z_color = pickle.load(zz)
    log('Latent encoding pca and visualization...')
    pca = analyze_zN(z_mean, z_color, zdir, skip_umap=False, num_pcs=2, num_ksamples=20)

    # 2. fitting z to the ground-truth rotational angles using an affine transformation
    z_pcfile = os.path.join(args.zdir, 'z_pc.pkl')  #
    with open(z_pcfile, "rb") as zz:
        z_pc = pickle.load(zz)
        z_gt = pickle.load(zz)
    log('Latent encoding fitting...')
    if not os.path.exists(f'{zdir}/z_fitting'):
        os.mkdir(f'{zdir}/z_fitting')
    # analyze_zfit_leastsquare(z_pc, z_gt, z_color, args.mirror, f'{zdir}/z_fitting')
    analyze_zfit_leastsquare(z_mean, z_gt, z_color, args.mirror, f'{zdir}/z_fitting')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='__doc__')
    args = add_args(parser).parse_args()
    main(args)

