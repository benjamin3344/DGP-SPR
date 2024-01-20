import cryodrgn.dataset as dataset
import pickle
import cryodrgn.utils
import torch
from cryodrgn.pose import PoseTracker
import cryodrgn.ctf as ctf


log = cryodrgn.utils.log

#def get_loaders(particles, norm, invert_data, ind, keepreal):
def get_loaders(args):

    # load index filter
    if args.ind is not None:
        log('Filtering image dataset with {}'.format(args.ind))
        ind = pickle.load(open(args.ind, 'rb'))
    else: ind = None

    # load datasets
    if args.lazy:
        data = dataset.LazyMRCData(args.particles, norm=args.norm, invert_data=args.invert_data, ind=ind,
                                   keepreal=args.use_real, window=args.window, datadir=args.datadir,
                                   window_r=args.window_r)
    elif args.preprocessed:
        log(f'Using preprocessed inputs. Ignoring any --window/--invert-data options')
        data = dataset.PreprocessedMRCData(args.particles, norm=args.norm, ind=ind)
    else:
        data = dataset.MRCData(args.particles, norm=args.norm, invert_data=args.invert_data, ind=ind,
                               keepreal=args.use_real, window=args.window, datadir=args.datadir,

                               max_threads=args.max_threads, window_r=args.window_r)

    device = torch.device('cuda')
    Nimg = data.N
    D = data.D

    # load poses
    if args.do_pose_sgd: assert args.domain == 'hartley', "Need to use --domain hartley if doing pose SGD"
    do_pose_sgd = args.do_pose_sgd
    posetracker = PoseTracker.load(args.poses, Nimg, D, 's2s2' if do_pose_sgd else None, ind, device=device)

    # load ctf
    if args.ctf is not None:
        if args.use_real:
            raise NotImplementedError("Not implemented with real-space encoder. Use phase-flipped images instead")
        log('Loading ctf params from {}'.format(args.ctf))
        ctf_params = ctf.load_ctf_for_training(D-1, args.ctf)
        if args.ind is not None: ctf_params = ctf_params[ind]
        assert ctf_params.shape == (Nimg, 8)
        ctf_params = torch.tensor(ctf_params, device=device)
    else: ctf_params = None

    return data, posetracker, ctf_params

