# DGP-SPR: Deep generative priors for single-particle reconstruction. 

This repository shared the source code for the paper *"Deep Generative Priors for Biomolecular 3D Heterogeneous Reconstruction from Cryo-EM Projections"* accepted by the Journal of Structural Biology. Variational autoencoders (VAEs) with three types of deep generative priors were learned for latent variable inference and heterogeneous 3D reconstruction via Bayesian inference. More specifically, three priors were incoorperated into the backbone structures (CryoDRGN):  "Variational Mixture of Posteriors" priors (**VampPrior-SPR**), non-parametric exemplar-based priors (**ExemplarPrior-SPR**) and priors from latent score-based generative models (**LSGM-SPR**). Codes were adapted from the following repositories:

- CryoDRGN: [https://github.com/ml-struct-bio/cryodrgn](https://github.com/ml-struct-bio/cryodrgn) v1.0.0b0
- VAE-vampprior: [https://github.com/jmtomczak/vae_vampprior](https://github.com/jmtomczak/vae_vampprior)
- Exemplar-VAE: [https://github.com/sajadn/Exemplar-VAE](https://github.com/sajadn/Exemplar-VAE)
- LSGM: [https://github.com/NVlabs/LSGM](https://github.com/NVlabs/LSGM)


## Requirements

- python 3.7
- torch 1.8.0
- torchdiffeq 0.2.2 
- please check example_scripts for other libraries

## Data

The experiments can be run on cryo-EM datasets with extracted particle stacks (.mrc or .mrcs), ctf and poses estimation (.pkl). Please check CryoDRGN([https://github.com/ml-struct-bio/cryodrgn](https://github.com/ml-struct-bio/cryodrgn)) for pre-processing. The commonly-used experimental datasets include:

- EMPIAR-10076: 50S ribosome assembly
- EMPIAR-10180: Pre-catalytic spliceosome
- other datasets used in CryoDRGN and 3DVA (https://zenodo.org/records/4355284)

New simulated and experimental datasets built in this paper include:

- Discrete states: one_rect, two_circles, three_rect ([https://zenodo.org/deposit/7557414](https://zenodo.org/deposit/7557414))
- Continuous datasets: three_ovals ([https://zenodo.org/deposit/10574539](https://zenodo.org/deposit/10574539))
- EMPIAR-11487/11473/11461: cowpea chlorotic mottle virus (CCMV) capsid  ([https://zenodo.org/deposit/10574539](https://zenodo.org/deposit/10574539))



## Models 

### CryoDRGN

train_vae.py is a simplified implementation of CryoDRGN. As an alternative way, you can download the most up-to-date CryoDRGN.
```
$ python train_vae.py $DATADIR/data/L17Combine_weight_local.mrcs --poses $DATADIR/data/pose.pkl --ctf $DATADIR/data/ctf.pkl --zdim 16 -n 100 --enc-dim 1024 --enc-layers 3 --dec-dim 1024 --dec-layers 3 --amp --uninvert-data --root $RESULT --ind $DATADIR/intersection.96393.pkl --lazy
```

### VampPrior-SPR
Except for the *--number-components*, you can also specify *--pseudoinputs-mean* and *--pseudoinputs-std* for the initiallization of pseudo inputs.

```
$ python train_vampprior.py $DATADIR/particles.256.mrcs --poses $DATADIR/pose.pkl --ctf $DATADIR/ctf.pkl --zdim 10 -n 101 -b 8  --enc-dim 256 --enc-layers 3 --dec-dim 256 --dec-layers 3 --amp --lazy --lr 0.0001 --root $RESULT --save 'exp_vampprior' --number-components 50 --checkpoint 5
```



### ExemplarPrior-SPR
In ExemplarPrior-SPR, *--number-components* neighboring exemplars in the latent space were used to calculated the priors for regularization. The cache table was updated after each batch. Time complexity will be reduced if the cache table was updated less frequently (we will upload the codes later). 
```
$ python train_exemplar.py $DATADIR/particles.256.mrcs  --poses $DATADIR/pose.pkl --ctf $DATADIR/ctf.pkl --zdim 10 -n 101 --root $RESULT --save 'exp_exemplar'  --enc-dim 256 --enc-layers 3 --dec-dim 256 --dec-layers 3  --amp --lazy --lr 0.0001 --beta 1 --checkpoint 5 --batch-size 8 --prior 'exemplar' --number-cachecomponents 5000 --approximate-prior --log-interval 10000
```


### LSGM-SPR

```
$ python train_lsgm.py $DATADIR/particles.256.txt --poses $DATADIR/pose.pkl --ctf $DATADIR/ctf.pkl --zdim 10 --epochs 101 --root $RESULT --save 'exp_lsgm_5e-4' --vada_checkpoint $RESULT/2/exp_lsgm_5e-4/checkpoint.pt --cont_training --enc-dim 256 --enc-layers 3 --dec-dim 256 --dec-layers 3 --dropout 0.1 --batch_size 8 --num_scales_dae 2 --weight_decay_norm_vae 1e-2 --weight_decay_norm_dae 0. --num_channels_dae 8 --train_vae --num_cell_per_scale_dae 1 --learning_rate_dae 3e-4 --learning_rate_min_dae 3e-4 --train_ode_solver_tol 1e-5  --sde_type vpsde --iw_sample_p ll_iw --num_process_per_node 1 --dae_arch ncsnpp --embedding_scale 50 --mixing_logit_init -3 --warmup_epochs 0 --lazy --disjoint_training --iw_sample_q ll_iw --iw_sample_p drop_sigma2t_iw --embedding_dim 64 --beta 0.1 --learning_rate_vae 5e-4 --weight_decay 0 --uninvert-data
```


### Evaluation
```

```

### Analyze

Ongoing....
Different codes were used to analyze simulated datasets with continuous and discrete states. For datasets <br>

For continuous states, the ground-truth angle_of_rotation.txt and learned z.100.pkl are required.
```
python analyze/analyze_plot_leastsquare_continuous_10d_paper.py /path/to/folder/with_result_z100pkl --methods 'exemplar'
```

For discrete states, zzcolor.pkl are required which is a pkl file which appended the ground-truth color to the z.100.pkl...... pending
```
python analyze/analyze_plot_leastsquare_10d.py /path/to/folder/with_zzcolor_pkl 
```

## Citation

Cite our paper using the following bibtex item:
```
@article{shi2024deep,
  title={Deep Generative Priors for Biomolecular 3D Heterogeneous Reconstruction from Cryo-EM Projections},
  author={Shi, Bin and Zhang, Kevin and Fleet, David J and McLeod, Robert A and Miller, RJ Dwayne and Howe, Jane Y},
  journal={Journal of Structural Biology},
  pages={108073},
  year={2024},
  publisher={Elsevier}
}
111
Or: <br>
```
Shi, Bin, et al. "Deep Generative Priors for Biomolecular 3D Heterogeneous Reconstruction from Cryo-EM Projections." Journal of Structural Biology (2024): 108073.
```


## Acknowledgements

The project is supported by the Natural Sciences and Engineering Research Council of Canada (NSERC)’s Discovery Grant. The Open Centre for the Characterization of Advanced Materials (OCCAM) group is funded by the Canada Foundation for Innovation. This research was enabled in part by support provided by Compute Canada.  

To-do lists:
(2) Load "K" in VampPrior-SPR  <br>
(3) Merging different methods   <br>
(4) Read-me  <br>
(5) Analyze codes were uploaded and would be re-organized later. <br>
(6) Add __init__ file for subfolders for importing libraries.