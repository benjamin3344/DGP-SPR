# DGP-SPR: Deep generative priors for single-particle reconstruction. 

This repository shared the source code for the paper *"Deep Generative Priors for Biomolecular 3D Heterogeneous Reconstruction from Cryo-EM Projections"* accepted by the Journal of Structural Biology. Variational autoencoders (VAEs) with three types of deep generative priors were learned for latent variable inference and heterogeneous 3D reconstruction via Bayesian inference. More specifically, three priors were incoorperated into the backbone structures (CryoDRGN):  "Variational Mixture of Posteriors" priors (**VampPrior-SPR**), non-parametric exemplar-based priors (**ExemplarPrior-SPR**) and priors from latent score-based generative models (**LSGM-SPR**). Codes were adapted from the following repositories:

- CryoDRGN: [https://github.com/ml-struct-bio/cryodrgn](https://github.com/ml-struct-bio/cryodrgn) v1.0.0
- VAE-vampprior: [https://github.com/jmtomczak/vae_vampprior](https://github.com/jmtomczak/vae_vampprior)
- Exemplar-VAE: [https://github.com/sajadn/Exemplar-VAE](https://github.com/sajadn/Exemplar-VAE)
- LSGM: [https://github.com/NVlabs/LSGM](https://github.com/NVlabs/LSGM)


## Requirements

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

## Citation

Pending... waiting for production


## Acknowledgements

The project is supported by the Natural Sciences and Engineering Research Council of Canada (NSERC)’s Discovery Grant. The Open Centre for the Characterization of Advanced Materials (OCCAM) group is funded by the Canada Foundation for Innovation. This research was enabled in part by support provided by Compute Canada.  

To-do lists:
(1) License update  <br>
(2) Load "K" in VampPrior-SPR  <br>
(3) Merging different methods   <br>
(4) Read-me  <br>
