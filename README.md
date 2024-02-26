# DGP-SPR: Deep generative priors for single-particle reconstruction. 

This repository shared the source code for the paper *"Deep Generative Priors for Biomolecular 3D Heterogeneous Reconstruction from Cryo-EM Projections"* accepted by the Journal of Structural Biology. Variational autoencoders (VAEs) with three types of deep generative priors were learned for latent variable inference and heterogeneous 3D reconstruction via Bayesian inference. More specifically, three priors were incoorperated into the backbone structures (CryoDRGN):  "Variational Mixture of Posteriors" priors (VampPrior-SPR), non-parametric exemplar-based priors (ExemplarPrior-SPR) and priors from latent score-based generative models (LSGM-SPR). Codes were adapted from the following repositories:

- CryoDRGN: [https://github.com/ml-struct-bio/cryodrgn](https://github.com/ml-struct-bio/cryodrgn) v1.0.0
- VAE-vampprior: [https://github.com/jmtomczak/vae_vampprior](https://github.com/jmtomczak/vae_vampprior)
- Exemplar-VAE: [https://github.com/sajadn/Exemplar-VAE](https://github.com/sajadn/Exemplar-VAE)
- LSGM: [https://github.com/NVlabs/LSGM](https://github.com/NVlabs/LSGM)


To-do lists:
(1) License update  <br>
(2) Load "K" in VampPrior-SPR  <br>
(3) Merging different methods   <br>
(4) Read-me  <br>
