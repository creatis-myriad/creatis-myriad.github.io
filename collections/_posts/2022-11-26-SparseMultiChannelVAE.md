---
layout: review
title: "Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data"
tags: VAE, Sparsity, Heterogeneous, Data
author: "Olivier Bernard"
cite:
    authors: "Luigi Antelmi, Nicholas Ayache, Philippe Robert, Marco Lorenzi"
    title:   "Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data"
    venue:   "International Conference on Machine Learning (ICML) 2019"
pdf: "http://proceedings.mlr.press/v97/antelmi19a/antelmi19a.pdf"
---

# Notes

* Here are some (highly) useful links: [repo](https://github.com/ggbioing/mcvae), [video - slider on 1h05](https://youtube.videoken.com/embed/n5e2qNQ-h6E?tocitem=67), [slides](https://icml.cc/media/Slides/icml/2019/hallb(12-16-00)-12-17-00-5118-sparse_multi-ch.pdf)
* This post has been jointly created by Olivier Bernard, Romain Deleat-Besson and Nathan Painchaud.

# Highlights

* The objective of this paper is to develop a formalism to handle with heterogenous data through VAE formulation.
* This is achieved through two major innovations: i) Variational distribution of each channel/modality/type of data are constrained to a common target prior in the latent space to bring interpretability; ii) Parsimonious latent representations are enforced by variational dropout to make the method computationally advantageous and more easily interpretable.


# Method

&nbsp;

# Appendix

## KL-divergence of the proposed model



