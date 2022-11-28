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

$$\mathbb{E}_{c}[D_{KL}\left( q(z \vert x_c) \parallel p(z \vert x_1,\cdots,x_C)\right)]$$

$$=\mathbb{E}_{c}[D_{KL}( q(z \vert x_c) \parallel p(z \vert x))]$$

$$=\mathbb{E}_{c}\left[\int_{z}{q(z \vert x_c)\cdot \left[ log\left(q(z \vert x_c)\right) - log\left(p(z \vert x)\right) \right] \, dz}\right]$$

$$\text{using} \quad p(z \vert x) = \frac{p(x \vert z) \cdot p(z)}{p(x)}$$

$$=\mathbb{E}_{c}\left[\int_{z}{q(z \vert x_c)\cdot \left[ log\left(q(z \vert x_c)\right) - log\left(p(x \vert z)\right) - log\left(p(z)\right) + log\left(p(x)\right) \right] \, dz}\right]$$

$$=\underbrace{\mathbb{E}_{c}\left[\int_{z}{q(z \vert x_c)\cdot log\left(p(x)\right)\,dz}\right]}_{Eq_1} + \underbrace{\mathbb{E}_{c}\left[\int_{z}{q(z \vert x_c)\cdot \left[ log\left(q(z \vert x_c)\right) - log\left(p(z)\right) \right] \, dz}\right]}_{Eq_2}-\underbrace{\mathbb{E}_{c}\left[\int_{z}{q(z \vert x_c)\cdot log\left(p(x \vert z)\right)\,dz}\right]}_{Eq_3}$$

&nbsp;

$$Eq_1 = \mathbb{E}_{c}[log\left(p(x)\right)\cdot \underbrace{\int_{z}{q(z \vert x_c)\,dz}}_{=1}]$$

$$Eq_1 = \mathbb{E}_{c}[log\left(p(x)\right)] = log\left(p(x)\right)$$

&nbsp;

$$Eq_2 = \mathbb{E}_{c}\left[\int_{z}{q(z \vert x_c)\cdot \left[ log\left(q(z \vert x_c)\right) - log\left(p(z)\right) \right] \, dz}\right]$$

$$Eq2 = \mathbb{E}_{c}\left[D_{KL}\left( q(z \vert x_c) \parallel p(z)\right)\right]$$

&nbsp;

$$Eq_3 = \mathbb{E}_{c}\left[\int_{z}{q(z \vert x_c)\cdot log\left(p(x \vert z)\right)\,dz}\right]$$

$$Eq_3 = \mathbb{E}_{c}\left[ \mathbb{E}_{z \sim q(z \vert x_c)}[log\left(p(x \vert z)\right)] \right]$$

&nbsp;

We finally have:

$$\mathbb{E}_{c}[D_{KL}( q(z \vert x_c) \parallel p(z \vert x))] = $$

$$log\left(p(x)\right) + \mathbb{E}_{c}\left[D_{KL}\left( q(z \vert x_c) \parallel p(z)\right) - \mathbb{E}_{z \sim q(z \vert x_c)}[log\left(p(x \vert z)\right)] \right]$$

$$\mathbb{E}_{c}[D_{KL}( q(z \vert x_c) \parallel p(z \vert x))] + \mathcal{L}= log\left(p(x)\right)$$

where $$\mathcal{L}$$ is the ***Evidence Lower BOund (ELBO)***, whose expression is given by:

$$\mathcal{L} = \mathbb{E}_{c}\left[\mathbb{E}_{z \sim q(z \vert x_c)}[log\left(p(x \vert z)\right)] - D_{KL}\left( q(z \vert x_c) \parallel p(z)\right) \right]$$





