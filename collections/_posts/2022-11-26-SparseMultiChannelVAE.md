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

&nbsp;

# Highlights

* The objective of this paper is to develop a formalism to handle with heterogenous data through VAE formulation.
* This is achieved through two major innovations: 

1) Variational distributions of each channel/modality/type of data are constrained to a common target prior in the latent space to bring interpretability;
> this can be seen as a process of alignment 

2) Parsimonious latent representations are enforced by ***variational dropout*** to make the method computationally advantageous and more easily interpretable.

&nbsp;

# Method

## Starting point

* Observations (e.g. patient information) are composed by subsets of information called channels as follows:

$$\textbf{x}=\{\textbf{x}_1,\cdots,\textbf{x}_C\}$$ 

$$\quad \quad$$ where each $$\textbf{x}_c$$ is a $$d_c$$-dimensional vector.

* $$\textbf{z}$$ is a $$l$$-dimensional latent variable which is supposed to be commonly shared by each $$\textbf{x}_c$$

* Every channel brings by itself some information about the latent variable distribution, the posterior $$p(\textbf{z} \vert \textbf{x})$$ can thus be approximated through the individual distributions $$q(\textbf{z} \vert \textbf{x}_c)$$

* Since each channel provides a different approximation, each $$q(\textbf{z} \vert \textbf{x}_c)$$ can be constrained to be as close as possible to the target posterior distribution by minimizing the following expression:

$$\mathbb{E}_{c}[D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z} \vert \textbf{x}_1,\cdots,\textbf{x}_C)\right)]$$

where $$\mathbb{E}_{c}$$ is the average over channels computed empirically.

The following figure illustrates the underlying concept:

![](/collections/images/smcvae/modeling_encoder_side.jpg)

&nbsp;

The minimization of the above equation is equivalent to the maximization of the following Evidence Lower BOund - ELBO (the corresponding demonstration is given in the appendix of this post):

$$\mathcal{L} = \mathbb{E}_{c}\left[\mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \textbf{x}_c)}[log\left(p(\textbf{x} \vert \textbf{z})\right)] - D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z})\right) \right]$$

* Making the hypothesis that every channels is conditionally independent from all others given $$\textbf{z}$$, we have:

$$p(\textbf{x} \vert \textbf{z})=\prod_{i=1}^{C}p(\textbf{x}_i \vert \textbf{z})$$

$$\mathcal{L}$$ can thus be rewritten as:

$$\mathcal{L} = \mathbb{E}_{c}\left[\mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \textbf{x}_c)}\left[\sum_{i=1}^{C}log\left(p(\textbf{x}_i \vert \textbf{z})\right)\right] - D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z})\right) \right]$$

&nbsp;

The minimization of $$\mathbb{E}_{c}[D_{KL}( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z} \vert \textbf{x}))]$$ is thus equivalent to the minimization of 

$$\mathcal{L^{*}} = \mathbb{E}_{c}\left[D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z})\right) - \mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \textbf{x}_c)}\left[\sum_{i=1}^{C}log\left(p(\textbf{x}_i \vert \textbf{z})\right)\right]\right]$$


The following figure illustrates the minimization of $$\mathcal{L^{*}}$$:

![](/collections/images/smcvae/core_minimization.jpg)

&nbsp;

# Appendix

## KL-divergence of the proposed model

$$\mathbb{E}_{c}[D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z} \vert \textbf{x}_1,\cdots,\textbf{x}_C)\right)]$$

$$=\mathbb{E}_{c}[D_{KL}( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z} \vert \textbf{x}))]$$

$$=\mathbb{E}_{c}\left[\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\cdot \left[ log\left(q(\textbf{z} \vert \textbf{x}_c)\right) - log\left(p(\textbf{z} \vert \textbf{x})\right) \right] \, dz}\right]$$

$$\text{using} \quad p(\textbf{z} \vert \textbf{x}) = \frac{p(\textbf{x} \vert \textbf{z}) \cdot p(\textbf{z})}{p(\textbf{x})}$$

$$=\mathbb{E}_{c}\left[\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\cdot \left[ log\left(q(\textbf{z} \vert \textbf{x}_c)\right) - log\left(p(\textbf{x} \vert \textbf{z})\right) - log\left(p(\textbf{z})\right) + log\left(p(\textbf{x})\right) \right] \, dz}\right]$$

$$=\underbrace{\mathbb{E}_{c}\left[\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\cdot log\left(p(\textbf{x})\right)\,dz}\right]}_{Eq_1} + \underbrace{\mathbb{E}_{c}\left[\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\cdot \left[ log\left(q(\textbf{z} \vert \textbf{x}_c)\right) - log\left(p(\textbf{z})\right) \right] \, dz}\right]}_{Eq_2}-\underbrace{\mathbb{E}_{c}\left[\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\cdot log\left(p(\textbf{x} \vert \textbf{z})\right)\,dz}\right]}_{Eq_3}$$

&nbsp;

$$Eq_1 = \mathbb{E}_{c}\left[\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\cdot log\left(p(\textbf{x})\right)\,dz}\right]$$

$$Eq_1 = \mathbb{E}_{c}[log\left(p(\textbf{x})\right)\cdot \underbrace{\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\,dz}}_{=1}]$$

$$Eq_1 = \mathbb{E}_{c}[log\left(p(\textbf{x})\right)] = log\left(p(\textbf{x})\right)$$

&nbsp;

$$Eq_2 = \mathbb{E}_{c}\left[\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\cdot \left[ log\left(q(\textbf{z} \vert \textbf{x}_c)\right) - log\left(p(\textbf{z})\right) \right] \, dz}\right]$$

$$Eq_2 = \mathbb{E}_{c}\left[D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z})\right)\right]$$

&nbsp;

$$Eq_3 = \mathbb{E}_{c}\left[\int_{\textbf{z}}{q(\textbf{z} \vert \textbf{x}_c)\cdot log\left(p(\textbf{x} \vert \textbf{z})\right)\,dz}\right]$$

$$Eq_3 = \mathbb{E}_{c}\left[ \mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \textbf{x}_c)}[log\left(p(\textbf{x} \vert \textbf{z})\right)] \right]$$

&nbsp;

We finally have:

$$\mathbb{E}_{c}[D_{KL}( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z} \vert \textbf{x}))] = $$

$$log\left(p(\textbf{x})\right) + \mathbb{E}_{c}\left[D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z})\right) - \mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \textbf{x}_c)}[log\left(p(\textbf{x} \vert \textbf{z})\right)] \right]$$

&nbsp;

This last equation can be rewritten as:

$$\mathbb{E}_{c}[D_{KL}( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z} \vert \textbf{x}))] + \mathcal{L}= log\left(p(\textbf{x})\right)$$

where $$\mathcal{L}$$ is the ***Evidence Lower BOund (ELBO)***, whose expression is given by:

$$\mathcal{L} = \mathbb{E}_{c}\left[\mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \textbf{x}_c)}[log\left(p(\textbf{x} \vert \textbf{z})\right)] - D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z})\right) \right]$$

&nbsp;

Since $$D_{KL}$$ is a measure of distance between two distributions, its value is $$\geq 0$$, which leads to the following relation:

$$\underbrace{\mathbb{E}_{c}[D_{KL}( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z} \vert \textbf{x}))]}_{\geq 0} + \underbrace{\mathcal{L}}_{\leq 0} = \underbrace{log\left(p(\textbf{x})\right)}_{\leq 0\text{ and fixed}}$$

> Thus, by tweaking $$\{q(\textbf{z} \vert \textbf{x}_c)\}_{c=1:C}$$, we can seek to maximize the ELBO $$\mathcal{L}$$, which will imply the minimization of the KL divergence 

&nbsp;

So the minimization of $$\mathbb{E}_{c}[D_{KL}( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z} \vert \textbf{x}))]$$ is equivalent to the maximization of $$\mathcal{L}$$, or the minimization of 

$$\mathcal{L^{*}} = \mathbb{E}_{c}\left[D_{KL}\left( q(\textbf{z} \vert \textbf{x}_c) \parallel p(\textbf{z})\right) - \mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \textbf{x}_c)}[log\left(p(\textbf{x} \vert \textbf{z})\right)]\right]$$





