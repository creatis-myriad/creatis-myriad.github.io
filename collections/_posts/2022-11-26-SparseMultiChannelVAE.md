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

* The objective of this paper is to develop a formalism called sMSVAE (sparse multi-channel VAE) to handle with heterogenous data through VAE formulation.
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

## Derivation of the ELBO loss

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

## Reconstruction of missing channels

* The terms $$\mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \textbf{x}_c)}\left[\sum_{i=1}^{C}log\left(p(\textbf{x}_i \vert \textbf{z})\right)\right]$$  force each channel to the joint decoding of itself and every other channel
at the same time

* This property allows to reconstruct missing channels $$\{\hat{\textbf{x}}_i\}$$ for the available ones $$\{\tilde{\textbf{x}}_j\}$$ as:

$$\hat{\textbf{x}}_i = \mathbb{E}_j\left[\mathbb{E}_{\textbf{z} \sim q(\textbf{z} \vert \tilde{x}_c)}\left[\left(p(\textbf{x}_i \vert \textbf{z})\right)\right]\right]$$

&nbsp;

## Comparison with standard VAE

* In case of $$C=1$$, the proposed sMCVAE model is similar to a classical VAE. 

* sMCVAE is different from a VAE where all the channels are concatenated into a single one. In that case there cannot be missing channels if we want to infer the latent space variables

* sMCVAE is also different from a stack of $$C$$ independent VAEs, in which the $$C$$ latent spaces are no more related to each-other. 

> The dependence between encoding and decoding across channels stems from the joint approximation of the posterior distribution

&nbsp;

# Inducing sparse latent representation

## Motivations

* From simulations, the authors found that the lower bound $$\mathcal{L}$$ generally reaches the maximum value at convergence when the number of fitted latent dimensions coincide with the true one used to generate the data

* They also found that the performance of their method also depends on the effectiveness of the latent space dimensions chosen in relation to the application

* The authors proposed to solve this issue by automatically inferring the dimensions of the latent variables using a sparsity constraint on $$z$$ thanks to the strategy described hereafter

&nbsp;

## Regularization via dropout

TODO


&nbsp;

# Results

## Medical imaging data

The dataset is composed of

* 504 subjects of the Alzheimer's Disease Neuroimaging Initiative (ADNI) database
* Clinical channel is complsed by six continuous variables: age, results to mini-mental state examination, adas-cog, cdr, faq tests, scholarship level
* Three imaging channels: structural MRI, functional FDG-PET, Amyloid-PET. For each modality, 90 image intensities were computed from 90 brain regions mapped in the AAL atlas. This strategy produces 90 features arrays for each image. Lastly, data was centered and standardized across features. 

Their sMCVAE was compared with:

* MCVAE - their model without the sparsity constrain
* iVAEs - learning of independant VAE per channel
* VAE - learning a single VAE that takes as input all the channels at once

For each model class, multi-layer architectures were tested, ranging from 1 up to 4 layers for the encoding and decoding structures, with a sigmoid activation applied to all but last layer. 

After training, the latent space for each model was used to classify neurological diseases (MCI and Dementia) by means of ***Linear Discriminant Analysis***. During inference, as far the smcVAE or the mcVAE are concerned, each channel was used to compute a latent variable $$\textbf{z}_i$$ and the average value was calculated to populate the final latent space used for the classification task

For the sparse method, they selected the subspace generated by the most relevant latent dimensions identified by variational dropout $$(p<0.2)$$. Thanks to that, they identified 5 optimal latent dimensions

![](/collections/images/smcvae/medical_results_1.jpg)

&nbsp;

The encoding of the test set in the latent space given by our sMCVAE model is
depicted in the figure below where the visualization is limited to the 2D subspace generated by the two most relevant dimensions

![](/collections/images/smcvae/medical_results_2.jpg)

This subspace appears stratified by age and disease status, across roughly orthogonal directions. 





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





