---
layout: review
title: "Multi-Channel Stochastic Variationnal Inference"
tags: "Fusion, Multimodality, Machine Learning"
author: "Romain Deleat-besson"
cite:
    authors: "Luigi Antelmi, Nicholas Ayache, Philippe Robert and Marco Lorenzi"
    title:   "Multi-Channel Stochastic Variational Inference for the Joint Analysis of Heterogeneous Biomedical Data in Alzheimer's Disease"
    venue:   "arXiv 2018"
pdf: "https://arxiv.org/pdf/1808.03662.pdf"
---



# Notes

* Unsupervised method.
* **Gaussian linear case** (for validation purposes)
* An other paper based on deep learning methods was done in 2019 by the same authors : [Sparse Multi Channal Variationnal Auto Encoder](http://proceedings.mlr.press/v97/antelmi19a/antelmi19a.pdf).
* A [Github repository](https://gitlab.inria.fr/epione_ML/mcvae) is available  for the 2019's paper.

* Link to the [VAE tutorial](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-vae.html) 


# Highlights

* The aim of this paper is to take into account several inputs for a better joint analysis.
* They proposed a novel Multi-channel stochastic generative model.
* Their proposed method can have applications for general data fusion technique.


# Introduction

When a patient suffer from the alzheimer’s Disease, physicians use lots of various sources of informations. However, simple univariate correlation analyses are limited in modeling power. To overcome the limitations of mass-univariate analysis, methods such as PLS, RRR or CCA and their variants (non-linearity, multi-channels) were used in biomedical research. The main issue is that those methods are not generative.
The authors proposed a novel multi-channel stochastic generative model for the joint analysis of multi-channel heterogeneous data.


# Methods

Let $$ x = \{x_c\}_{c=1}^{C} $$ be a single observation of a set of C channels, where each $$ x_c ∈ \mathbb{R}^{d_c} $$ is a $$ d_c $$-dimensional vector.

* $$ p(z) $$ : prior
* $$ p(x_c \vert z, θ_c) $$ : likelihood distribution
* $$ q(z \vert x_c, φ_c) $$ : probability density function

They aim at minimizing :

$$ \underset{q∈Q}{argmin} \: \mathbb{E}_c[D_{KL} (q (z \vert x_c, φ_c) \parallel p (z \vert x_1, . . . , x_C , θ_1, . . . , θ_C ) )] $$

**Hypothesis :** Every channel is conditionally independent from all the others given z (it will allow to factorize the data likelihood, see supplementary material in the paper for more details). 

After developping the equation from above they have :

$$ \mathscr{L} = \mathbb{E}_c \left[ \mathbb{E}_{q(z \vert x_c )} \left[  \sum_{i=1}^{C} ln \: p(x_i \vert z) ] − D_{KL} (q (z \vert x_c) \parallel p (z) ) \right] \right] $$


A figure used in the [Github repository](https://gitlab.inria.fr/epione_ML/mcvae) to understand the global idea :

![](/collections/images/MCSVI/Multi Latent Space.jpg)


# Results

* They have created synthetic data (according to the equations shown below) and tested them with different parameters to prove the viability of their model.

$$ z ∼ N (0; I_l) $$

$$ \epsilon ∼ N (0; I_{dc} ) $$

$$ G_c = diag (R_c R_c{^T} ){^{-1/2}}R_c $$

$$ x_c = G_c z + snr{^{−1/2}}.\epsilon $$

* Here are the results with synthetic datas :

![](/collections/images/MCSVI/results part 1.jpg)
It shows that their model perform well compared to single-channel model.


* They have tried it on real datas as well :

![](/collections/images/MCSVI/results part 2.jpg)


* An example of generated data :

![](/collections/images/MCSVI/results part 3.jpg)
According to typical results in the litterature for Alzheimer's Disease, the result they have is coherent.


# Conclusion

* Multi-channel model that jointly analyzes multimodality data.
* Model that can generate the datas from the latent space
* Linear method (that can be easily changed with deep learning methods)


# References

The paper from 2019 with a deep learning and a more complex method :

[1] Antelmi, Luigi and Ayache, Nicholas and Robert, Philippe and Lorenzi, Marco. Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data (2019). PMLR. 


