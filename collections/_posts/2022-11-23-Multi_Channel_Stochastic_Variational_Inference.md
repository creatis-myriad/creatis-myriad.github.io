---
layout: review
title: "Multi-Channel Stochastic Variational Inference for the Joint Analysis of Heterogeneous Biomedical Data in Alzheimer's Disease"
tags: fusion multimodality machine-Learning
author: "Romain Deleat-besson"
cite:
    authors: "Luigi Antelmi, Nicholas Ayache, Philippe Robert and Marco Lorenzi"
    title:   "Multi-Channel Stochastic Variational Inference for the Joint Analysis of Heterogeneous Biomedical Data in Alzheimer's Disease"
    venue:   "Machine Learning in Clinical Neuroimaging (MLCN), MICCAI Workshop 2018"
pdf: "https://arxiv.org/pdf/1808.03662.pdf"
---



# Notes

* Unsupervised method.
* **Gaussian linear case** (for validation purposes)
* Another paper based on deep learning methods was done in 2019 by the same authors: [Sparse Multi-Channel Variational Autoencoder](http://proceedings.mlr.press/v97/antelmi19a/antelmi19a.pdf).
* A [Github repository](https://gitlab.inria.fr/epione_ML/mcvae) is available for the 2019 paper.

* Link to the [VAE tutorial](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-vae.html) 


# Highlights

* The aim of this paper is to take into account several types of inputs (which are referred to as _channels_ in the paper) for a better joint analysis.
* They proposed a novel multi-channel stochastic generative model.
* Their proposed method can have applications for a generic data fusion technique.


# Introduction

When a patient suffer from Alzheimer’s disease, physicians use lots of various sources of informations. However, simple univariate correlation analyses are limited in modeling power. To overcome the limitations of mass-univariate analysis, methods such as PLS, RRR or CCA and their variants (non-linearity, multi-channels) were used in biomedical research. The main issue is that those methods are not generative.
The authors proposed a novel multi-channel stochastic generative model for the joint analysis of multi-channel heterogeneous data.


# Methods

Let $$ \textbf{x}  = \{\textbf{x}_c\}_{c=1}^{C} $$ be a single observation of a set of C channels, where each $$ \textbf{x}_c ∈ \mathbb{R}^{d_c} $$ is a $$ d_c $$-dimensional vector.

* $$ p(\textbf{z}) $$ : prior
* $$ p(\textbf{x}_c \vert \textbf{z}, \pmb{θ}_c) $$ : likelihood distribution
* $$ q(\textbf{z} \vert \textbf{x}_c, \pmb{\phi}_c) $$ : probability density function

The aim is to minimize:

$$ \underset{q∈Q}{argmin} \: \mathbb{E}_c[D_{KL} (q (\textbf{z} \vert \textbf{x}_c, \pmb{\phi}_c) \parallel p (\textbf{z} \vert \textbf{x}_1, . . . , \textbf{x}_C , \pmb{θ}_1, . . . , \pmb{θ}_C ) )] $$

**Hypothesis :** Every channel is conditionally independent from all the others given z (it will allow to factorize the data likelihood, see supplementary material in the paper for more details). 

After developping the equation from above they have :

$$ \mathscr{L}(\textbf{θ}, \pmb{\phi}, \textbf{x}) = \mathbb{E}_c \left[ \mathbb{E}_{q(\textbf{z} \vert \textbf{x}_c )} \left[  \sum_{i=1}^{C} ln \: p(\textbf{x}_i \vert \textbf{z}, \textbf{θ}_c) \right] − D_{KL} (q (\textbf{z} \vert \textbf{x}_c, \pmb{\phi}_c) \parallel p (\textbf{z}) ) \right] $$


A figure used in the [Github repository](https://gitlab.inria.fr/epione_ML/mcvae) to understand the global idea :

![](/collections/images/MCSVI/Multi_Latent_Space.jpg)


# Results

* They have created synthetic data (according to the equations shown below) and tested them with different parameters to prove the viability of their model.

$$ \textbf{z} ∼ N (\textbf{0}; \textbf{I}_l) $$

$$ \textbf{ϵ} ∼ N (\textbf{0}; \textbf{I}_{d_c} ) $$

$$ \textbf{G}_c = diag (\textbf{R}_c \textbf{R}_c{^T} ){^{-1/2}}\textbf{R}_c $$

$$ \textbf{x}_c = \textbf{G}_c \textbf{z} + snr{^{−1/2}}.\textbf{ϵ} $$

* Here are the results with synthetic data :

![](/collections/images/MCSVI/results_part_1.jpg)
It shows that their model perform well compared to single-channel model.


* They have tried it on real data as well :

![](/collections/images/MCSVI/results_part_2.jpg)

> Fig. 3: Pairwise representations of one latent dimension (out of 16) inferred from each of the four data channel. Although the optimization is not supervised to enforce clustering, subjects appear stratified by disease classes.


* An example of generated data :

![](/collections/images/MCSVI/results_part_3.jpg)
Having a generative model allows them to validate the latent space they obtained.
Moreover, according to typical results in the litterature for Alzheimer's Disease, the result they have is coherent.


# Conclusion

* Multi-channel model that jointly analyzes multimodality data.
* Model that can generate the data from the latent space
* Linear method (that can be easily changed with deep learning methods)


# References

The paper from 2019 with a deep learning and a more complex method :

[1] Antelmi, Luigi and Ayache, Nicholas and Robert, Philippe and Lorenzi, Marco. Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data (2019). PMLR. 


