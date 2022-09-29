---
layout: post
title:  "What about the conditional variational autoencoder?"
author: 'Olivier Bernard'
date:   2022-09-28
categories: autoencoder, conditional, variational, VAE
---

# Notes

* This tutorial was mainly inspired by the following [paper1](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) and [paper2](https://proceedings.neurips.cc/paper/2018/file/473447ac58e1cd7e96172575f48dca3b-Paper.pdf).

&nbsp;

- [**Introduction**](#introduction)
  - [VAE](#vae)

&nbsp;

## **Introduction**

Conditional variational autoencoders (cVAE) should not been seen as an extension of conventional VAE! cVAE are also based on variational inference, but the overall objective is different: 
* In the VAE formalism, a pipeline is optimized to produce output as close as possible to the input data in order to build an efficient latent space with reduced dimensionality.
* In cVAE formalism, another pipeline is optimize to build a latent space that captures annotator variability.

### VAE

A complete tutorial on VAE can be found [here](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-vae.html). The graph below summarizes the overall strategy used in VAE formalism.

![](/collections/images/cvae/vae_final_representation.jpg)

In comparison, the graph below shows the overall strategy used in the CVAE formalism.

![](/collections/images/cvae/cvae_final_representation.jpg)

&nbsp;

## **Variational inference**

### Key concept

The goal of conditional variational autoencoder is to approximate a distribution $$p(y/x)$$ through a latent space that captures annotator variability such that $$p(y/x,z)$$ captures multiple plausible classiication/segmentation hypothesis. The following scheme is applied:
* for a given observation $$x$$, a set of latent variables $$z_i$$ is generated from a prior distribution $$p(z/x)$$ thanks to the sampling of the corresponding latent space.
* The set of latent variables are then combined with the observation and passed through the conditional generative process $$p(y/x,z)$$ to generate samples from the distribution $$y$$.
* The resulting predictive distribution is finally obtained through the following expression:

$$p(y/x) = \int_z{p(y/x,z) \cdot p(z/x) \,dz}$$

&nbsp;

Based on the Bayes' theorem, the estimation of predictive distribution $$p(y/x)$$ will require the computation of $$p(z/x,y)$$. Unfortunatly, it can be shown that this posterior distribution is intractable and that the computation of $$p(z/x,y)$$ can not be optimized directly. 

>The perdictive distribution $$p(y/x)$$ can be optimized through the evidence lower bound (ELBO), similar to the [VAE](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-vae.html).

&nbsp;

In the conditional VAE formalism, $$p(z/x,y)$$ is approximated by a Gaussian distribution $$q(z/x,y)$$ whose mean $$\mu_{post}$$ and covariance $$\sigma_{post}$$ are defined by two functions $$g(x,y)$$ and $$h(x,y)$$.

$$q(z/x,y) = \mathcal{N}\left(g(x,y),h(x,y)\right)$$


We thus have a family of candidates for variational inference and need to find the best approximation among this family by minimising the KL divergence between the approximation $$q(z/x,y)$$ and the target $$p(z/x,y)$$. In other words, we are looking for the optimal $$g^∗$$ and $$h^∗$$ such that:

$$\left(g^*,h^*\right) = \underset{(g,h)}{\arg\min} \,\,\, D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right)$$



