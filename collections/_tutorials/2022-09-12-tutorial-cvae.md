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


